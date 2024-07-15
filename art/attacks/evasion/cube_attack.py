# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Cube Attack on XGBoost models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from typing import Optional, Tuple

from art.attacks.attack import EvasionAttack
from art.estimators.classification import XGBoostClassifier
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class CubeAttack(EvasionAttack):
    """
    Implementation of the Cube Attack on XGBoost models.
    """

    attack_params = ["eps", "n_trials", "p", "independent_delta", "min_val", "max_val"]
    _estimator_requirements = (XGBoostClassifier,)

    def __init__(
        self,
        estimator: XGBoostClassifier,
        eps: float = 0.1,
        n_trials: int = 100,
        p: float = 0.5,
        independent_delta: bool = False,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> None:
        """
        :param estimator: A trained XGBoost classifier.
        :param eps: The maximum perturbation.
        :param n_trials: The number of trials.
        :param p: The probability to change a coordinate.
        :param independent_delta: Whether to use independent deltas for each input sample.
        :param min_val: Minimum allowed value for features.
        :param max_val: Maximum allowed value for features.
        """
        super().__init__(estimator=estimator)
        self.eps = eps
        self.n_trials = n_trials
        self.p = p
        self.independent_delta = independent_delta
        self.min_val = min_val
        self.max_val = max_val
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array.

        :param x: Input samples.
        :param y: Target values (class labels).
        :return: Array of adversarial examples.
        """
        if y is None:
            y = self.estimator.predict(x)
        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)

        deltas = self._binary_search(self.cube_attack, self.estimator, x, y, self.n_trials)

        return x + deltas

    def cube_attack(self, f, X, y, eps, n_trials, p=0.5, deltas_init=None, independent_delta=False, min_val=0.0, max_val=1.0):
        """ A simple, but efficient black-box attack that just adds random steps of values in {-2eps, 0, 2eps}
        (i.e., the considered points are always corners). The random change is added if the loss decreases for a
        particular point. The only disadvantage of this method is that it will never find decision regions inside the
        Linf-ball which do not intersect any corner. But tight LRTE (compared to RTE/URTE) suggest that this doesn't happen.
            `f` is any function that has f.fmargin() method that returns class scores.
            `eps` can be a scalar or a vector of size X.shape[0].
            `min_val`, `max_val` are min/max allowed values for values in X (e.g. 0 and 1 for images). This can be adjusted
            depending on the feature range of the data. It's also possible to specify the as numpy vectors.
        """
        assert type(eps) is float or type(eps) is np.ndarray

        p_neg_eps = p/2  # probability of sampling -2eps
        p_pos_eps = p/2  # probability of sampling +2eps
        p_zero = 1 - p  # probability of not doing an update
        num, dim = X.shape
        # independent deltas work better for adv. training but slow down attacks
        size_delta = (num, dim) if independent_delta else (1, dim)

        if deltas_init is None:
            deltas_init = np.zeros(size_delta)
        # this init is important, s.t. there is no violation of bounds
        f_x_vals_min = self._fmargin(f, X, y) 

        if deltas_init is not None:  # evaluate the provided deltas and take them if they are better
            X_adv = np.clip(X + deltas_init, np.maximum(min_val, X - eps), np.minimum(max_val, X + eps))
            deltas = X_adv - X  # because of the projection above, the new delta vector is not just +-eps
            f_x_vals = self._fmargin(f, X_adv, y)
            idx_improved = f_x_vals < f_x_vals_min
            f_x_vals_min = idx_improved * f_x_vals + ~idx_improved * f_x_vals_min
            deltas = idx_improved[:, None] * deltas_init + ~idx_improved[:, None] * deltas

        else:
            deltas = deltas_init

        i_trial = 0
        while i_trial < n_trials:
            # +-2*eps is *very* important to escape local minima; +-eps has very unstable performance
            new_deltas = np.random.choice([-1, 0, 1], p=[p_neg_eps, p_zero, p_pos_eps], size=size_delta)
            new_deltas = 2 * eps * new_deltas  # if eps is a vector, then it's an outer product num x 1 times 1 x dim
            X_adv = np.clip(X + deltas + new_deltas, np.maximum(min_val, X - eps), np.minimum(max_val, X + eps))
            new_deltas = X_adv - X  # because of the projection above, the new delta vector is not just +-eps
            f_x_vals = self._fmargin(f, X_adv, y)
            idx_improved = f_x_vals < f_x_vals_min
            f_x_vals_min = idx_improved * f_x_vals + ~idx_improved * f_x_vals_min
            deltas = idx_improved[:, None] * new_deltas + ~idx_improved[:, None] * deltas
            i_trial += 1

        return f_x_vals_min, deltas

    def _binary_search(self, attack, f, X, y, n_trials_attack, cleanup=True):
        """
        Binary search to find the minimal perturbation that changes the class using `attack`.
        Supports a single eps only.
        """
        n_iter_bs = 10  # precision up to the 4th digit
        num, dim = X.shape
        deltas = np.zeros([num, dim])
        eps = np.ones((num, 1))
        eps_step = 1.0
        for i_iter_bs in range(n_iter_bs):
            f_x_vals, new_deltas = attack(f, X, y, eps, n_trials_attack, p=0.5, deltas_init=deltas, independent_delta=False)
            print('iter_bs {}: yf={}, eps={}'.format(i_iter_bs, f_x_vals, eps.flatten()))
            idx_adv = f_x_vals[:, None] < 0.0  # if adversarial, reduce the eps
            eps = idx_adv * (eps - eps_step/2) + ~idx_adv * (eps + eps_step/2)
            deltas = idx_adv * new_deltas + ~idx_adv * deltas
            eps_step /= 2

        yf = self._fmargin(f, X + deltas, y)
        print('yf after binary search: yf={}, Linf={}'.format(yf, np.abs(deltas).max(1)))
        if np.any(yf >= 0.0):
            print('The class was not changed (before cleanup)! Some bug apparently!')

        if cleanup:
            # If some eps/-eps do not change the prediction for a particular example, use delta_i = 0 instead.
            # Better for interpretability. Caution: needs num * dim function evaluations, thus advisable to use only
            # for visualizations, but not for LRTE.
            for i in range(dim):
                deltas_i_zeroed = np.copy(deltas)
                deltas_i_zeroed[:, i] = 0.0
                f_x_vals = self._fmargin(f, X + deltas_i_zeroed, y)
                idx_adv = f_x_vals < 0.0
                deltas = idx_adv[:, None] * deltas_i_zeroed + ~idx_adv[:, None] * deltas

        yf = self._fmargin(f, X + deltas, y)
        print('yf after cleanup: yf={}, Linf={}'.format(yf, np.abs(deltas).max(1)))
        if np.any(yf >= 0.0):
            print('The class was not changed (after cleanup)! Some bug apparently!')

        return deltas

    def _fmargin(self, f, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the functional margin for the input X and labels y.
        
        :param f: Estimator with a predict method.
        :param X: Input samples.
        :param y: True labels.
        :return: Functional margin values.
        """
        y_pred = f._model.predict(X, output_margin=True)
        y = y.reshape(-1)
        return np.exp(-y * y_pred)  # Functional margin: exp(-y * raw score)


    def _check_params(self) -> None:
        if self.eps <= 0:
            raise ValueError("The eps parameter must be strictly positive.")
        if self.n_trials <= 0:
            raise ValueError("The n_trials parameter must be strictly positive.")
        if not (0 <= self.p <= 1):
            raise ValueError("The p parameter must be in the range [0, 1].")
        if self.min_val >= self.max_val:
            raise ValueError("min_val must be less than max_val.")
