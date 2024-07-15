# MIT License
# (License text omitted for brevity)
"""
This module implements the Sampling Attack for tabular data on Decision Tree-based models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from typing import Optional

from art.attacks.attack import EvasionAttack
from art.estimators.classification import XGBoostClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier, ScikitlearnRandomForestClassifier
from art.estimators.classification import ClassifierMixin
from art.utils import check_and_transform_label_format
from art.estimators.estimator import BaseEstimator

logger = logging.getLogger(__name__)

class SamplingAttack(EvasionAttack):
    """
    Implementation of the Sampling Attack for tabular data on Decision Tree-based models.
    """

    attack_params = ["eps", "n_trials","min_val","max_val"]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator,
        eps: float = 0.1,
        n_trials: int = 100,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> None:
        """
        :param estimator: A trained decision tree or tree ensemble model.
        :param eps: The maximum perturbation.
        :param n_trials: The number of trials.
        :param min_val: minimum value clip
        :param max_val: maximium value clip
        """
        super().__init__(estimator=estimator)
        self.eps = eps
        self.n_trials = n_trials
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

        adv_x = np.array([self._attack_single_sample(sample, label) for sample, label in zip(x, y)])
        return adv_x

    def _attack_single_sample(self, x: np.ndarray, y: int) -> np.ndarray:
        f_x_vals = np.zeros(self.n_trials)
        deltas = np.random.uniform(-self.eps, self.eps, size=(self.n_trials, x.shape[0]))

        for i in range(self.n_trials - 1):
            perturbed_pts = np.clip(x + deltas[i], self.min_val, self.max_val)
            f_x_vals[i] = self._fmargin(perturbed_pts, y)  # Ensure this is a scalar value

        f_x_vals[self.n_trials - 1] = self._fmargin(x, y)  # Ensure this is a scalar value
        idx_min = np.argmin(y * f_x_vals)
        delta = deltas[idx_min]

        return x + delta

    def _fmargin(self, x: np.ndarray, y: int) -> float:
        """
        Compute the functional margin for the input x and labels y.
        """
        y_pred = self.estimator.predict(x.reshape(1, -1))
        return float(y_pred[0, y].item())

    def _check_params(self) -> None:
        if self.eps <= 0:
            raise ValueError("The eps parameter must be strictly positive.")
        if self.n_trials <= 0:
            raise ValueError("The n_trials parameter must be strictly positive.")
