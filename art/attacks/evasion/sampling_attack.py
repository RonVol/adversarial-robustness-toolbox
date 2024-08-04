# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
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

# Copyright (c) 2019, Maksym Andriushchenko and Matthias Hein
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
A simple attack just by sampling in the Linf-box around the points. More of a sanity check.

Paper link: https://arxiv.org/abs/1906.03526
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from typing import Optional

from art.attacks.attack import EvasionAttack
from art.estimators.classification import ClassifierMixin
from art.utils import check_and_transform_label_format
from art.estimators.estimator import BaseEstimator

logger = logging.getLogger(__name__)

class SamplingAttack(EvasionAttack):
    """
    A simple attack just by sampling in the Linf-box around the points. More of a sanity check.

    Paper link: https://arxiv.org/abs/1906.03526
    """

    attack_params = ["eps", "n_trials","min_val","max_val"]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator,
        eps: float = 0.1,
        n_trials: int = 100
    ) -> None:
        """
        :param estimator: A trained decision tree or tree ensemble model.
        :param eps: The maximum perturbation.
        :param n_trials: The number of trials.
        """
        super().__init__(estimator=estimator)
        self.eps = eps
        self.n_trials = n_trials
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

        if self.estimator.clip_values is not None:
            self.clip_min, self.clip_max = self.estimator.clip_values
        else:
            self.clip_min, self.clip_max = np.min(x), np.max(x)

        adv_x = np.array([self._attack_single_sample(sample, label) for sample, label in zip(x, y)])
        return adv_x

    def _attack_single_sample(self, x: np.ndarray, y: int) -> np.ndarray:
        f_x_vals = np.zeros(self.n_trials)
        deltas = np.random.uniform(-self.eps, self.eps, size=(self.n_trials, x.shape[0]))

        for i in range(self.n_trials - 1):
            perturbed_pts = np.clip(x + deltas[i], self.clip_min, self.clip_max)
            f_x_vals[i] = self._fmargin(perturbed_pts, y)

        f_x_vals[self.n_trials - 1] = self._fmargin(x, y)
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
        
        if not isinstance(self.n_trials, int) or self.n_trials < 0:
            raise ValueError("The number of trials must be a non-negative integer.")
