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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from sklearn.tree import DecisionTreeClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from sklearn.datasets import load_breast_cancer
import numpy as np

from art.attacks.evasion.sampling_attack import SamplingAttack
from art.estimators.classification.scikitlearn import SklearnClassifier

from tests.utils import TestBase, master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestSamplingAttack(TestBase):
    """
    A unittest class for testing the SamplingAttack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        # Get Digits dataset
        data = load_breast_cancer()
        cls.X = data.data
        cls.y = data.target

    def test_scikitlearn(self):
        clf = DecisionTreeClassifier()
        x_original = self.X.copy()
        clf.fit(self.X, self.y)
        clf_art = SklearnClassifier(clf)
        attack = SamplingAttack(estimator=clf_art, eps=0.1, n_trials=10)
        adv = attack.generate(self.X[:25])
        # Ensure some crafting succeeded
        self.assertTrue(np.sum(clf.predict(adv) != clf.predict(self.X[:25])) > 0)
        
        # Check that X has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_original - self.X))), 0.0, delta=0.00001)

    def test_check_params(self):
        clf = DecisionTreeClassifier()
        clf.fit(self.X, self.y)
        clf_art = SklearnClassifier(clf)

        with self.assertRaises(ValueError):
            _ = SamplingAttack(estimator=clf_art, eps=-0.1)

        with self.assertRaises(ValueError):
            _ = SamplingAttack(estimator=clf_art, n_trials=-1)

    def test_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(SamplingAttack, [BaseEstimator, ClassifierMixin])


if __name__ == "__main__":
    unittest.main()
