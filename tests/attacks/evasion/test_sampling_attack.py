import unittest
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification import XGBoostClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier, ScikitlearnRandomForestClassifier
from art.attacks.evasion import SamplingAttack
from art.estimators.classification import SklearnClassifier
import matplotlib.pyplot as plt

class TestSamplingAttack(unittest.TestCase):
    def setUp(self):
        data = load_breast_cancer()
        X_train, y_train = data.data[:400], data.target[:400]
        X_test, y_test = data.data[400:], data.target[400:]

        # Decision Tree
        self.dt_model = DecisionTreeClassifier().fit(X_train, y_train)
        self.dt_classifier = SklearnClassifier(model=self.dt_model, clip_values=(X_train.min(), X_train.max()))

        # Random Forest
        self.rf_model = RandomForestClassifier().fit(X_train, y_train)
        self.rf_classifier = SklearnClassifier(model=self.rf_model, clip_values=(X_train.min(), X_train.max()))

        # XGBoost
        self.xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
        self.xgb_classifier = XGBoostClassifier(model=self.xgb_model, clip_values=(X_train.min(), X_train.max()))

        self.X_test = X_test
        self.y_test = y_test

    def test_sampling_attack(self):
        self._run_attack(self.dt_classifier, "Decision Tree")
        self._run_attack(self.rf_classifier, "Random Forest")
        self._run_attack(self.xgb_classifier, "XGBoost")

    def _run_attack(self, classifier, model_name):
        # Evaluate the original accuracy
        original_preds = np.argmax(classifier.predict(self.X_test), axis=1)
        original_accuracy = accuracy_score(self.y_test, original_preds)
        print(f"Original Accuracy ({model_name}): {original_accuracy}")

        # Apply the Sampling Attack
        attack = SamplingAttack(classifier, eps=0.1, n_trials=10)
        X_test_adv = attack.generate(self.X_test, self.y_test)

        # Evaluate the adversarial accuracy
        adv_preds = np.argmax(classifier.predict(X_test_adv), axis=1)
        adversarial_accuracy = accuracy_score(self.y_test, adv_preds)
        print(f"Adversarial Accuracy ({model_name}): {adversarial_accuracy}")

        # Show an example of an adversarial example against its original
        print(f"Original Example ({model_name}):", self.X_test[0])
        print(f"Adversarial Example ({model_name}):", X_test_adv[0])

        # Calculate the differences
        differences = X_test_adv[0] - self.X_test[0]
        print(f"Differences ({model_name}):", differences)

        # Plot the differences
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Example Feature Values ({model_name})")
        plt.bar(range(len(self.X_test[0])), self.X_test[0])
        plt.yscale('log')  # Use a logarithmic scale to better visualize the differences
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')

        plt.subplot(1, 2, 2)
        plt.title(f"Adversarial Example Feature Values ({model_name})")
        plt.bar(range(len(X_test_adv[0])), X_test_adv[0])
        plt.yscale('log')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')

        plt.tight_layout()
        plt.show()

        # Print the prediction changes for a few examples
        for i in range(5):
            print(f"Original Prediction for example {i} ({model_name}): {original_preds[i]}")
            print(f"Adversarial Prediction for example {i} ({model_name}): {adv_preds[i]}")
            print(f"Original values ({model_name}): {self.X_test[i]}")
            print(f"Adversarial values ({model_name}): {X_test_adv[i]}")
            print(f"Differences ({model_name}): {X_test_adv[i] - self.X_test[i]}\n")

if __name__ == '__main__':
    unittest.main()
