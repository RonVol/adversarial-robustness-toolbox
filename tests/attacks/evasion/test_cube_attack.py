# tests/attacks/evasion/test_cube_attack.py

import unittest
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import CubeAttack

class TestCubeAttack(unittest.TestCase):
    def setUp(self):
        data = load_breast_cancer()
        X_train, y_train = data.data[:400], data.target[:400]
        X_test, y_test = data.data[400:], data.target[400:]

        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        self.classifier = XGBoostClassifier(model=model)
        self.X_test = X_test
        self.y_test = y_test

    def test_cube_attack(self):
        # Evaluate the original accuracy
        original_preds = np.argmax(self.classifier.predict(self.X_test), axis=1)
        original_accuracy = accuracy_score(self.y_test, original_preds)
        print(f"Original Accuracy: {original_accuracy}")

        # Apply the Cube Attack
        attack = CubeAttack(self.classifier, eps=0.1, n_trials=10)
        X_test_adv = attack.generate(self.X_test, self.y_test)

        # Evaluate the adversarial accuracy
        adv_preds = np.argmax(self.classifier.predict(X_test_adv), axis=1)
        adversarial_accuracy = accuracy_score(self.y_test, adv_preds)
        print(f"Adversarial Accuracy: {adversarial_accuracy}")

        # Show an example of an adversarial example against its original
        print("Original:", self.X_test[0])
        print("Adversarial:", X_test_adv[0])

        # Plot the original and adversarial examples
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Example")
        plt.imshow(self.X_test[0].reshape((15, 2)), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Adversarial Example")
        plt.imshow(X_test_adv[0].reshape((15, 2)), cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()
