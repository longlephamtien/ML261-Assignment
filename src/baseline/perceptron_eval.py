import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data import get_data
from src.from_scratch.perceptron import Perceptron as CustomPerceptron

def main():
    print("Loading UCI HAR dataset (Normalized)...")
    data = get_data(normalize=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Filter for binary classification: Class 1 (Walking) vs Class 6 (Laying)
    # The Perceptron is inherently a binary classifier.
    train_mask = np.isin(y_train, [1, 6])
    test_mask = np.isin(y_test, [1, 6])
    
    X_train_bin = X_train[train_mask]
    y_train_bin = np.where(y_train[train_mask] == 1, 1, 0)
    
    X_test_bin = X_test[test_mask]
    y_test_bin = np.where(y_test[test_mask] == 1, 1, 0)

    print(f"Binary Dataset Size - Train: {len(X_train_bin)}, Test: {len(X_test_bin)}")
    
    print("-" * 50)
    print("Training Custom Perceptron (from scratch)...")
    custom_ppn = CustomPerceptron(learning_rate=0.01, max_iter=20, random_state=42)
    custom_ppn.fit(X_train_bin, y_train_bin)
    
    train_acc_custom = accuracy_score(y_train_bin, custom_ppn.predict(X_train_bin))
    test_acc_custom = accuracy_score(y_test_bin, custom_ppn.predict(X_test_bin))
    print(f"Custom Perceptron - Train Accuracy: {train_acc_custom:.4f}")
    print(f"Custom Perceptron - Test Accuracy:  {test_acc_custom:.4f}")
    print(f"Misclassifications per epoch: {custom_ppn.errors_}")

    print("-" * 50)
    print("Training Scikit-Learn Perceptron (benchmark)...")
    sk_ppn = SklearnPerceptron(eta0=0.01, max_iter=20, random_state=42)
    sk_ppn.fit(X_train_bin, y_train_bin)
    
    train_acc_sk = accuracy_score(y_train_bin, sk_ppn.predict(X_train_bin))
    test_acc_sk = accuracy_score(y_test_bin, sk_ppn.predict(X_test_bin))
    print(f"Sklearn Perceptron - Train Accuracy: {train_acc_sk:.4f}")
    print(f"Sklearn Perceptron - Test Accuracy:  {test_acc_sk:.4f}")

if __name__ == "__main__":
    main()
