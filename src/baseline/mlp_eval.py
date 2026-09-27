"""
W05 Experiment: MLP capacity and regularization on UCI HAR.

Compares:
  1. Custom MLP (from scratch) at different hidden-layer sizes (capacity)
  2. Effect of L2 regularization
  3. Benchmark against scikit-learn MLPClassifier
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data import get_data
from src.from_scratch.mlp import MLP


def run_experiment():
    print("Loading UCI HAR dataset (Normalized)...")
    data = get_data(normalize=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Remap labels from 1-6 to 0-5 for softmax indexing
    y_train_z = y_train - 1
    y_test_z = y_test - 1
    n_features = X_train.shape[1]
    n_classes = 6

    print(f"Features: {n_features}, Classes: {n_classes}")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print()

    # ------------------------------------------------------------------
    # Experiment 1: Capacity (varying hidden layer size)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Experiment 1: Capacity (hidden layer size)")
    print("=" * 60)

    configs = [
        ("Small  [32]",       [n_features, 32, n_classes]),
        ("Medium [128]",      [n_features, 128, n_classes]),
        ("Large  [256, 128]", [n_features, 256, 128, n_classes]),
    ]

    for name, layers in configs:
        mlp = MLP(
            layer_sizes=layers,
            activation="relu",
            learning_rate=0.01,
            max_iter=100,
            random_state=42,
            l2_lambda=0.0,
        )
        mlp.fit(X_train, y_train_z)

        train_pred = mlp.predict(X_train)
        test_pred = mlp.predict(X_test)

        train_acc = accuracy_score(y_train_z, train_pred)
        test_acc = accuracy_score(y_test_z, test_pred)
        test_f1 = f1_score(y_test_z, test_pred, average="macro")

        print(f"  {name}")
        print(f"    Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test Macro-F1: {test_f1:.4f}")
        print(f"    Final loss: {mlp.loss_history_[-1]:.4f}")
        print()

    # ------------------------------------------------------------------
    # Experiment 2: L2 Regularization
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Experiment 2: L2 Regularization (lambda)")
    print("=" * 60)

    lambdas = [0.0, 0.0001, 0.001, 0.01]
    for lam in lambdas:
        mlp = MLP(
            layer_sizes=[n_features, 128, n_classes],
            activation="relu",
            learning_rate=0.01,
            max_iter=100,
            random_state=42,
            l2_lambda=lam,
        )
        mlp.fit(X_train, y_train_z)

        train_pred = mlp.predict(X_train)
        test_pred = mlp.predict(X_test)

        train_acc = accuracy_score(y_train_z, train_pred)
        test_acc = accuracy_score(y_test_z, test_pred)
        gap = train_acc - test_acc

        print(f"  lambda={lam:.4f} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Gap: {gap:.4f}")

    print()

    # ------------------------------------------------------------------
    # Experiment 3: Benchmark against scikit-learn
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Experiment 3: Scikit-Learn MLPClassifier Benchmark")
    print("=" * 60)

    sk_mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="sgd",
        learning_rate_init=0.01,
        max_iter=100,
        random_state=42,
    )
    sk_mlp.fit(X_train, y_train)
    sk_pred = sk_mlp.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)
    sk_f1 = f1_score(y_test, sk_pred, average="macro")
    print(f"  Sklearn MLP (128) | Test Acc: {sk_acc:.4f} | Test Macro-F1: {sk_f1:.4f}")


if __name__ == "__main__":
    run_experiment()
