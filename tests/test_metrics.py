"""
Tests for evaluation metrics.

These verify that the from-scratch metrics produce correct results
before using them to evaluate models.
"""

import numpy as np
import pytest
from src.metrics import accuracy, macro_f1, confusion_matrix, f1_per_class


class TestAccuracy:
    def test_perfect_prediction(self):
        y = np.array([1, 2, 3, 1, 2])
        assert accuracy(y, y) == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([2, 2, 2])
        assert accuracy(y_true, y_pred) == 0.0

    def test_partial(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 0, 0])
        assert accuracy(y_true, y_pred) == 0.5


class TestConfusionMatrix:
    def test_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred, n_classes=2)
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(cm, expected)

    def test_perfect(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        cm = confusion_matrix(y_true, y_pred, n_classes=3)
        np.testing.assert_array_equal(cm, np.eye(3, dtype=int))


class TestMacroF1:
    def test_perfect(self):
        y = np.array([1, 2, 3, 1, 2, 3])
        assert macro_f1(y, y) == pytest.approx(1.0)

    def test_zero(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([2, 2, 2])
        assert macro_f1(y_true, y_pred) == pytest.approx(0.0)
