"""
Evaluation metrics for CO3117 ML Assignment.

Common experimental protocol:
    - Primary metric: Macro-F1 (multiclass)
    - Secondary: Accuracy, Confusion Matrix
    - Always use the same metrics across all models for fair comparison.
"""

import numpy as np
from collections import Counter


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
    y_pred : np.ndarray, shape (n_samples,)

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Compute confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
    y_pred : np.ndarray, shape (n_samples,)
    n_classes : int or None
        Number of classes. If None, inferred from data.

    Returns
    -------
    np.ndarray, shape (n_classes, n_classes)
        C[i, j] = number of samples with true label i and predicted label j.
    """
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Compute per-class precision."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    col_sums = cm.sum(axis=0)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1
    return np.diag(cm) / col_sums


def recall_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Compute per-class recall."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return np.diag(cm) / row_sums


def f1_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Compute per-class F1 score."""
    p = precision_per_class(y_true, y_pred, n_classes)
    r = recall_per_class(y_true, y_pred, n_classes)
    denom = p + r
    denom[denom == 0] = 1
    return 2 * p * r / denom


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> float:
    """Compute macro-averaged F1 score (PRIMARY METRIC).

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
    y_pred : np.ndarray, shape (n_samples,)
    n_classes : int or None

    Returns
    -------
    float
        Macro-F1 in [0, 1].
    """
    f1s = f1_per_class(y_true, y_pred, n_classes)
    return float(np.mean(f1s))


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: dict | None = None,
) -> str:
    """Generate a text classification report.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    label_names : dict or None
        Mapping from label index to name.

    Returns
    -------
    str
        Formatted classification report.
    """
    classes = sorted(set(y_true) | set(y_pred))
    n_classes = max(classes) + 1

    p = precision_per_class(y_true, y_pred, n_classes)
    r = recall_per_class(y_true, y_pred, n_classes)
    f1 = f1_per_class(y_true, y_pred, n_classes)

    lines = []
    lines.append(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * 67)

    for c in classes:
        name = label_names.get(c, str(c)) if label_names else str(c)
        support = int(np.sum(y_true == c))
        lines.append(f"{name:<25} {p[c]:>10.4f} {r[c]:>10.4f} {f1[c]:>10.4f} {support:>10}")

    lines.append("-" * 67)
    lines.append(f"{'Macro Avg':<25} {np.mean(p[classes]):>10.4f} {np.mean(r[classes]):>10.4f} {macro_f1(y_true, y_pred, n_classes):>10.4f} {len(y_true):>10}")
    lines.append(f"{'Accuracy':<25} {'':>10} {'':>10} {accuracy(y_true, y_pred):>10.4f} {len(y_true):>10}")

    return "\n".join(lines)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: dict | None = None,
) -> dict:
    """Run the full evaluation protocol.

    Returns
    -------
    dict with keys: accuracy, macro_f1, confusion_matrix, report
    """
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, label_names),
    }
