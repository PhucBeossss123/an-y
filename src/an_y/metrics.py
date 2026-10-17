"""Shared metrics for the binary sincere/sarcastic task."""

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


SINCERE = 0
SARCASTIC = 1


def summarize(y_true, y_pred) -> dict:
    """Return macro-F1, sarcastic-class F1, and per-class detail."""
    matrix = confusion_matrix(y_true, y_pred, labels=[SINCERE, SARCASTIC])
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_sarcastic": float(f1_score(
            y_true, y_pred, average="binary", pos_label=SARCASTIC, zero_division=0
        )),
        "precision_sincere": float(precision_score(
            y_true, y_pred, pos_label=SINCERE, zero_division=0
        )),
        "recall_sincere": float(recall_score(
            y_true, y_pred, pos_label=SINCERE, zero_division=0
        )),
        "precision_sarcastic": float(precision_score(
            y_true, y_pred, pos_label=SARCASTIC, zero_division=0
        )),
        "recall_sarcastic": float(recall_score(
            y_true, y_pred, pos_label=SARCASTIC, zero_division=0
        )),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }
