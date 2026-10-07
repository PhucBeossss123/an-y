"""Shared-formula metric helper used until Tuấn's evaluator is delivered."""

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def summarize(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_sarcastic": float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "precision_sincere": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_sincere": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_sarcastic": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_sarcastic": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }
