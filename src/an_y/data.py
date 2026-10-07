"""Small adapter for the official English Subtask A files."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_train(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "train.En.csv"
    frame = pd.read_csv(path)
    if "tweet" not in frame or "sarcastic" not in frame:
        raise ValueError(f"Unexpected training columns in {path}: {list(frame.columns)}")
    id_column = next((c for c in frame.columns if str(c).startswith("Unnamed:")), None)
    if id_column is None:
        raise ValueError("Expected the source row index column to preserve sample IDs")
    result = pd.DataFrame({
        "sample_id": frame[id_column].astype(str).map(lambda x: f"train:{x}"),
        "text": frame["tweet"].fillna("").astype(str),
        "label": frame["sarcastic"].astype(int),
    })
    _validate(result, "train")
    return result


def load_test(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "task_A_En_test.csv"
    frame = pd.read_csv(path)
    if "text" not in frame or "sarcastic" not in frame:
        raise ValueError(f"Unexpected test columns in {path}: {list(frame.columns)}")
    result = pd.DataFrame({
        "sample_id": [f"test:{i}" for i in range(len(frame))],
        "text": frame["text"].fillna("").astype(str),
        "label": frame["sarcastic"].astype(int),
    })
    _validate(result, "test")
    return result


def provisional_split(data_dir: str | Path, seed: int = 42):
    """Temporary stratified split until Huy supplies the shared split."""
    train = load_train(data_dir)
    fit, validation = train_test_split(
        train, test_size=0.2, random_state=seed, stratify=train["label"]
    )
    overlap = set(fit["text"]) & set(validation["text"])
    if overlap:
        warnings.warn(
            f"Found {len(overlap)} exact texts across provisional train/validation; "
            "they are reported, not removed.",
            RuntimeWarning,
        )
    return fit.reset_index(drop=True), validation.reset_index(drop=True)


def save_errors(frame: pd.DataFrame, predictions, path: str | Path) -> None:
    """Save actual mistakes with source text for manual, evidence-based review."""
    predicted = np.asarray(predictions)
    wrong = frame["label"].to_numpy() != predicted
    mistakes = frame.loc[wrong, ["sample_id", "text", "label"]].copy()
    mistakes["y_pred"] = predicted[wrong]
    mistakes = mistakes.rename(columns={"label": "y_true"})
    mistakes["analysis_note"] = ""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mistakes.to_csv(path, index=False)


def _validate(frame: pd.DataFrame, split: str) -> None:
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values in {split}")
    if frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"Missing text or label in {split}")
    labels = set(frame["label"].unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"Expected both binary labels 0/1 in {split}, got {labels}")
