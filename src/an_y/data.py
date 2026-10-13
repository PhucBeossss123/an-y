"""Load only text and binary labels from official English Subtask A files."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_train(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "train.En.csv"
    source = pd.read_csv(path)
    if not {"tweet", "sarcastic"}.issubset(source.columns):
        raise ValueError(f"Missing tweet or sarcastic column in {path}")

    index_column = next(
        (name for name in source if str(name).startswith("Unnamed:")), None
    )
    if index_column is None:
        raise ValueError(f"Expected the source row index column in {path}")
    sample_ids = [f"train:{value}" for value in source[index_column].astype(str)]
    return _make_frame(sample_ids, source["tweet"], source["sarcastic"], "train")


def load_test(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "task_A_En_test.csv"
    source = pd.read_csv(path)
    if not {"text", "sarcastic"}.issubset(source.columns):
        raise ValueError(f"Missing text or sarcastic column in {path}")
    sample_ids = [f"test:{row}" for row in range(len(source))]
    return _make_frame(sample_ids, source["text"], source["sarcastic"], "test")


def provisional_split(
    data_dir: str | Path, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use a temporary stratified split until the shared split is delivered."""
    train = load_train(data_dir)
    fit_rows, validation_rows = train_test_split(
        train, test_size=0.2, random_state=seed, stratify=train["label"]
    )
    overlap = set(fit_rows["text"]) & set(validation_rows["text"])
    if overlap:
        warnings.warn(
            f"Found {len(overlap)} exact texts across provisional train/validation; "
            "they are reported, not removed.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fit_rows.reset_index(drop=True), validation_rows.reset_index(drop=True)


def save_errors(frame: pd.DataFrame, predicted_labels, path: str | Path) -> None:
    """Save model mistakes with IDs and source text for manual review."""
    predictions = np.asarray(predicted_labels)
    if len(predictions) != len(frame):
        raise ValueError("Prediction count does not match source rows")

    wrong = frame["label"].to_numpy() != predictions
    mistakes = frame.loc[wrong, ["sample_id", "text", "label"]].copy()
    mistakes["y_pred"] = predictions[wrong]
    mistakes = mistakes.rename(columns={"label": "y_true"})
    mistakes["analysis_note"] = ""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mistakes.to_csv(output_path, index=False)


def _make_frame(sample_ids, text_values, label_values, split: str) -> pd.DataFrame:
    labels = pd.to_numeric(pd.Series(label_values), errors="raise")
    observed = set(labels.dropna().unique())
    if labels.isna().any() or not observed.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1 in {split}, got {observed}")

    frame = pd.DataFrame({
        "sample_id": sample_ids,
        "text": pd.Series(text_values).fillna("").astype(str),
        "label": labels.astype(int),
    })
    _validate(frame, split)
    return frame


def _validate(frame: pd.DataFrame, split: str) -> None:
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values in {split}")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Expected both binary labels 0/1 in {split}")
