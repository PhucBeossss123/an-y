"""Load only text and binary labels from official English Subtask A files."""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SHARED_SPLIT_FILE = Path("data/splits/shared_split_seed42.csv")
SHARED_SPLIT_MANIFEST = Path("data/splits/manifest.json")


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
    """Recreate the provisional split used by the archived Phuc experiments."""
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


def load_shared_split(
    data_dir: str | Path = "data/raw",
    split_file: str | Path = SHARED_SPLIT_FILE,
    manifest_file: str | Path = SHARED_SPLIT_MANIFEST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve committed train/validation IDs against the official training file."""
    split_path = Path(split_file)
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    for filename, record in manifest["source_files"].items():
        source_path = Path(data_dir) / filename
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Official source checksum mismatch: {source_path}")
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if digest != manifest["split_sha256"]:
        raise ValueError("Shared split checksum does not match its manifest")

    source = load_train(data_dir)
    split = pd.read_csv(split_path, dtype={"sample_id": str, "split": str})
    if list(split.columns) != ["sample_id", "split"]:
        raise ValueError("Shared split must contain only sample_id and split")
    if split["sample_id"].duplicated().any():
        raise ValueError("Shared split contains duplicate sample IDs")
    if set(split["split"].unique()) != {"train", "validation"}:
        raise ValueError("Shared split must define train and validation rows")

    source_ids = set(source["sample_id"])
    split_ids = set(split["sample_id"])
    if source_ids != split_ids:
        missing = sorted(source_ids - split_ids)[:5]
        extra = sorted(split_ids - source_ids)[:5]
        raise ValueError(
            f"Shared split IDs do not match source; missing={missing}, extra={extra}"
        )
    assignments = split.set_index("sample_id")["split"]
    row_splits = source["sample_id"].map(assignments)
    fit = source.loc[row_splits.eq("train")].reset_index(drop=True)
    validation = source.loc[row_splits.eq("validation")].reset_index(drop=True)
    return fit, validation


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
