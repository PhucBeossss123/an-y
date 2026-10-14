"""Small adapter for the official English Subtask A files."""

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
