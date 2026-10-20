"""Choose a RoBERTa decision threshold on validation, then optionally score test."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from an_y.data import load_shared_split, load_test
from an_y.metrics import summarize
from an_y.roberta import EncodedTexts, _collate


def score_frame(model, tokenizer, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        EncodedTexts(frame, tokenizer, 128),
        batch_size=8,
        shuffle=False,
        collate_fn=_collate(tokenizer),
    )
    logits = []
    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits.append(model(**batch).logits.cpu())
    return torch.softmax(torch.cat(logits), dim=-1)[:, 1].numpy()


def write_predictions(frame: pd.DataFrame, scores: np.ndarray, threshold: float, path: Path) -> dict:
    predictions = (scores >= threshold).astype(int)
    pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str),
        "y_true": frame["label"].astype(int),
        "y_pred": predictions,
    }).to_csv(path, index=False)
    return summarize(frame["label"].to_numpy(), predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/team/roberta/seed_42/best_checkpoint"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/shared_split_seed42.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/roberta_threshold"))
    parser.add_argument("--evaluate-test", action="store_true", help="Score official test only after selecting on validation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device).eval()
    _, validation = load_shared_split(args.data_dir, args.split_file, args.split_manifest)
    validation_scores = score_frame(model, tokenizer, validation, device)

    candidates = []
    for threshold in np.arange(0.30, 0.701, 0.005):
        metrics = summarize(validation["label"], (validation_scores >= threshold).astype(int))
        candidates.append((float(threshold), metrics))
    threshold, validation_metrics = max(
        candidates, key=lambda item: (item[1]["macro_f1"], item[1]["f1_sarcastic"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {"validation": validation_metrics}
    write_predictions(
        validation,
        validation_scores,
        threshold,
        args.output_dir / "validation_predictions.csv",
    )
    print(f"Selected threshold on validation: {threshold:.3f}")
    print(f"Validation metrics: {json.dumps(validation_metrics)}")

    if args.evaluate_test:
        test = load_test(args.data_dir)
        test_scores = score_frame(model, tokenizer, test, device)
        results["official_test"] = write_predictions(
            test,
            test_scores,
            threshold,
            args.output_dir / "official_test_predictions.csv",
        )
        print(f"Official test metrics: {json.dumps(results['official_test'])}")

    (args.output_dir / "metrics.json").write_text(
        json.dumps({
            "model": "RoBERTa-base",
            "checkpoint": str(args.checkpoint),
            "selection_split": "shared validation",
            "decision_threshold_on_sarcastic_score": threshold,
            "score_is_calibrated_probability": False,
            "metrics": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
