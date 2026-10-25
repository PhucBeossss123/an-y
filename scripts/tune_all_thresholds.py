"""Tune binary decision thresholds for the five classifiers on shared validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from an_y.data import load_shared_split, load_test, load_train
from an_y.metrics import summarize
from an_y.bert import EncodedTexts, _collate
from an_y.baseline import make_model
from an_y.zero_shot import read_config


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def encoder_scores(checkpoint: Path, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    config_path = checkpoint / "training_config.json"
    max_length = int(read_json(config_path).get("max_length", 128)) if config_path.exists() else 128
    loader = DataLoader(
        EncodedTexts(frame, tokenizer, max_length),
        batch_size=8,
        shuffle=False,
        collate_fn=_collate(tokenizer),
    )
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            outputs.append(torch.softmax(model(**batch).logits, dim=-1)[:, 1].cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(outputs).numpy()


def zero_shot_scores(frame: pd.DataFrame, output_dir: Path) -> tuple[np.ndarray, str]:
    config = read_config()
    prompt = read_json(output_dir / "selected_prompt.json")
    classifier = pipeline(
        "zero-shot-classification",
        model=config["model"],
        revision=config["model_revision"],
        device=0 if torch.cuda.is_available() else -1,
    )
    sarcastic_label = next(label for label, value in prompt["label_to_id"].items() if value == 1)
    scores = []
    texts = frame["text"].tolist()
    for start in range(0, len(texts), 32):
        results = classifier(
            texts[start : start + 32],
            candidate_labels=prompt["candidate_labels"],
            hypothesis_template=prompt["hypothesis_template"],
            multi_label=False,
            batch_size=config["batch_size"],
            truncation=True,
            max_length=config["hypothesis_max_length"],
        )
        for result in results:
            scores.append(dict(zip(result["labels"], result["scores"]))[sarcastic_label])
    return np.asarray(scores, dtype=np.float64), prompt["id"]


def candidate_thresholds(scores: np.ndarray, default: float, probability_score: bool) -> list[float]:
    if probability_score:
        values = np.arange(0.05, 0.951, 0.005).tolist()
    else:
        values = np.quantile(scores, np.linspace(0.01, 0.99, 197)).tolist()
    return sorted(set([float(default), *map(float, values)]))


def select_threshold(labels: np.ndarray, scores: np.ndarray, default: float, probability_score: bool):
    candidates = []
    for threshold in candidate_thresholds(scores, default, probability_score):
        metrics = summarize(labels, (scores >= threshold).astype(int))
        candidates.append((threshold, metrics))
    return max(
        candidates,
        key=lambda item: (
            item[1]["macro_f1"],
            item[1]["f1_sarcastic"],
            -abs(item[0] - default),
        ),
    )


def evaluate(frame: pd.DataFrame, scores: np.ndarray, threshold: float, path: Path) -> dict:
    predictions = (scores >= threshold).astype(int)
    pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str),
        "y_true": frame["label"].astype(int),
        "y_pred": predictions,
    }).to_csv(path, index=False)
    return summarize(frame["label"], predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--split-file", type=Path, default=ROOT / "data/splits/shared_split_seed42.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data/splits/manifest.json")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--phuc-output-dir", type=Path, default=ROOT / "outputs/shared_review_20261017")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/all_thresholds")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    fit, validation = load_shared_split(args.data_dir, args.split_file, args.split_manifest)
    test = load_test(args.data_dir) if args.evaluate_test else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = {}

    for name in ("bert", "roberta", "distilbert"):
        selection = read_json(args.outputs_dir / "team" / name / "selected_checkpoint.json")
        checkpoint = Path(selection["checkpoint"])
        validation_scores = encoder_scores(checkpoint, validation, device)
        threshold, validation_metrics = select_threshold(
            validation["label"].to_numpy(), validation_scores, 0.5, True
        )
        model_result = {
            "checkpoint": str(checkpoint),
            "selected_seed": selection["seed"],
            "decision_threshold": threshold,
            "validation": validation_metrics,
        }
        model_dir = args.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        evaluate(validation, validation_scores, threshold, model_dir / "validation_predictions.csv")
        print(f"{name}: validation threshold={threshold:.3f} {json.dumps(validation_metrics)}", flush=True)
        if test is not None:
            test_scores = encoder_scores(checkpoint, test, device)
            model_result["official_test"] = evaluate(
                test, test_scores, threshold, model_dir / "official_test_predictions.csv"
            )
            print(f"{name}: official test {json.dumps(model_result['official_test'])}", flush=True)
        outputs[name] = model_result

    svm_selection = read_json(args.phuc_output_dir / "baseline_validation_metrics.json")
    svm_class_weight = svm_selection["class_weight"]
    svm_model = make_model(
        c=float(svm_selection["C"]),
        class_weight=None if svm_class_weight == "none" else svm_class_weight,
    )
    svm_model.fit(fit["text"], fit["label"])
    svm_validation_scores = np.asarray(svm_model.decision_function(validation["text"]), dtype=np.float64)
    svm_threshold, svm_validation_metrics = select_threshold(
        validation["label"].to_numpy(), svm_validation_scores, 0.0, False
    )
    svm_result = {
        "C": svm_selection["C"],
        "class_weight": svm_class_weight,
        "decision_threshold": svm_threshold,
        "validation": svm_validation_metrics,
    }
    svm_dir = args.output_dir / "tfidf_svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    evaluate(validation, svm_validation_scores, svm_threshold, svm_dir / "validation_predictions.csv")
    print(f"tfidf_svm: validation threshold={svm_threshold:.4f} {json.dumps(svm_validation_metrics)}", flush=True)
    if test is not None:
        full_train = load_train(args.data_dir)
        svm_model.fit(full_train["text"], full_train["label"])
        svm_test_scores = np.asarray(svm_model.decision_function(test["text"]), dtype=np.float64)
        svm_result["official_test"] = evaluate(
            test, svm_test_scores, svm_threshold, svm_dir / "official_test_predictions.csv"
        )
        print(f"tfidf_svm: official test {json.dumps(svm_result['official_test'])}", flush=True)
    outputs["tfidf_svm"] = svm_result

    zero_shot_validation_scores, prompt_id = zero_shot_scores(validation, args.phuc_output_dir)
    zero_shot_threshold, zero_shot_validation_metrics = select_threshold(
        validation["label"].to_numpy(), zero_shot_validation_scores, 0.5, True
    )
    zero_shot_result = {
        "decision_threshold": zero_shot_threshold,
        "prompt_id": prompt_id,
        "validation": zero_shot_validation_metrics,
    }
    zero_shot_dir = args.output_dir / "zero_shot"
    zero_shot_dir.mkdir(parents=True, exist_ok=True)
    evaluate(
        validation,
        zero_shot_validation_scores,
        zero_shot_threshold,
        zero_shot_dir / "validation_predictions.csv",
    )
    print(f"zero_shot: validation threshold={zero_shot_threshold:.4f} {json.dumps(zero_shot_validation_metrics)}", flush=True)
    if test is not None:
        zero_shot_test_scores, _ = zero_shot_scores(test, args.phuc_output_dir)
        zero_shot_result["official_test"] = evaluate(
            test,
            zero_shot_test_scores,
            zero_shot_threshold,
            zero_shot_dir / "official_test_predictions.csv",
        )
        print(f"zero_shot: official test {json.dumps(zero_shot_result['official_test'])}", flush=True)
    outputs["zero_shot"] = zero_shot_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({
            "selection_split": "shared validation",
            "objective": "maximum Macro-F1, then sarcastic-class F1, then nearest default threshold",
            "score_semantics": "raw model score; not a calibrated probability",
            "models": outputs,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
