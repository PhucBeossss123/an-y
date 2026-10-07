"""Cross-check saved predictions, IDs, and metric files against source labels."""

import json
from pathlib import Path

import joblib
import pandas as pd

from an_y.data import load_test, provisional_split
from an_y.metrics import summarize


def check_predictions(path: Path, expected, metrics_path: Path, prompt_id=None):
    pred = pd.read_csv(path)
    assert pred["sample_id"].tolist() == expected["sample_id"].tolist(), path.name
    assert pred["sample_id"].is_unique, path.name
    assert pred["y_true"].tolist() == expected["label"].tolist(), path.name
    computed = summarize(pred["y_true"], pred["y_pred"])
    stored_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if isinstance(stored_payload, list):
        stored = next(item for item in stored_payload if item["prompt_id"] == prompt_id)
    else:
        stored = stored_payload
    for key in ("macro_f1", "f1_sarcastic", "precision_sincere", "recall_sincere", "precision_sarcastic", "recall_sarcastic"):
        assert abs(computed[key] - stored[key]) < 1e-12, f"{path.name}: {key}"
    return len(pred)


def main():
    output = Path("outputs")
    _, validation = provisional_split("data/raw")
    test = load_test("data/raw")
    checks = {
        "baseline_validation": check_predictions(output / "baseline_validation_predictions.csv", validation, output / "baseline_validation_metrics.json"),
        "baseline_official_test": check_predictions(output / "baseline_test_predictions.csv", test, output / "baseline_test_metrics.json"),
    }
    baseline_saved = pd.read_csv(output / "baseline_test_predictions.csv")
    reloaded_model = joblib.load(output / "svm.joblib")
    reloaded_pred = reloaded_model.predict(test["text"])
    assert reloaded_pred.tolist() == baseline_saved["y_pred"].tolist(), "reloaded SVM predictions"
    checks["reloaded_svm_test"] = len(reloaded_pred)
    prompt_config = json.loads(Path("config/prompts.json").read_text(encoding="utf-8"))
    prompt_map = {prompt["id"]: prompt["label_to_id"] for prompt in prompt_config["prompts"]}
    for prompt in ("original", "statement", "explicit_negation", "social_post", "ironic_or_literal"):
        pred_file = output / f"zero_shot_provisional_validation_seed_42_{prompt}_predictions.csv"
        if not pred_file.exists():
            # Keep the saved name stable even when run() receives a descriptive split label.
            pred_file = output / f"zero_shot_provisional_validation_seed_42_{prompt}_predictions.csv"
        checks[f"zero_shot_validation_{prompt}"] = check_predictions(
            pred_file,
            validation,
            output / "zero_shot_validation_metrics.json",
            prompt,
        )
        validation_predictions = pd.read_csv(pred_file)
        mapped = validation_predictions["selected_label"].map(prompt_map[prompt])
        assert mapped.tolist() == validation_predictions["y_pred"].tolist(), f"validation label map: {prompt}"
    for prompt in ("original", "explicit_negation"):
        checks[f"zero_shot_test_{prompt}"] = check_predictions(
            output / f"zero_shot_official_test_{prompt}_predictions.csv",
            test,
            output / "zero_shot_test_metrics.json",
            prompt,
        )
        test_predictions = pd.read_csv(output / f"zero_shot_official_test_{prompt}_predictions.csv")
        mapped = test_predictions["selected_label"].map(prompt_map[prompt])
        assert mapped.tolist() == test_predictions["y_pred"].tolist(), f"test label map: {prompt}"
    selected = json.loads((output / "selected_prompt.json").read_text(encoding="utf-8"))
    validation_results = json.loads((output / "zero_shot_validation_metrics.json").read_text(encoding="utf-8"))
    best = max(validation_results, key=lambda item: (item["macro_f1"], item["f1_sarcastic"]))
    assert selected["id"] == best["prompt_id"], (selected["id"], best["prompt_id"])
    print(json.dumps({"status": "passed", "rows_checked": checks}, indent=2))


if __name__ == "__main__":
    main()
