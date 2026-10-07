"""Predict one English text with the saved baseline or the zero-shot model."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .zero_shot import classify, make_pipeline, read_config


LABEL_NAMES = {0: "sincere", 1: "sarcastic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("baseline", "zero-shot"))
    parser.add_argument("--text", required=True, help="English text to classify")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--prompt-id", default="selected", help="Zero-shot prompt id or 'selected'")
    args = parser.parse_args()

    if args.method == "baseline":
        model_path = Path(args.output_dir) / "svm.joblib"
        if not model_path.exists():
            raise SystemExit("Missing saved SVM. Run baseline validation first to select and save the model.")
        model = joblib.load(model_path)
        prediction = int(model.predict([args.text])[0])
        margin = float(model.decision_function([args.text])[0])
        result = {
            "method": "tfidf_linear_svm",
            "y_pred": prediction,
            "label": LABEL_NAMES[prediction],
            "decision_margin": margin,
            "score_note": "SVM decision margin; not a calibrated probability",
        }
    else:
        config = read_config()
        selected_path = Path(args.output_dir) / "selected_prompt.json"
        if args.prompt_id == "selected":
            if not selected_path.exists():
                raise SystemExit("Missing selected prompt. Run zero-shot validation first.")
            prompt_id = json.loads(selected_path.read_text(encoding="utf-8"))["id"]
        else:
            prompt_id = args.prompt_id
        try:
            prompt = next(item for item in config["prompts"] if item["id"] == prompt_id)
        except StopIteration as error:
            choices = ", ".join(item["id"] for item in config["prompts"])
            raise SystemExit(f"Unknown prompt {prompt_id!r}. Choose: {choices}, selected") from error
        frame = pd.DataFrame([{"sample_id": "single-input", "text": args.text, "label": 0}])
        pipe = make_pipeline(config)
        labels, scores, predictions = classify(pipe, frame, prompt, config)
        prediction = predictions[0]
        result = {
            "method": "zero_shot",
            "prompt_id": prompt_id,
            "selected_label": labels[0],
            "y_pred": prediction,
            "label": LABEL_NAMES[prediction],
            "selected_score": scores[0],
            "score_note": "NLI score normalized over candidate labels; not a calibrated probability",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
