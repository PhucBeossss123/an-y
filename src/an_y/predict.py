"""Predict one English text with the saved baseline or a zero-shot prompt."""

import argparse
import json
from pathlib import Path

import joblib

from .bert import predict_bert
from .distilbert import predict_distilbert
from .roberta import predict_roberta
from .zero_shot import classify, make_pipeline, read_config


LABEL_NAMES = {0: "sincere", 1: "sarcastic"}


def predict_baseline(text: str, output_dir: Path) -> dict:
    model_path = output_dir / "svm.joblib"
    if not model_path.exists():
        raise SystemExit(
            "Missing saved SVM. Run baseline validation first to select and save the model."
        )

    model = joblib.load(model_path)
    prediction = int(model.predict([text])[0])
    margin = float(model.decision_function([text])[0])
    return {
        "method": "tfidf_linear_svm",
        "y_pred": prediction,
        "label": LABEL_NAMES[prediction],
        "decision_margin": margin,
        "score_note": "SVM decision margin; not a calibrated probability",
    }


def choose_prompt(config: dict, prompt_id: str, output_dir: Path) -> dict:
    if prompt_id == "selected":
        selected_path = output_dir / "selected_prompt.json"
        if not selected_path.exists():
            raise SystemExit("Missing selected prompt. Run zero-shot validation first.")
        prompt_id = json.loads(selected_path.read_text(encoding="utf-8"))["id"]

    for prompt in config["prompts"]:
        if prompt["id"] == prompt_id:
            return prompt

    choices = ", ".join(prompt["id"] for prompt in config["prompts"])
    raise SystemExit(f"Unknown prompt {prompt_id!r}. Choose: {choices}, selected")


def predict_zero_shot(
    text: str, prompt_id: str, output_dir: Path
) -> dict:
    config = read_config()
    prompt = choose_prompt(config, prompt_id, output_dir)
    classifier = make_pipeline(config)
    selected_labels, selected_scores, predicted_labels = classify(
        classifier,
        [text],
        prompt,
        config,
    )

    prediction = predicted_labels[0]
    return {
        "method": "zero_shot",
        "prompt_id": prompt["id"],
        "selected_label": selected_labels[0],
        "y_pred": prediction,
        "label": LABEL_NAMES[prediction],
        "selected_score": selected_scores[0],
        "score_note": "NLI score; not a calibrated probability",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method", choices=("baseline", "zero-shot", "bert", "roberta", "distilbert")
    )
    parser.add_argument("--text", required=True, help="English text to classify")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint", help="Fine-tuned transformer checkpoint directory")
    parser.add_argument(
        "--prompt-id",
        default="selected",
        help="Zero-shot prompt id or 'selected'",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.method == "baseline":
        result = predict_baseline(args.text, output_dir)
    elif args.method == "zero-shot":
        result = predict_zero_shot(args.text, args.prompt_id, output_dir)
    else:
        model_dir = {
            "bert": "bert",
            "roberta": "roberta",
            "distilbert": "distilbert",
        }[args.method]
        if args.checkpoint:
            checkpoint = Path(args.checkpoint)
        else:
            selection_path = Path("outputs/team") / model_dir / "selected_checkpoint.json"
            if not selection_path.exists():
                raise SystemExit(
                    f"Missing {selection_path}; run scripts/package_team_results.py first."
                )
            checkpoint = Path(
                json.loads(selection_path.read_text(encoding="utf-8"))["checkpoint"]
            )
        if not checkpoint.exists():
            raise SystemExit(
                f"Missing fine-tuned checkpoint {checkpoint}; train the model or pass --checkpoint."
            )
        predictor = {
            "bert": predict_bert,
            "roberta": predict_roberta,
            "distilbert": predict_distilbert,
        }[args.method]
        result = predictor(args.text, checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
