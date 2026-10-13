"""Compare fixed NLI prompts and select one using validation only."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

from .data import load_test, provisional_split, save_errors
from .metrics import summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "prompts.json"
TEXT_BATCH_SIZE = 32
PROGRESS_INTERVAL = 128


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read the fixed checkpoint and prompt definitions."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, values) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def make_pipeline(config: dict):
    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda:0" if device == 0 else "cpu"
    print(f"Zero-shot device: {device_name}")
    return pipeline(
        "zero-shot-classification",
        model=config["model"],
        revision=config["model_revision"],
        device=device,
    )


def classify(pipe, texts: list[str], prompt: dict, config: dict):
    """Return the winning label, its NLI score, and binary class for each text."""
    candidates = prompt["candidate_labels"]
    label_to_id = prompt["label_to_id"]
    if set(label_to_id) != set(candidates) or set(label_to_id.values()) != {0, 1}:
        raise ValueError(f"Invalid binary label mapping in prompt {prompt['id']}")

    chosen_labels = []
    chosen_scores = []
    predictions = []

    for start in range(0, len(texts), TEXT_BATCH_SIZE):
        batch_texts = texts[start : start + TEXT_BATCH_SIZE]
        batch_results = pipe(
            batch_texts,
            candidate_labels=candidates,
            hypothesis_template=prompt["hypothesis_template"],
            multi_label=False,
            batch_size=config["batch_size"],
            truncation=True,
            max_length=config["hypothesis_max_length"],
        )
        for result in batch_results:
            scores_by_label = dict(zip(result["labels"], result["scores"]))
            missing_labels = set(candidates) - set(scores_by_label)
            if missing_labels:
                raise ValueError(
                    f"Model omitted candidate labels in prompt {prompt['id']}: "
                    f"{sorted(missing_labels)}"
                )

            selected_label = max(candidates, key=scores_by_label.__getitem__)
            chosen_labels.append(selected_label)
            chosen_scores.append(float(scores_by_label[selected_label]))
            predictions.append(int(label_to_id[selected_label]))

        processed = start + len(batch_texts)
        if processed % PROGRESS_INTERVAL < len(batch_texts) or processed == len(texts):
            print(f"  {prompt['id']}: scored {processed}/{len(texts)}", flush=True)

    return chosen_labels, chosen_scores, predictions


def run_prompts(
    frame: pd.DataFrame,
    prompts: list[dict],
    config: dict,
    output_dir: Path,
    split_name: str,
) -> list[dict]:
    """Score prompts and save one prediction and error file per prompt."""
    classifier = make_pipeline(config)
    summaries = []

    for prompt in prompts:
        labels, scores, predicted_labels = classify(
            classifier,
            frame["text"].tolist(),
            prompt,
            config,
        )
        summary = {
            "method": "zero_shot",
            "prompt_id": prompt["id"],
            "split": split_name,
            "checkpoint": config["model"],
            "candidate_labels": prompt["candidate_labels"],
            "hypothesis_template": prompt["hypothesis_template"],
            "score_semantics": (
                "NLI scores normalized over the candidate labels; "
                "not calibrated probabilities"
            ),
            **summarize(frame["label"], predicted_labels),
        }
        summaries.append(summary)

        prediction_rows = pd.DataFrame(
            {
                "sample_id": frame["sample_id"],
                "y_true": frame["label"],
                "y_pred": predicted_labels,
                "selected_label": labels,
                "selected_score": scores,
            }
        )
        prediction_path = (
            output_dir / f"zero_shot_{split_name}_{prompt['id']}_predictions.csv"
        )
        prediction_rows.to_csv(prediction_path, index=False)
        save_errors(
            frame,
            predicted_labels,
            output_dir / f"zero_shot_{split_name}_{prompt['id']}_errors.csv",
        )
        print(json.dumps(summary, indent=2))

    return summaries


def run_validation(data_dir: str, output_dir: Path, config: dict) -> None:
    _, validation = provisional_split(data_dir)
    summaries = run_prompts(
        validation,
        config["prompts"],
        config,
        output_dir,
        "provisional_validation_seed_42",
    )
    summaries.sort(
        key=lambda result: (result["macro_f1"], result["f1_sarcastic"]),
        reverse=True,
    )

    write_json(output_dir / "zero_shot_validation_metrics.json", summaries)
    prompt_by_id = {prompt["id"]: prompt for prompt in config["prompts"]}
    selected_prompt = prompt_by_id[summaries[0]["prompt_id"]]
    write_json(output_dir / "selected_prompt.json", selected_prompt)
    print(f"Selected on validation only: {selected_prompt['id']}")


def run_test(data_dir: str, output_dir: Path, config: dict) -> None:
    selected_path = output_dir / "selected_prompt.json"
    if not selected_path.exists():
        raise SystemExit(
            "Run zero-shot validation first; prompt selection must precede test evaluation."
        )

    selected_prompt = json.loads(selected_path.read_text(encoding="utf-8"))
    prompt_by_id = {prompt["id"]: prompt for prompt in config["prompts"]}
    if selected_prompt["id"] not in prompt_by_id:
        raise SystemExit("Saved prompt is not present in config/prompts.json.")

    original_prompt = prompt_by_id["original"]
    test = load_test(data_dir)
    prompts_to_test = [original_prompt]
    if selected_prompt["id"] != original_prompt["id"]:
        prompts_to_test.append(prompt_by_id[selected_prompt["id"]])

    summaries = run_prompts(
        test,
        prompts_to_test,
        config,
        output_dir,
        "official_test",
    )
    write_json(output_dir / "zero_shot_test_metrics.json", summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "test", "smoke"))
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_config()

    if args.action == "smoke":
        _, validation = provisional_split(args.data_dir)
        original_prompt = next(
            prompt for prompt in config["prompts"] if prompt["id"] == "original"
        )
        run_prompts(
            validation.head(8).reset_index(drop=True),
            [original_prompt],
            config,
            output_dir,
            "smoke_only_not_results",
        )
    elif args.action == "validate":
        run_validation(args.data_dir, output_dir, config)
    else:
        run_test(args.data_dir, output_dir, config)


if __name__ == "__main__":
    main()
