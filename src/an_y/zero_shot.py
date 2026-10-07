"""Evaluate five fixed NLI prompts, choosing only on validation."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

from .data import load_test, provisional_split, save_errors
from .metrics import summarize


def read_config():
    return json.loads(Path("config/prompts.json").read_text(encoding="utf-8"))


def make_pipeline(config):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Zero-shot device: {'cuda:0' if device == 0 else 'cpu'}")
    return pipeline(
        "zero-shot-classification",
        model=config["model"],
        revision=config["model_revision"],
        device=device,
    )


def classify(pipe, frame, prompt, config):
    results = []
    texts = frame["text"].tolist()
    chunk_size = 32
    for start in range(0, len(texts), chunk_size):
        results.extend(pipe(
            texts[start:start + chunk_size],
            candidate_labels=prompt["candidate_labels"],
            hypothesis_template=prompt["hypothesis_template"],
            multi_label=False,
            batch_size=config["batch_size"],
            truncation=True,
            max_length=config["hypothesis_max_length"],
        ))
        if (start + chunk_size) % 128 == 0 or start + chunk_size >= len(texts):
            print(f"  {prompt['id']}: scored {min(start + chunk_size, len(texts))}/{len(texts)}", flush=True)
    labels, scores, preds = [], [], []
    mapping = prompt["label_to_id"]
    if set(mapping) != set(prompt["candidate_labels"]):
        raise ValueError(f"Incomplete explicit label mapping in prompt {prompt['id']}")
    for result in results:
        score_by_label = dict(zip(result["labels"], result["scores"]))
        chosen = max(prompt["candidate_labels"], key=score_by_label.__getitem__)
        labels.append(chosen)
        scores.append(float(score_by_label[chosen]))
        preds.append(int(mapping[chosen]))
    return labels, scores, preds


def run(frame, prompts, config, output, split):
    pipe = make_pipeline(config)
    summaries = []
    for prompt in prompts:
        label_names, scores, preds = classify(pipe, frame, prompt, config)
        result = {
            "method": "zero_shot",
            "prompt_id": prompt["id"],
            "split": split,
            "checkpoint": config["model"],
            "candidate_labels": prompt["candidate_labels"],
            "hypothesis_template": prompt["hypothesis_template"],
            "score_semantics": "mutually exclusive NLI scores normalized over the two candidate labels; not calibrated probabilities",
            **summarize(frame["label"], preds),
        }
        summaries.append(result)
        pred_frame = pd.DataFrame({
            "sample_id": frame["sample_id"],
            "text": frame["text"],
            "y_true": frame["label"],
            "y_pred": preds,
            "selected_label": label_names,
            "selected_score": scores,
        })
        pred_frame.drop(columns="text").to_csv(output / f"zero_shot_{split}_{prompt['id']}_predictions.csv", index=False)
        save_errors(frame, preds, output / f"zero_shot_{split}_{prompt['id']}_errors.csv")
        print(json.dumps(result, indent=2))
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate", "test", "smoke"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = read_config()
    prompts = config["prompts"]

    if args.action == "smoke":
        _, frame = provisional_split(args.data_dir)
        run(frame.head(8).reset_index(drop=True), [next(p for p in prompts if p["id"] == "original")], config, output, "smoke_only_not_results")
        return

    if args.action == "validate":
        _, frame = provisional_split(args.data_dir)
        results = run(frame, prompts, config, output, "provisional_validation_seed_42")
        results.sort(key=lambda x: (x["macro_f1"], x["f1_sarcastic"]), reverse=True)
        (output / "zero_shot_validation_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        selected = next(prompt for prompt in prompts if prompt["id"] == results[0]["prompt_id"])
        (output / "selected_prompt.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        print(f"Selected on validation only: {selected['id']}")
        return

    choice_path = output / "selected_prompt.json"
    if not choice_path.exists():
        raise SystemExit("Run zero_shot validate first; prompt selection must precede test evaluation.")
    selected = json.loads(choice_path.read_text(encoding="utf-8"))
    original = next(prompt for prompt in prompts if prompt["id"] == "original")
    frame = load_test(args.data_dir)
    results = run(frame, [original] if selected["id"] == "original" else [original, selected], config, output, "official_test")
    (output / "zero_shot_test_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
