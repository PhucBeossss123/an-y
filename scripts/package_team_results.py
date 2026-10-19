"""Validate, summarize, and package the shared-split transformer comparison."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from an_y.data import load_shared_split, load_test
from an_y.metrics import summarize


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("bert", "roberta", "distilbert")
METRICS = ("macro_f1", "f1_sarcastic", "precision_sarcastic", "recall_sarcastic")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_predictions(path: Path, expected: pd.DataFrame, stored: dict) -> None:
    predictions = pd.read_csv(path)
    assert predictions["sample_id"].tolist() == expected["sample_id"].tolist(), path
    assert predictions["sample_id"].is_unique, path
    assert predictions["y_true"].tolist() == expected["label"].tolist(), path
    assert not ({"text", "tweet", "rephrase", "rationale"} & set(predictions.columns))
    computed = summarize(predictions["y_true"], predictions["y_pred"])
    for metric in METRICS:
        assert abs(computed[metric] - stored[metric]) < 1e-12, (path, metric)


def mean_std(seeds: list[dict], split: str, metric: str) -> tuple[float, float]:
    values = [seed[split][metric] for seed in seeds]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main() -> None:
    outputs = ROOT / "outputs" / "team"
    results = ROOT / "results" / "team"
    results.mkdir(parents=True, exist_ok=True)
    fit, validation = load_shared_split()
    test = load_test(ROOT / "data" / "raw")
    rows = []
    model_records = []

    for name in MODEL_NAMES:
        source = outputs / name
        metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
        seeds = metrics["seeds"]
        assert [seed["seed"] for seed in seeds] == [42, 123, 2026], name
        model_target = results / name
        model_target.mkdir(parents=True, exist_ok=True)
        clean_seeds = []
        for seed in seeds:
            seed_id = seed["seed"]
            run_dir = source / f"seed_{seed_id}"
            validate_predictions(
                run_dir / "validation_predictions.csv", validation, seed["validation"]
            )
            validate_predictions(
                run_dir / "official_test_predictions.csv", test, seed["official_test"]
            )
            target = model_target / f"seed_{seed_id}"
            target.mkdir(parents=True, exist_ok=True)
            for filename in ("validation_predictions.csv", "official_test_predictions.csv"):
                shutil.copy2(run_dir / filename, target / filename)
            clean_seeds.append({
                "seed": seed_id,
                "selected_epoch": seed["selected_epoch"],
                "validation": seed["validation"],
                "official_test": seed["official_test"],
            })

        clean_metrics = {
            key: value for key, value in metrics.items() if key != "seeds"
        }
        clean_metrics["seeds"] = clean_seeds
        (model_target / "metrics.json").write_text(
            json.dumps(clean_metrics, indent=2) + "\n", encoding="utf-8"
        )
        best_seed = max(
            seeds,
            key=lambda seed: (
                seed["validation"]["macro_f1"], seed["validation"]["f1_sarcastic"]
            ),
        )
        selected_checkpoint = str(
            (source / f"seed_{best_seed['seed']}" / "best_checkpoint").resolve()
        )
        (source / "selected_checkpoint.json").write_text(
            json.dumps({"seed": best_seed["seed"], "checkpoint": selected_checkpoint}, indent=2) + "\n",
            encoding="utf-8",
        )
        record = {
            "architecture": name,
            "model_name": metrics["model_name"],
            "revision_requested": metrics["revision_requested"],
            "revision_resolved": metrics["revision_resolved"],
            "validation_macro_f1_mean": mean_std(seeds, "validation", "macro_f1")[0],
            "validation_macro_f1_std": mean_std(seeds, "validation", "macro_f1")[1],
            "validation_f1_sarcastic_mean": mean_std(seeds, "validation", "f1_sarcastic")[0],
            "validation_f1_sarcastic_std": mean_std(seeds, "validation", "f1_sarcastic")[1],
            "test_macro_f1_mean": mean_std(seeds, "official_test", "macro_f1")[0],
            "test_macro_f1_std": mean_std(seeds, "official_test", "macro_f1")[1],
            "test_f1_sarcastic_mean": mean_std(seeds, "official_test", "f1_sarcastic")[0],
            "test_f1_sarcastic_std": mean_std(seeds, "official_test", "f1_sarcastic")[1],
            "best_validation_seed": best_seed["seed"],
            "best_checkpoint": selected_checkpoint,
            "seeds": clean_seeds,
        }
        model_records.append(record)
        rows.append({key: value for key, value in record.items() if key not in ("seeds", "best_checkpoint")})

    selected = max(
        model_records,
        key=lambda row: (row["validation_macro_f1_mean"], row["validation_f1_sarcastic_mean"]),
    )
    comparison = {
        "selection": "highest mean validation Macro-F1 across seeds; sarcastic F1 breaks ties",
        "selected_architecture": selected["architecture"],
        "selected_seed": selected["best_validation_seed"],
        "selected_checkpoint": selected["best_checkpoint"],
        "models": model_records,
    }
    (results / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(results / "model_comparison.csv", rows, list(rows[0]))
    (outputs / "selected_model.json").write_text(
        json.dumps({
            "architecture": comparison["selected_architecture"],
            "seed": comparison["selected_seed"],
            "checkpoint": comparison["selected_checkpoint"],
            "selected_on": "mean validation Macro-F1 across seeds",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Team transformer comparison", "",
        "All three encoders use the official shared train/validation IDs and official test, with seeds 42, 123, and 2026. Epochs are selected per seed on validation Macro-F1, with sarcastic-class F1 as tie-break. Test scores were not used to select the demo model.",
        "", "| Model | Validation Macro-F1 (mean ± sample SD) | Validation sarcastic F1 | Test Macro-F1 | Test sarcastic F1 |", "|---|---:|---:|---:|---:|",
    ]
    for model in model_records:
        lines.append(
            f"| {model['model_name']} | {model['validation_macro_f1_mean']:.4f} ± {model['validation_macro_f1_std']:.4f} "
            f"| {model['validation_f1_sarcastic_mean']:.4f} ± {model['validation_f1_sarcastic_std']:.4f} "
            f"| {model['test_macro_f1_mean']:.4f} ± {model['test_macro_f1_std']:.4f} "
            f"| {model['test_f1_sarcastic_mean']:.4f} ± {model['test_f1_sarcastic_std']:.4f} |"
        )
    lines.extend([
        "", f"Demo model selected on validation only: **{selected['model_name']}**, seed `{selected['best_validation_seed']}`.",
        "Per-seed ID-only predictions and metrics are stored in each model folder. Checkpoints remain under the ignored `outputs/team/` directory.",
        "", "Run `python scripts/demo.py` after training to launch the local demo.",
    ])
    (results / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified and packaged three architectures; validation selected {selected['architecture']} seed {selected['best_validation_seed']}.")


if __name__ == "__main__":
    main()
