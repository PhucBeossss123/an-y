"""Combine the available Huy BERT seeds into the fixed team seed set."""

import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "outputs" / "bert" / "full_three_seed_20261014"
TARGET = ROOT / "outputs" / "team" / "bert"
SEEDS = (42, 123, 2026)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    legacy = load_metrics(LEGACY / "metrics.json")
    current = load_metrics(TARGET / "metrics.json")
    for key in (
        "model_name",
        "revision_requested",
        "source_revision",
        "source_file_checksums",
        "split_sha256",
        "train_rows",
        "validation_rows",
        "official_test_rows",
    ):
        assert legacy[key] == current[key], (key, legacy[key], current[key])

    old_seeds = {item["seed"]: item for item in legacy["seeds"]}
    new_seeds = {item["seed"]: item for item in current["seeds"]}
    assert 123 in new_seeds and 42 in old_seeds and 2026 in old_seeds
    merged = {}
    for seed in SEEDS:
        source_root = TARGET if seed == 123 else LEGACY
        source = source_root / f"seed_{seed}"
        assert (source / "validation_predictions.csv").exists(), source
        assert (source / "official_test_predictions.csv").exists(), source
        if seed != 123:
            shutil.copytree(source, TARGET / f"seed_{seed}", dirs_exist_ok=True)
        merged[seed] = (new_seeds if seed == 123 else old_seeds)[seed]

    seeds = [merged[seed] for seed in SEEDS]
    result = dict(current)
    result["config"] = dict(current["config"])
    result["config"]["seeds"] = list(SEEDS)
    result["seeds"] = seeds
    result["revision_resolved"] = (
        current["revision_resolved"]
        or legacy["revision_resolved"]
        or current["revision_requested"]
    )
    for split_key, summary_key in (
        ("validation", "validation_summary"),
        ("official_test", "official_test_summary"),
    ):
        result[summary_key] = {
            metric: {
                "mean": float(np.mean([run[split_key][metric] for run in seeds])),
                "std_sample": float(np.std(
                    [run[split_key][metric] for run in seeds], ddof=1
                )),
            }
            for metric in ("macro_f1", "f1_sarcastic")
        }
    result["seed_sources"] = {
        "42": "outputs/bert/full_three_seed_20261014",
        "123": "retrained with scripts/train_transformer.py on the pinned revision",
        "2026": "outputs/bert/full_three_seed_20261014",
    }
    (TARGET / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("Merged BERT seeds 42, 123, 2026; run package_team_results.py to validate predictions.")


if __name__ == "__main__":
    main()
