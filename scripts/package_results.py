"""Copy small reproducible outputs into the tracked handover directory."""

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main():
    REPORTS.mkdir(exist_ok=True)
    names = [
        "baseline_validation_metrics.json",
        "baseline_validation_search.csv",
        "baseline_validation_predictions.csv",
        "baseline_test_metrics.json",
        "baseline_test_predictions.csv",
        "zero_shot_validation_metrics.json",
        "zero_shot_test_metrics.json",
        "selected_prompt.json",
        *[f"zero_shot_provisional_validation_seed_42_{prompt}_predictions.csv" for prompt in (
            "original", "statement", "explicit_negation", "social_post", "ironic_or_literal"
        )],
        "zero_shot_official_test_original_predictions.csv",
        "zero_shot_official_test_explicit_negation_predictions.csv",
        "data_audit.json",
    ]
    for name in names:
        source = OUTPUT / name
        if not source.exists():
            raise FileNotFoundError(f"Missing actual run output: {source}")
        shutil.copy2(source, REPORTS / name)
    shutil.copy2(ROOT / "data/raw/manifest.json", REPORTS / "data_manifest.json")
    shutil.copy2(ROOT / "data/raw/LICENSE", REPORTS / "source_LICENSE.txt")
    print(f"Packaged {len(names) + 2} small files into {REPORTS}")


if __name__ == "__main__":
    main()
