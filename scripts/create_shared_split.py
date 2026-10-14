"""Create and checksum the fixed, stratified development split for the team."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from an_y.data import load_train


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_manifest_path = args.data_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for name in ("train.En.csv", "task_A_En_test.csv"):
        if sha256(args.data_dir / name) != source_manifest["files"][name]["sha256"]:
            raise SystemExit(f"Official file checksum mismatch: {args.data_dir / name}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / f"shared_split_seed{args.seed}.csv"
    manifest_path = args.output_dir / "manifest.json"
    if not args.force and (split_path.exists() or manifest_path.exists()):
        raise SystemExit("Split output already exists; inspect it or pass --force")

    source = load_train(args.data_dir)
    _, validation_ids = train_test_split(
        source["sample_id"],
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=source["label"],
    )
    assignment = pd.Series("train", index=source.index, dtype="object")
    assignment.loc[validation_ids.index] = "validation"
    split = pd.DataFrame({"sample_id": source["sample_id"], "split": assignment})
    split.to_csv(split_path, index=False, lineterminator="\n")
    audited = source.assign(split=assignment.to_numpy())
    cross_split_ids = []
    for _, group in audited.groupby("text", sort=False):
        if group["split"].nunique() > 1:
            cross_split_ids.extend(group["sample_id"].tolist())
    repeated = source[source["text"].duplicated(keep=False)]
    conflicting_groups = int((repeated.groupby("text")["label"].nunique() > 1).sum())

    records = {}
    for name in ("train.En.csv", "task_A_En_test.csv"):
        record = source_manifest["files"][name]
        records[name] = {
            "source_path": record["source_path"],
            "sha256": record["sha256"],
        }
    result = {
        "repository": source_manifest["repository"],
        "source_revision": source_manifest["revision"],
        "source_files": records,
        "split_file": split_path.name,
        "split_sha256": sha256(split_path),
        "split_column_policy": "sample_id and split only; no text or annotation fields",
        "method": "one stratified train_test_split of official train labels",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "rows": {
            name: int(count)
            for name, count in split["split"].value_counts().sort_index().items()
        },
        "labels_by_split": {
            name: {
                str(int(label)): int(count)
                for label, count in source.loc[split["split"].eq(name), "label"]
                .value_counts().sort_index().items()
            }
            for name in ("train", "validation")
        },
        "duplicate_audit": {
            "exact_text_overlap_fit_validation_rows": len(cross_split_ids),
            "exact_text_overlap_fit_validation_sample_ids": cross_split_ids,
            "repeated_official_train_rows": int(len(repeated)),
            "repeated_text_groups_with_conflicting_labels": conflicting_groups,
            "policy": "report duplicates; do not remove or relabel rows automatically",
        },
        "official_test_policy": "kept separate; IDs are not in this split file",
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
