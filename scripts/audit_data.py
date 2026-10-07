"""Write a read-only audit of split sizes, labels, IDs, and exact-text overlap."""

import json
from pathlib import Path

from an_y.data import load_test, load_train, provisional_split


def summary(frame):
    return {
        "n": int(len(frame)),
        "labels": {str(int(k)): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "unique_ids": int(frame["sample_id"].nunique()),
        "unique_texts": int(frame["text"].nunique()),
    }


def main():
    train = load_train("data/raw")
    fit, validation = provisional_split("data/raw")
    test = load_test("data/raw")
    repeated_train = train[train["text"].duplicated(keep=False)]
    overlap_texts = set(fit["text"]) & set(validation["text"])
    overlap_rows = train[train["text"].isin(overlap_texts)]
    conflicting_duplicate_texts = int(
        (repeated_train.groupby("text")["label"].nunique() > 1).sum()
    )
    duplicate_groups = []
    for _, group in repeated_train.groupby("text", sort=False):
        duplicate_groups.append({
            "sample_ids": group["sample_id"].tolist(),
            "labels": [int(value) for value in group["label"].tolist()],
            "conflicting_labels": group["label"].nunique() > 1,
        })
    result = {
        "train_source": summary(train),
        "provisional_fit": summary(fit),
        "provisional_validation": summary(validation),
        "official_test": summary(test),
        "exact_text_overlap_fit_validation": len(set(fit["text"]) & set(validation["text"])),
        "exact_text_overlap_train_test": len(set(train["text"]) & set(test["text"])),
        "fit_validation_overlap_rows": [
            {"sample_id": row.sample_id, "label": int(row.label)}
            for row in overlap_rows.itertuples(index=False)
        ],
        "repeated_train_text_rows": int(len(repeated_train)),
        "repeated_train_text_groups_with_conflicting_labels": conflicting_duplicate_texts,
        "repeated_train_groups": duplicate_groups,
        "repeated_train_sample_ids": repeated_train["sample_id"].tolist(),
    }
    output = Path("outputs/data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
