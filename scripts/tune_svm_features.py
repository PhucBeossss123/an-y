"""Compare richer TF-IDF feature sets on the committed shared validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from an_y.baseline import make_context_model, make_model
from an_y.data import load_shared_split, load_test, load_train
from an_y.metrics import summarize


def make_richer_model(c: float, class_weight: str | None, char_weight: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=160_000,
                ),
            ),
        ],
        transformer_weights={"word": 1.0, "char": char_weight},
    )
    return Pipeline(
        [
            ("tfidf", features),
            (
                "svm",
                LinearSVC(C=c, class_weight=class_weight, random_state=42),
            ),
        ]
    )


def select_threshold(labels: np.ndarray, margins: np.ndarray) -> tuple[float, dict]:
    candidates = np.quantile(margins, np.linspace(0.03, 0.97, 95))
    candidates = np.unique(np.r_[candidates, 0.0])
    scored = [
        (float(threshold), summarize(labels, (margins >= threshold).astype(int)))
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[1]["macro_f1"], item[1]["f1_sarcastic"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/experiments/svm_feature_search")
    )
    parser.add_argument(
        "--finalize-test", action="store_true",
        help="After validation selection, refit on official train and score official test once.",
    )
    parser.add_argument(
        "--phuc-output-dir", type=Path, default=Path("outputs/shared_review_20261017")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit, validation = load_shared_split(args.data_dir)
    y_true = validation["label"].to_numpy()

    current = make_model(c=0.5, class_weight="balanced")
    current.fit(fit["text"], fit["label"])
    current_margins = current.decision_function(validation["text"])
    current_threshold, current_metrics = select_threshold(y_true, current_margins)
    results = [{
        "features": "word unigram/bigram (existing)",
        "C": 0.5,
        "class_weight": "balanced",
        "char_weight": 0.0,
        "threshold": current_threshold,
        **current_metrics,
    }]
    best_model = current
    best_result = results[0]

    for c in (0.25, 0.5, 1.0, 2.0):
        for class_weight in (None, "balanced"):
            for char_weight in (0.25, 0.5, 1.0):
                model = make_richer_model(c, class_weight, char_weight)
                model.fit(fit["text"], fit["label"])
                margins = model.decision_function(validation["text"])
                threshold, metrics = select_threshold(y_true, margins)
                result = {
                    "features": "word unigram/trigram + char_wb 3-5gram",
                    "C": c,
                    "class_weight": class_weight or "none",
                    "char_weight": char_weight,
                    "threshold": threshold,
                    **metrics,
                }
                results.append(result)
                key = (metrics["macro_f1"], metrics["f1_sarcastic"])
                best_key = (best_result["macro_f1"], best_result["f1_sarcastic"])
                if key > best_key:
                    best_model, best_result = model, result
                print(json.dumps(result), flush=True)

    for c in (0.25, 0.5, 1.0):
        for class_weight in (None, "balanced"):
            for char_weight in (0.5, 1.0):
                model = make_context_model(c, class_weight, char_weight)
                model.fit(fit["text"], fit["label"])
                margins = model.decision_function(validation["text"])
                threshold, metrics = select_threshold(y_true, margins)
                result = {
                    "features": "word/character TF-IDF + rhetorical/polarity markers",
                    "C": c,
                    "class_weight": class_weight or "none",
                    "char_weight": char_weight,
                    "threshold": threshold,
                    **metrics,
                }
                results.append(result)
                key = (metrics["macro_f1"], metrics["f1_sarcastic"])
                best_key = (best_result["macro_f1"], best_result["f1_sarcastic"])
                if key > best_key:
                    best_model, best_result = model, result
                print(json.dumps(result), flush=True)

    results.sort(key=lambda row: (row["macro_f1"], row["f1_sarcastic"]), reverse=True)
    (args.output_dir / "validation_search.json").write_text(
        json.dumps(
            {
                "selection_split": "shared_validation_seed_42",
                "selected_on": "Macro-F1, then sarcastic-class F1",
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    joblib.dump(best_model, args.output_dir / "svm.joblib")
    pd.DataFrame(
        {
            "sample_id": validation["sample_id"],
            "y_true": y_true,
            "y_pred": (best_model.decision_function(validation["text"]) >= best_result["threshold"]).astype(int),
        }
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    (args.output_dir / "selected_model.json").write_text(
        json.dumps(best_result, indent=2) + "\n", encoding="utf-8"
    )
    print("SELECTED " + json.dumps(best_result), flush=True)

    if args.finalize_test:
        if "rhetorical/polarity markers" in best_result["features"]:
            final_model = make_context_model(
                float(best_result["C"]),
                None if best_result["class_weight"] == "none" else best_result["class_weight"],
                float(best_result["char_weight"]),
            )
        elif "char_wb" in best_result["features"]:
            final_model = make_richer_model(
                float(best_result["C"]),
                None if best_result["class_weight"] == "none" else best_result["class_weight"],
                float(best_result["char_weight"]),
            )
        else:
            final_model = make_model(
                float(best_result["C"]),
                None if best_result["class_weight"] == "none" else best_result["class_weight"],
            )
        official_train = load_train(args.data_dir)
        official_test = load_test(args.data_dir)
        final_model.fit(official_train["text"], official_train["label"])
        test_predictions = final_model.predict(official_test["text"])
        test_metrics = summarize(official_test["label"], test_predictions)
        pd.DataFrame(
            {
                "sample_id": official_test["sample_id"],
                "y_true": official_test["label"],
                "y_pred": test_predictions,
            }
        ).to_csv(args.output_dir / "official_test_predictions.csv", index=False)
        (args.output_dir / "official_test_metrics.json").write_text(
            json.dumps({"method": "tfidf_svm_context", "split": "official_test", **test_metrics}, indent=2) + "\n",
            encoding="utf-8",
        )
        args.phuc_output_dir.mkdir(parents=True, exist_ok=True)
        model_name = "svm_context_20261021.joblib"
        joblib.dump(final_model, args.phuc_output_dir / model_name)
        settings = {
            "model_file": model_name,
            "decision_threshold": float(best_result["threshold"]),
            "selected_on": "shared_validation_seed_42",
            "validation_macro_f1": float(best_result["macro_f1"]),
            "validation_f1_sarcastic": float(best_result["f1_sarcastic"]),
            "features": best_result["features"],
            "C": best_result["C"],
            "class_weight": best_result["class_weight"],
            "char_weight": best_result["char_weight"],
            "official_test": test_metrics,
        }
        (args.phuc_output_dir / "svm_context.json").write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8"
        )
        print("OFFICIAL TEST " + json.dumps(test_metrics), flush=True)


if __name__ == "__main__":
    main()
