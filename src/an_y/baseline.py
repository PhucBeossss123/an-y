"""Train and evaluate the TF-IDF + Linear SVM baseline."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import load_shared_split, load_test, load_train, provisional_split, save_errors
from .metrics import summarize


C_VALUES = (0.5, 1.0, 2.0)
CLASS_WEIGHTS = (None, "balanced")


def make_model(c: float = 1.0, class_weight: str | None = None) -> Pipeline:
    """Build one complete vectorizer-and-classifier pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "svm",
                LinearSVC(C=c, class_weight=class_weight, random_state=42),
            ),
        ]
    )


def prediction_frame(frame: pd.DataFrame, predicted_labels) -> pd.DataFrame:
    """Keep predictions linked to their source row and true label."""
    return pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "y_true": frame["label"],
            "y_pred": predicted_labels,
        }
    )


def write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


def fit_and_score(model: Pipeline, train: pd.DataFrame, evaluate: pd.DataFrame):
    model.fit(train["text"], train["label"])
    predicted_labels = model.predict(evaluate["text"])
    scores = summarize(evaluate["label"], predicted_labels)
    return predicted_labels, scores


def run_validation(
    data_dir: str, output_dir: Path, validation_split: str = "shared"
) -> None:
    """Select C and class weighting using validation, shared by default."""
    if validation_split == "shared":
        train, validation = load_shared_split(data_dir)
        split_name = "shared_validation_seed_42"
    else:
        train, validation = provisional_split(data_dir)
        split_name = "provisional_validation_seed_42"
    search_rows = []
    best_candidate = None

    for c in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            model = make_model(c=c, class_weight=class_weight)
            predicted_labels, scores = fit_and_score(model, train, validation)
            search_row = {
                "C": c,
                "class_weight": class_weight or "none",
                **scores,
            }
            search_rows.append(search_row)

            # Macro-F1 is the primary selection metric; sarcastic F1 breaks ties.
            selection_key = (scores["macro_f1"], scores["f1_sarcastic"])
            if best_candidate is None or selection_key > best_candidate["key"]:
                best_candidate = {
                    "key": selection_key,
                    "model": model,
                    "row": search_row,
                    "predictions": predicted_labels,
                }

    assert best_candidate is not None  # The fixed candidate lists are non-empty.
    pd.DataFrame(search_rows).to_csv(
        output_dir / "baseline_validation_search.csv", index=False
    )
    joblib.dump(best_candidate["model"], output_dir / "svm.joblib")
    prediction_frame(validation, best_candidate["predictions"]).to_csv(
        output_dir / "baseline_validation_predictions.csv", index=False
    )
    save_errors(
        validation,
        best_candidate["predictions"],
        output_dir / "baseline_validation_errors.csv",
    )

    result = {
        "method": "tfidf_linear_svm",
        "split": split_name,
        **best_candidate["row"],
    }
    write_json(output_dir / "baseline_validation_metrics.json", result)
    print(json.dumps(result, indent=2))


def run_test(data_dir: str, output_dir: Path) -> None:
    """Refit the validation-selected pipeline on official train, then test once."""
    model_path = output_dir / "svm.joblib"
    metrics_path = output_dir / "baseline_validation_metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        raise SystemExit("Run validation first to select and save the baseline model.")

    train = load_train(data_dir)
    test = load_test(data_dir)
    model = joblib.load(model_path)

    # The official test split is used only after validation has selected settings.
    predicted_labels, scores = fit_and_score(model, train, test)
    joblib.dump(model, model_path)

    selected = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = {
        "method": "tfidf_linear_svm",
        "split": "official_test",
        "selected_on": selected["split"],
        "C": selected["C"],
        "class_weight": selected["class_weight"],
        **scores,
    }
    prediction_frame(test, predicted_labels).to_csv(
        output_dir / "baseline_test_predictions.csv", index=False
    )
    save_errors(test, predicted_labels, output_dir / "baseline_test_errors.csv")
    write_json(output_dir / "baseline_test_metrics.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Run validation selection first, then evaluate the official test split.",
    )
    parser.add_argument(
        "--validation-split",
        choices=("shared", "provisional"),
        default="shared",
        help="Use Huy's shared split, or the historical provisional seed-42 split.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "validation":
        run_validation(args.data_dir, output_dir, args.validation_split)
    else:
        run_test(args.data_dir, output_dir)


if __name__ == "__main__":
    main()
