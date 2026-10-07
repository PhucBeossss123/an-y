"""Train and evaluate the TF-IDF + Linear SVM baseline."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import load_test, load_train, provisional_split, save_errors
from .metrics import summarize


def make_model(c=1.0, class_weight=None):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("svm", LinearSVC(C=c, class_weight=class_weight, random_state=42)),
    ])


def predictions(frame, pred):
    return pd.DataFrame({
        "sample_id": frame["sample_id"],
        "y_true": frame["label"],
        "y_pred": pred,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.split == "validation":
        train, evaluate = provisional_split(args.data_dir)
        candidates = [(c, weight) for c in (0.5, 1.0, 2.0) for weight in (None, "balanced")]
        rows = []
        best = None
        for c, weight in candidates:
            model = make_model(c, weight)
            model.fit(train["text"], train["label"])
            pred = model.predict(evaluate["text"])
            score = summarize(evaluate["label"], pred)
            row = {"C": c, "class_weight": weight or "none", **score}
            rows.append(row)
            key = (score["macro_f1"], score["f1_sarcastic"])
            if best is None or key > best[0]:
                best = (key, model, row, pred)
        pd.DataFrame(rows).to_csv(output / "baseline_validation_search.csv", index=False)
        joblib.dump(best[1], output / "svm.joblib")
        predictions(evaluate, best[3]).to_csv(output / "baseline_validation_predictions.csv", index=False)
        save_errors(evaluate, best[3], output / "baseline_validation_errors.csv")
        result = {"method": "tfidf_linear_svm", "split": "provisional_validation_seed_42", **best[2]}
        (output / "baseline_validation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    model_path = output / "svm.joblib"
    if not model_path.exists():
        raise SystemExit("Run validation first to select and save the baseline configuration.")
    train = load_train(args.data_dir)
    evaluate = load_test(args.data_dir)
    model = joblib.load(model_path)
    # Refit the selected pipeline on all official train rows only after validation selection.
    model.fit(train["text"], train["label"])
    joblib.dump(model, model_path)
    pred = model.predict(evaluate["text"])
    selected = json.loads((output / "baseline_validation_metrics.json").read_text(encoding="utf-8"))
    result = {
        "method": "tfidf_linear_svm",
        "split": "official_test",
        "selected_on": "provisional_validation_seed_42",
        "C": selected["C"],
        "class_weight": selected["class_weight"],
        **summarize(evaluate["label"], pred),
    }
    predictions(evaluate, pred).to_csv(output / "baseline_test_predictions.csv", index=False)
    save_errors(evaluate, pred, output / "baseline_test_errors.csv")
    (output / "baseline_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
