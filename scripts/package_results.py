"""Package verified experiment outputs as reports and append-only run records."""

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results" / "phuc"
ERROR_NOTES = {
    ("baseline", "test:2"): ("contextual_contradiction", "The broken website is framed as a consequence of people wanting the service; the surface reads as a complaint."),
    ("baseline", "test:4"): ("emoji_or_short_text", "A short sincere positive statement with an emoji is predicted as sarcastic."),
    ("baseline", "test:20"): ("emoji_or_short_text", "A sincere sign-off with emojis is predicted as sarcastic; emoji alone is not a sarcasm cue."),
    ("baseline", "test:21"): ("surface_positive_sarcasm", "Sarcastic praise about never making dirty moves conflicts with the intended criticism."),
    ("baseline", "test:47"): ("surface_positive_sarcasm", "Ironic praise of a politician follows positive lexical cues."),
    ("zero_shot", "test:1"): ("insufficient_context", "The one-word item lacks visible context; the model predicts sarcasm for a sincere example."),
    ("zero_shot", "test:2"): ("contextual_contradiction", "The apparent statement about intent and a broken website is read literally."),
    ("zero_shot", "test:20"): ("emoji_or_short_text", "A friendly night-time sign-off with emojis is predicted as sarcastic."),
    ("zero_shot", "test:21"): ("surface_positive_sarcasm", "Hyperbolic sports praise is read as sincere without external context."),
    ("zero_shot", "test:84"): ("contextual_contradiction", "Giving children sweets before bed is praised despite the contradictory expected consequence."),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def confusion_csv(metrics, path):
    matrix = metrics["confusion_matrix_labels_0_1"]
    write_csv(path, [
        {"true_label": 0, "pred_sincere_0": matrix[0][0], "pred_sarcastic_1": matrix[0][1]},
        {"true_label": 1, "pred_sincere_0": matrix[1][0], "pred_sarcastic_1": matrix[1][1]},
    ], ["true_label", "pred_sincere_0", "pred_sarcastic_1"])


def main():
    REPORTS.mkdir(exist_ok=True)
    validation_metrics_source = read_json(OUTPUT / "zero_shot_validation_metrics.json")
    names = [
        "baseline_validation_metrics.json", "baseline_validation_search.csv",
        "baseline_validation_predictions.csv", "baseline_test_metrics.json",
        "baseline_test_predictions.csv", "zero_shot_validation_metrics.json",
        "zero_shot_test_metrics.json", "selected_prompt.json",
        *[
            f"zero_shot_{item['split']}_{item['prompt_id']}_predictions.csv"
            for item in validation_metrics_source
        ],
        "zero_shot_official_test_original_predictions.csv",
        "zero_shot_official_test_explicit_negation_predictions.csv", "data_audit.json",
    ]
    for name in names:
        source = OUTPUT / name
        if not source.exists():
            raise FileNotFoundError(f"Missing actual run output: {source}")
        shutil.copy2(source, REPORTS / name)
    shutil.copy2(ROOT / "data/raw/manifest.json", REPORTS / "data_manifest.json")
    shutil.copy2(ROOT / "data/raw/LICENSE", REPORTS / "source_LICENSE.txt")

    baseline_validation = read_json(REPORTS / "baseline_validation_metrics.json")
    baseline_test = read_json(REPORTS / "baseline_test_metrics.json")
    validation_metrics = read_json(REPORTS / "zero_shot_validation_metrics.json")
    test_metrics = read_json(REPORTS / "zero_shot_test_metrics.json")
    prompt_config = read_json(ROOT / "config/prompts.json")
    data_manifest = read_json(REPORTS / "data_manifest.json")
    prompt_by_id = {prompt["id"]: prompt for prompt in prompt_config["prompts"]}
    selected_prompt = read_json(REPORTS / "selected_prompt.json")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_sources = [
        ("baseline", "validation", None, baseline_validation, "baseline_validation_predictions.csv"),
        ("baseline", "test", None, baseline_test, "baseline_test_predictions.csv"),
    ]
    run_sources.extend(
        ("zero_shot", "validation", item["prompt_id"], item,
         f"zero_shot_{item['split']}_{item['prompt_id']}_predictions.csv")
        for item in validation_metrics
    )
    run_sources.extend(
        ("zero_shot", "test", item["prompt_id"], item,
         f"zero_shot_official_test_{item['prompt_id']}_predictions.csv")
        for item in test_metrics
    )

    runs = []
    for method, split, prompt_id, metrics, prediction_name in run_sources:
        prompt_suffix = f"-{prompt_id}" if prompt_id else ""
        run_id = f"{method}-{split}{prompt_suffix}-{stamp}"
        run_dir = RESULTS / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        prediction_source = REPORTS / prediction_name
        prediction_target = run_dir / "predictions.csv"
        shutil.copy2(prediction_source, prediction_target)
        metric_target = run_dir / "metrics.json"
        metric_target.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        confusion_csv(metrics, run_dir / "confusion_matrix.csv")

        if method == "baseline":
            config = {
                "method": "TF-IDF word (1,2)-gram + LinearSVC",
                "C": metrics["C"], "class_weight": metrics["class_weight"],
                "random_state": 42, "split_seed": 42,
                "vectorizer_fit": "training rows only",
                "selected_on": metrics.get("selected_on"),
            }
        else:
            prompt = prompt_by_id[prompt_id]
            config = {
                "method": "Transformers zero-shot-classification",
                "checkpoint": prompt_config["model"],
                "model_revision": prompt_config["model_revision"],
                "prompt_id": prompt_id,
                "candidate_labels": prompt["candidate_labels"],
                "label_to_id": prompt["label_to_id"],
                "hypothesis_template": prompt["hypothesis_template"],
                "multi_label": False,
                "batch_size": prompt_config["batch_size"],
                "hypothesis_max_length": prompt_config["hypothesis_max_length"],
                "selected_by_validation": prompt_id == selected_prompt["id"],
                "selected_on": metrics.get("selected_on"),
            }
        config.update({
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": data_manifest["task"],
            "source_repository": data_manifest["repository"],
            "source_revision": data_manifest["revision"],
            "source_file_sha256": {name: record["sha256"] for name, record in data_manifest["files"].items()},
            "split": metrics["split"],
            "split_seed": 42 if "seed_42" in metrics["split"] else None,
            "split_status": (
                "provisional; not the shared validation split"
                if "provisional" in metrics["split"]
                else "shared validation" if "validation" in metrics["split"]
                else "official test"
            ),
            "n": metrics["n"],
            "prediction_sha256": sha256(prediction_target),
        })
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (run_dir / "README.md").write_text(
            f"# Run `{run_id}`\n\n"
            f"- Method: {config['method']}\n"
            f"- Split: `{metrics['split']}` ({config['split_status']}); n={metrics['n']}\n"
            f"- Macro-F1: {metrics['macro_f1']:.6f}; F1 sarcastic: {metrics['f1_sarcastic']:.6f}\n"
            f"- Source revision: `{config['source_revision']}`\n\n"
            "Files: [configuration](config.json), [metrics](metrics.json), "
            "[predictions](predictions.csv), [confusion matrix](confusion_matrix.csv).\n",
            encoding="utf-8",
        )
        runs.append({"run_id": run_id, "method": method, "split": split, "prompt_id": prompt_id,
                     "metrics": metrics, "config": config, "prediction_file": prediction_target})

    # Keep one row for each method/prompt and each evaluated split.
    summary_rows = []
    for run in runs:
        metrics = run["metrics"]
        summary_rows.append({
            "run_id": run["run_id"], "method": run["method"], "prompt_id": run["prompt_id"] or "",
            "split": metrics["split"], "split_status": run["config"]["split_status"],
            "n": metrics["n"], "macro_f1": metrics["macro_f1"], "f1_sarcastic": metrics["f1_sarcastic"],
            "precision_sarcastic": metrics["precision_sarcastic"], "recall_sarcastic": metrics["recall_sarcastic"],
            "precision_sincere": metrics["precision_sincere"], "recall_sincere": metrics["recall_sincere"],
            "selected_by_validation": run["config"].get("selected_by_validation", ""),
            "checkpoint_or_model": run["config"].get("checkpoint", run["config"].get("method")),
        })
    write_csv(RESULTS / "summary.csv", summary_rows, list(summary_rows[0]))

    shared_validation = (
        baseline_validation["split"].startswith("shared")
        and all("shared" in item["split"] for item in validation_metrics)
    )
    status = (
        "Baseline và zero-shot dùng split validation chung; prompt được chọn trên validation; official test đã đánh giá."
        if shared_validation else
        "Các artifact hiện có vẫn dùng split validation tạm seed 42. Loader split chung đã có; cần chạy lại để cập nhật kết quả Phúc."
    )
    lines = [
        "# Kết quả Phúc — iSarcasmEval English Subtask A", "",
        f"**Trạng thái:** {status}", "",
        "- [Lượt chạy mới nhất trên shared split seed 42](shared_seed42_20261017/README.md)",
        "- [Báo cáo phương pháp và phân tích lỗi](../../doc/bao_cao_phuc.md)",
        "- [Hướng dẫn nhóm và chạy lại](../../doc/cach_lam.md)",
        "- [Tổng hợp CSV](summary.csv) · [So sánh prompt validation](prompt_comparison.csv) · [Lỗi ví dụ](errors.csv)",
        "- [Manifest/checksum dữ liệu](../../reports/data_manifest.json) · [Audit dữ liệu](../../reports/data_audit.json)", "",
        "## Run được đóng gói trong lần chạy này", "",
        "| Method/prompt | Split | n | Macro-F1 | F1 sarcastic | Precision / recall sarcastic |",
        "|---|---|---:|---:|---:|---:|",
    ]
    selected_id = selected_prompt["id"]
    featured = [
        row for row in summary_rows
        if row["method"] == "baseline" or row["prompt_id"] in ("original", selected_id)
    ]
    for row in featured:
        label = "TF-IDF + SVM" if row["method"] == "baseline" else f"zero-shot `{row['prompt_id']}`"
        lines.append(
            f"| {label} | {row['split']} | {row['n']} | {row['macro_f1']:.4f} | {row['f1_sarcastic']:.4f} "
            f"| {row['precision_sarcastic']:.4f} / {row['recall_sarcastic']:.4f} |"
        )
    lines.extend([
        "", "Zero-shot prompt selection uses **validation Macro-F1 only** (F1 sarcastic is the tie-break). "
        f"`{selected_prompt['id']}` was selected. Test results are reported after the choice and were not used to change it.",
        "", "## Trạng thái theo phương pháp", "",
        f"- **TF-IDF + SVM:** validation split `{baseline_validation['split']}`; official test split `{baseline_test['split']}`.",
        f"- **Zero-shot:** validation split `{validation_metrics[0]['split']}`; prompt được chọn trên validation. Test kết quả nằm trong các artifact trên.",
        "- **BERT-base:** xem kết quả và ghi chú tại [cách làm nhóm](../../doc/cach_lam.md).",
        "- **Nhánh cộng tác viên và trạng thái review:** xem [review](../../doc/review.md).",
        "", "## Các lượt chạy và artifact", "",
        "Mỗi thư mục run lưu cấu hình/nguồn/split/seed, metric, prediction và confusion matrix. ID run có timestamp; "
        "khi đóng gói lần khác, thư mục cũ không bị ghi đè.", "",
        "| Run | Method | Split | n | Macro-F1 | F1 sarcastic |", "|---|---|---|---:|---:|---:|",
    ])
    for row in summary_rows:
        label = row["method"] + (f" / {row['prompt_id']}" if row["prompt_id"] else "")
        lines.append(
            f"| [`{row['run_id']}`](runs/{row['run_id']}/README.md) | {label} | {row['split']} | {row['n']} "
            f"| {row['macro_f1']:.4f} | {row['f1_sarcastic']:.4f} |"
        )
    lines.extend([
        "", "Macro-F1 khác với F1 lớp sarcastic. Zero-shot score không phải xác suất hiệu chỉnh; SVM decision margin cũng không phải xác suất.",
        "",
    ])
    (RESULTS / "README.md").write_text("\n".join(lines), encoding="utf-8")

    prompt_rows = []
    for metrics in validation_metrics:
        prompt = prompt_by_id[metrics["prompt_id"]]
        prompt_rows.append({
            "prompt_id": metrics["prompt_id"], "candidate_labels": " | ".join(prompt["candidate_labels"]),
            "label_to_id": json.dumps(prompt["label_to_id"], ensure_ascii=False, sort_keys=True),
            "hypothesis_template": prompt["hypothesis_template"], "split": metrics["split"],
            "n": metrics["n"], "macro_f1": metrics["macro_f1"], "f1_sarcastic": metrics["f1_sarcastic"],
            "selected_by_validation": metrics["prompt_id"] == selected_prompt["id"],
        })
    write_csv(RESULTS / "prompt_comparison.csv", prompt_rows, list(prompt_rows[0]))

    curated_errors = []
    sources = {
        ("baseline", "baseline_test_predictions.csv"): None,
        ("zero_shot", f"zero_shot_official_test_{selected_prompt['id']}_predictions.csv"): selected_prompt["id"],
    }
    for (method, filename), prompt_id in sources.items():
        with (REPORTS / filename).open(encoding="utf-8-sig", newline="") as stream:
            predictions = {row["sample_id"]: row for row in csv.DictReader(stream)}
        for (note_method, sample_id), (group, note) in ERROR_NOTES.items():
            if note_method != method:
                continue
            row = predictions[sample_id]
            if int(row["y_true"]) == int(row["y_pred"]):
                raise ValueError(f"Curated error is not an error in {filename}: {sample_id}")
            curated_errors.append({
                "method": method, "prompt_id": prompt_id or "", "split": "official_test",
                "sample_id": sample_id, "y_true": row["y_true"], "y_pred": row["y_pred"],
                "error_group": group, "manual_note": note,
            })
    write_csv(RESULTS / "errors.csv", curated_errors, list(curated_errors[0]))
    print(f"Packaged {len(names) + 2} report files and {len(runs)} append-only runs into {RESULTS}")


if __name__ == "__main__":
    main()
