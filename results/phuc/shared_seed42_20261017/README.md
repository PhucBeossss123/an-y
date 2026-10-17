# Shared split run — Phúc

Run date: 2026-10-17. Data: official iSarcasmEval English Subtask A; validation IDs from `data/splits/shared_split_seed42.csv`; official test kept unchanged. Validation and prompt selection ran first, then official test was evaluated once for the selected configuration. No test result was used to change the selection.

## Results

| Method/prompt | Split | n | Macro-F1 | F1 sarcastic | Precision / recall sarcastic |
|---|---|---:|---:|---:|---:|
| TF-IDF + SVM | shared validation seed 42 | 694 | 0.5525 | 0.2772 | 0.3936 / 0.2139 |
| TF-IDF + SVM | official test | 1,400 | 0.5708 | 0.2826 | 0.2500 / 0.3250 |
| Zero-shot, selected `explicit_negation` | shared validation seed 42 | 694 | 0.5488 | 0.2764 | 0.3725 / 0.2197 |
| Zero-shot, `original` | official test | 1,400 | 0.4103 | 0.1083 | 0.0766 / 0.1850 |
| Zero-shot, selected `explicit_negation` | official test | 1,400 | 0.4476 | 0.0428 | 0.0460 / 0.0400 |

The prompt was selected by validation Macro-F1. The selected prompt has higher Macro-F1 but lower sarcastic-class F1 than the original prompt on test; this trade-off is reported as measured.

## Files

- `baseline_*_metrics.json`, `zero_shot_*_metrics.json`: run metrics and split labels.
- `prompt_comparison.csv`: validation comparison of all five prompts.
- `summary.csv`: principal validation/test results.
- Prediction CSVs contain IDs, gold/predicted labels and, for zero-shot, candidate score fields. They contain no tweet text, rephrases, or annotation columns.
- Historical provisional runs remain in `../runs/` and are not overwritten.

## Reproduction commands

```powershell
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs\shared_review_20261017 --split validation --validation-split shared
.\.venv\Scripts\python.exe -m an_y.zero_shot validate --data-dir data\raw --output-dir outputs\shared_review_20261017 --validation-split shared
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs\shared_review_20261017 --split test --validation-split shared
.\.venv\Scripts\python.exe -m an_y.zero_shot test --data-dir data\raw --output-dir outputs\shared_review_20261017
```
