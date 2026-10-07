# Review log

## Phúc baseline / zero-shot local implementation

- Reviewed version: initial local implementation in this previously empty workspace, before GitHub upload.
- Scope: shared data adapter fallback, TF-IDF + LinearSVC, NLI zero-shot prompts, result export and docs.
- Checks: `scripts/check_environment.py`, `scripts/audit_data.py`, validation and official-test runs, and `scripts/verify_outputs.py`; final results are in `doc/results_phuc.md`.
- Finding fixed: the first smoke run found a boolean-indexing error when exporting error rows (`save_errors`); changed it to a NumPy boolean mask and reran the smoke check successfully.
- Finding fixed: baseline test run now saves the refitted model and records the selected validation configuration in its metrics.
- Data note: one exact text crosses the provisional fit/validation split; the source train/test sets have no exact text overlap. Duplicate source rows are not removed. This remains a limitation until Huy's common split arrives.
- Resource note: the detected RTX 3050 is not exposed to the current PyTorch CPU wheel. The 2.51 GB official CUDA wheel download was stopped after no visible progress for 2.5 minutes; all reported experiments therefore ran on CPU.
- GitHub note: GCM account listing showed `PhucBeossss123`, but authenticated GitHub `/user` returned HTTP 401. No repo was created or remote added; fix authentication before push.
- Team code: none was present to review. Any code received from Huy, Tuấn, or Kiệt stays “chờ Phúc duyệt” until Phúc confirms the exact version.
- Status: local implementation and checks passed subject to the provisional-split limitation; GitHub publication pending correct authenticated access.
