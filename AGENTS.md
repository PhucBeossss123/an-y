# Project rules

- Keep project changes focused on the iSarcasmEval English Subtask A work assigned to Phúc: TF-IDF + SVM and zero-shot classification/prompt comparison.
- Huy owns shared data preparation and BERT-base; Tuấn owns shared metrics and RoBERTa-base; Kiệt owns DistilBERT, demo, and slide integration. Treat their code as awaiting Phúc's review and approval before any push.
- Use the official iSarcasmEval source, preserve its official test split, and keep downloaded data and model checkpoints out of Git. Record source revision and file checksums.
- Only `tweet`/`text` and the binary sarcasm label may feed the classifier. Keep rephrases and annotation columns out of features. Use `0 = sincere/non-sarcastic`, `1 = sarcastic`.
- Choose settings on validation before evaluating test. Report Macro-F1 separately from sarcastic-class F1; do not describe SVM margins or zero-shot scores as calibrated probabilities.
- Do not overwrite user changes, reset or clean data, force-push, merge to the main branch, create releases/tags, or change repository access.
- Phúc's own work may be committed and pushed on a dedicated branch after checks and review. Do not push other members' work until Phúc approves the exact version.
- Document commands, assumptions, and real results in `doc/cach_lam.md`; never invent results or claim a check that did not run.
