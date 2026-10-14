# Review log

## Phúc baseline / zero-shot

- Reviewed branch: `phuc/baseline-zero-shot`.
- Scope: shared fallback data adapter, TF-IDF + LinearSVC, NLI zero-shot prompts, saved predictions/metrics and handover docs.
- Verified commit at last handover: `7c9b9b5827f79093a61bb32d50e9044eb0ca435b` (`Add baseline and zero-shot experiments`).
- Checks previously run: environment check, `compileall`, data audit, baseline validation and official-test evaluation, five zero-shot validation prompts, original/selected official-test evaluation, `scripts/verify_outputs.py`, result packaging, and `git show --check`.
- Data limitation: the branch now contains a fixed stratified 80/20 ID split (seed 42), generated from official train. Phúc's baseline results still need rerunning with it. One exact text crosses fit/validation (`train:3067`, `train:3068`); repeated train text includes five conflicting-label groups. No source train/test text overlap. Nothing was silently removed.
- Runtime: experiments ran on CPU with Python 3.12.7, pandas 2.2.3, scikit-learn 1.5.2, PyTorch 2.5.1+cpu and Transformers 4.46.3. The RTX 3050 was not usable by the installed CPU-only PyTorch wheel; the large CUDA wheel download was stopped after 2.5 minutes without progress.
- GitHub: authenticated as `PhucBeossss123`; private repo `PhucBeossss123/an-y` exists. `main` is the shared starting branch. Phúc's result branch is `phuc/baseline-zero-shot`; exact SHAs and links are reported after verification.

## Team handoff status

| Owner | Deliverable | Status |
|---|---|---|
| Phúc | TF-IDF + SVM, zero-shot, prompt comparison and results | Implemented and evaluated on provisional validation and official test; rerun needed when the shared split arrives |
| Huy | Shared data loader/split and BERT-base | Split, loader, and three-seed BERT pipeline on `huy/data-bert`; all three seeds completed with validation-based epoch selection and official-test evaluation. Per-seed and aggregate results are recorded in `doc/cach_lam.md`; checkpoints and predictions remain local under ignored `outputs/bert/`. |
| Tuấn | RoBERTa-base and shared evaluation metrics | Waiting for GitHub username, invitation acceptance and handoff |
| Kiệt | DistilBERT, demo and slide integration | Waiting for GitHub username, invitation acceptance and handoff |

No collaborator invitations have been sent. No member-submitted code has been received. The `huy/data-bert` implementation is prepared by Phúc at the user's request; the reviewed code commit and checks are recorded below.

## Phúc review ban đầu — `huy/data-bert` — 14/10/2026

- Reviewed implementation commit: `de52d58d5610f8b2ae0c8a4fec0e595a0a545a83`.
- Read the complete branch diff. No blocking code issue found in source checksum validation, ID-only split serialization, text/label-only BERT inputs, validation-based epoch selection, test separation, or single-text prediction.
- Checks run: `compileall` passed; both new CLI `--help` commands passed; shared loader returned the expected 2,774/694 rows with only `sample_id`, `text`, and `label`; official test loader returned 1,400 rows; source and split SHA checks passed; new code files stayed below 500 lines; `git diff --check` and `git show --check` passed; prediction function loaded the local pilot checkpoint and returned a binary label.
- At this initial review, training evidence was limited to seed 13, epoch 1 validation: Macro-F1 `0.4349`, sarcastic F1 `0.0115`, confusion matrix `[[521, 0], [172, 1]]`. The pilot was incomplete and was not a final result; this status was superseded by the completed run below.

## Phúc follow-up review — `huy/data-bert` — 14/10/2026

- Reviewed branch at `007e18cf54d1f24746e922c0a18b7fb4ebbd3406`, including the complete shared split/data loader/BERT implementation and the final result documentation.
- No blocking issue found in source/split checksum validation, ID-only split storage, use of text and binary labels as model inputs, per-seed validation checkpoint selection, official-test separation, or binary single-text prediction.
- Completed all three seeds (13, 42, 2026), up to 3 epochs each. For each seed, checked that validation predictions have 694 IDs matching the fixed validation split and test predictions have 1,400 IDs/labels matching the official test; recomputed Macro-F1 and sarcastic-class F1 from the prediction CSVs and confirmed they match `metrics.json`.
- `git diff --check` passed. Model checkpoints, metrics JSON, and prediction CSVs are ignored under `outputs/bert/`; only the summarized real results and command are tracked in the handover documentation.
- Remaining disclosed limitations: the fixed split contains an exact-text pair across fit/validation and repeated training text with conflicting labels; no rows were removed or relabeled. `revision_resolved` is `null` in run metrics although the requested model revision is pinned to `86b5e0934494bd15c9632b12f734a8a67f723594`. The group plan proposes seeds `42, 123, 2026`, while this completed run used `13, 42, 2026`; agree on a common seed set before cross-model comparison and rerun if required. These do not block this reviewed branch; see `doc/cach_lam.md` for the result table and details.
- Follow-up documentation review found stale pilot-only status in `README.md` and `doc/giao_dien.md`; those statements have now been updated to match the completed run. The implementation is unchanged from the reviewed commit above.
