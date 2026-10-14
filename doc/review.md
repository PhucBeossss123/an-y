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
| Huy | Shared data loader/split and BERT-base | Split, loader, and three-seed training/prediction pipeline prepared on `huy/data-bert`; only a seed 13, epoch 1 validation pilot ran (Macro-F1 0.4349, sarcastic F1 0.0115). The CPU-only run stopped before epoch 2 completed; no official-test result or complete three-seed report. Its checkpoint is local under ignored `outputs/bert/` only. |
| Tuấn | RoBERTa-base and shared evaluation metrics | Waiting for GitHub username, invitation acceptance and handoff |
| Kiệt | DistilBERT, demo and slide integration | Waiting for GitHub username, invitation acceptance and handoff |

No collaborator invitations have been sent. No member-submitted code has been received. The `huy/data-bert` implementation is prepared by Phúc at the user's request; the reviewed code commit and checks are recorded below.

## Phúc review — `huy/data-bert` — 14/10/2026

- Reviewed implementation commit: `de52d58d5610f8b2ae0c8a4fec0e595a0a545a83`.
- Read the complete branch diff. No blocking code issue found in source checksum validation, ID-only split serialization, text/label-only BERT inputs, validation-based epoch selection, test separation, or single-text prediction.
- Checks run: `compileall` passed; both new CLI `--help` commands passed; shared loader returned the expected 2,774/694 rows with only `sample_id`, `text`, and `label`; official test loader returned 1,400 rows; source and split SHA checks passed; new code files stayed below 500 lines; `git diff --check` and `git show --check` passed; prediction function loaded the local pilot checkpoint and returned a binary label.
- Training evidence is limited to seed 13, epoch 1 validation: Macro-F1 `0.4349`, sarcastic F1 `0.0115`, confusion matrix `[[521, 0], [172, 1]]`. Training stopped before epoch 2 completed on CPU. Seeds 42 and 2026, final three-seed metrics, and official-test evaluation remain undone; the pilot is not a final result.
