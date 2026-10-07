# Review log

## Phúc baseline / zero-shot

- Reviewed branch: `phuc/baseline-zero-shot`.
- Scope: shared fallback data adapter, TF-IDF + LinearSVC, NLI zero-shot prompts, saved predictions/metrics and handover docs.
- Original experiment commit: `7c9b9b5827f79093a61bb32d50e9044eb0ca435b` (`Add baseline and zero-shot experiments`). The shared framework commit is `ceffa5d` (`Set up shared team workflow`).
- Checks run for this handoff: `python -m compileall -q src scripts`; `scripts/verify_outputs.py` (all baseline/zero-shot rows and saved SVM reload passed); `scripts/package_results.py` (9 append-only run records); one-sentence SVM and zero-shot predictions; `git diff --cached --check` / `git show --check`.
- Data limitation: current validation is a provisional stratified 80/20 split of official train with seed 42. One exact text crosses fit/validation (`train:3067`, `train:3068`); repeated train text includes five conflicting-label groups. No source train/test text overlap. Nothing was silently removed. Rerun on Huy's agreed split when delivered.
- Runtime: experiments ran on CPU with Python 3.12.7, pandas 2.2.3, scikit-learn 1.5.2, PyTorch 2.5.1+cpu and Transformers 4.46.3. The RTX 3050 was not usable by the installed CPU-only PyTorch wheel; the large CUDA wheel download was stopped after 2.5 minutes without progress.
- GitHub: authenticated as `PhucBeossss123`; private repo `PhucBeossss123/an-y` exists. `main` is the shared starting branch and is set as default. Phúc's result branch is `phuc/baseline-zero-shot`; exact SHAs and links are reported after verification.

## Team handoff status

| Owner | Deliverable | Status |
|---|---|---|
| Phúc | TF-IDF + SVM, zero-shot, prompt comparison and results | Implemented and evaluated on provisional validation and official test; rerun needed when the shared split arrives |
| Huy | Shared data loader/split and BERT-base | Waiting for GitHub username, invitation acceptance and handoff |
| Tuấn | RoBERTa-base and shared evaluation metrics | Waiting for GitHub username, invitation acceptance and handoff |
| Kiệt | DistilBERT, demo and slide integration | Waiting for GitHub username, invitation acceptance and handoff |

No collaborator invitations have been sent. No member code has been received, run or approved for push. Phúc must approve an exact `head_sha` before a member pushes.
