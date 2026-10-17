# Review log

## Phúc baseline / zero-shot

- Reviewed branch: `phuc/baseline-zero-shot`.
- Scope: shared fallback data adapter, TF-IDF + LinearSVC, NLI zero-shot prompts, saved predictions/metrics and handover docs.
- Original experiment commit: `7c9b9b5827f79093a61bb32d50e9044eb0ca435b` (`Add baseline and zero-shot experiments`). The shared framework commit is `ceffa5d` (`Set up shared team workflow`).
- Checks run for this handoff: `python -m compileall -q src scripts`; `scripts/verify_outputs.py` (all baseline/zero-shot rows and saved SVM reload passed); `scripts/package_results.py` (9 append-only run records); one-sentence SVM and zero-shot predictions; `git diff --cached --check` / `git show --check`.
- Tại thời điểm review ban đầu, các kết quả SVM/zero-shot đã lưu dùng split phát triển phân tầng 80/20 seed 42 từ official train; follow-up chạy shared split được ghi ở mục cuối. Một cặp văn bản trùng nằm hai phía fit/validation (`train:3067`, `train:3068`); train có năm nhóm văn bản trùng với nhãn xung đột. Không có văn bản trùng giữa train/test; không tự xóa mẫu.
- Khi tích hợp, giữ nguyên `load_shared_split()` của Huy và bổ sung tùy chọn split tạm cho việc tái tạo kết quả cũ. Lệnh validation SVM/zero-shot mặc định dùng split chung; dùng `--validation-split provisional` để tái tạo run lịch sử. `compileall`, giới hạn 500 dòng, hai lệnh CLI `--help`, và `git diff --check` đều đạt. Thí nghiệm mới chạy sau đó theo mục follow-up dưới đây.
- Runtime: experiments ran on CPU with Python 3.12.7, pandas 2.2.3, scikit-learn 1.5.2, PyTorch 2.5.1+cpu and Transformers 4.46.3. The RTX 3050 was not usable by the installed CPU-only PyTorch wheel; the large CUDA wheel download was stopped after 2.5 minutes without progress.
- GitHub: `PhucBeossss123/an-y` is public and `main` is the default branch. Phúc's result branch is `phuc/baseline-zero-shot`; exact SHAs and links are reported after verification.

## Team handoff status

| Owner | Deliverable | Status |
|---|---|---|
| Phúc | TF-IDF + SVM, zero-shot, so sánh prompt và kết quả | Đã chạy trên shared validation và official test; artifact ở `results/phuc/shared_seed42_20261017/` |
| Huy | Data loader/split chung và BERT-base | Split đã có trong `main`; seed 123 đã được chạy lại và ba seed `42, 123, 2026` được kiểm tra ID/nhãn/metric từ prediction CSV. |
| Tuấn | RoBERTa-base và metric đánh giá chung | Không thấy branch/PR bàn giao trong lần rà remote; pipeline RoBERTa và bảng so sánh chung đã được hoàn thiện trong working tree này, xem `results/team/`. |
| Kiệt | DistilBERT, demo và ghép slide | Không thấy branch/PR bàn giao trong lần rà remote; DistilBERT, demo local và slide tích hợp đã được hoàn thiện trong working tree này, xem `scripts/demo.py` và `doc/slide_nhom.md`. |

Trạng thái lời mời collaborator và quyền truy cập chưa được xác minh. PR báo cáo Huy đã được review và merge; lần rà remote gần nhất không thấy nhánh đóng góp của Kiệt hoặc Tuấn.

## Phúc review tích hợp mô hình nhóm — 18/10/2026

- Huy/BERT: reran seed 123 using the pinned revision and common split; merged it with the existing seed 42 and 2026 artifacts only after source revision/checksum/split hash matched. `scripts/package_team_results.py` recomputed metrics from all validation/test predictions and confirmed exact sample-ID order and gold labels.
- Tuấn/RoBERTa and common metrics: ran seeds 42/123/2026 using the shared runner; each checkpoint was selected on validation Macro-F1 before official-test prediction. Packager verified all 6 prediction files against the shared validation and official test.
- Kiệt/DistilBERT and demo/slide: ran seeds 42/123/2026; packager verified all prediction files. Single-text inference uses local checkpoints, and the demo defaults to the architecture/seed selected by mean validation Macro-F1. Slide-ready integrated content is in `doc/slide_nhom.md`.
- Selection: RoBERTa-base seed 42, selected by mean validation Macro-F1 `0.6151` across three seeds. Test scores were not used to select it. Three error groups and examples across all encoders are documented in `doc/team_errors.md`.
- CPU-only PyTorch 2.5.1, Transformers 4.46.3, Python 3.12.7. Model revisions and official dataset checksums are recorded in per-model run metrics and `results/team/model_comparison.json`. Raw data/checkpoints remain ignored.
- Final checks: package command passed and verified 18 validation/test prediction files across the 3×3 runs; `compileall`, the 500-line check (20 code files, largest 339 lines), CLI help, and `git diff --check` passed. CLI single-text prediction returned a binary label for each encoder. Local demo GET returned HTTP 200 with RoBERTa preselected; POST to the real RoBERTa checkpoint returned `sarcastic` (`y_pred=1`). The temporary server was stopped after the smoke check.
- These are completed team-scope deliverables assembled in Phúc's working tree; they are not represented as new code submitted by Huy, Tuấn, or Kiệt. Collaborator invitation/access state remains unverified.

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

## Review PR #7 của Huy — 17/10/2026

- PR `BERT: bổ sung kết quả seed 123 và báo cáo`, head `953b62b0e1eb94e6487971afd09f75312cffe012`, đã được merge vào `main` tại `7c0ae01c2d53c39f85826032437d6eeba8674440`.
- Review phát hiện default `scripts/train_bert.py` còn dùng seed `13, 42, 2026` trong khi báo cáo so sánh `42, 123, 2026`. Đã đổi default sang bộ seed báo cáo; đồng thời xóa dòng trắng thừa cuối `doc/bao_cao_bert.md`.
- `git diff --check` sạch trước khi merge. Các phép tính tổng hợp và precision/recall khớp với các số đã công bố; prediction chi tiết của seed 123 không có trong workspace nên chưa tính lại metric từ CSV nguồn.
- Không thấy branch hoặc PR của Kiệt (`3ndly`) hay Tuấn (`tuantran11022006-ops`) trong danh sách remote đã rà.

## Phúc rerun trên shared split — 17/10/2026

- Đã chạy SVM validation và cả 5 prompt zero-shot trên `shared_validation_seed_42` (694 mẫu); sau khi chốt `C=0.5`, `class_weight=balanced` và prompt `explicit_negation` theo validation Macro-F1, mới chấm official test (1,400 mẫu).
- Kết quả mới được lưu trong `results/phuc/shared_seed42_20261017/`, gồm metrics JSON, prediction CSV ID-only, prompt comparison và summary. Không đưa raw text hay checkpoint vào Git.
- Kết quả thật: SVM shared validation Macro-F1/F1 sarcastic `0.5525/0.2772`, official test `0.5708/0.2826`; zero-shot selected prompt shared validation `0.5488/0.2764`, official test `0.4476/0.0428`. Prompt gốc official test `0.4103/0.1083`. Lựa chọn không đổi theo test.
- Kiểm tra cục bộ sau cập nhật: compileall, line limit, CLI help, và diff check đạt. Không tìm thấy branch/PR của Kiệt hoặc Tuấn trong remote branch list đang clone.
