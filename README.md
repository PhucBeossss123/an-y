# Ẩn Ý — iSarcasmEval

Repo chung của nhóm 4 người cho bài toán phát hiện mỉa mai trên iSarcasmEval tiếng Anh, Subtask A. Repo hiện ở chế độ private; thành viên cần được Phúc mời và chấp nhận lời mời bằng tài khoản riêng trước khi clone.

## Bắt đầu tại đây

- [Cách tham gia nhóm và quy trình review trước khi push](doc/cach_lam.md)
- [Giao diện dữ liệu, split và prediction dùng chung](doc/giao_dien.md)
- [Phân công và yêu cầu đề tài](doc/phan_cong.md)
- [Rule của repo](AGENTS.md)
- [Trạng thái hiện tại của từng phần](doc/review.md)

## Kết quả Phúc

- [Kết quả Phúc](results/phuc/README.md) — bảng điểm, prediction, prompt và run.
- [Báo cáo phương pháp, kết quả và phân tích lỗi](doc/bao_cao_phuc.md)
- [Chạy lại baseline và zero-shot](doc/cach_lam.md#chay-lai-phan-cua-phuc)
- [Mở dashboard Streamlit](doc/streamlit.md)

Kết quả Phúc đã được chạy lại trên shared split seed 42; xem `results/phuc/shared_seed42_20261017/`. Bảng so sánh encoder và prediction ID-only đã được xác minh trên cùng validation/official test tại `results/team/`.

## Nhánh và phần việc

| Thành viên | Nhánh | Phạm vi |
|---|---|---|
| Phúc | `phuc/baseline-zero-shot` | TF-IDF + SVM, zero-shot, thử prompt, báo cáo phần mình |
| Huy | `huy/data-bert` | Dữ liệu/split chung, BERT-base |
| Tuấn | `tuan/roberta-evaluation` | RoBERTa-base, metric chung, tổng hợp |
| Kiệt | `kiet/distilbert-demo` | DistilBERT, demo, ghép slide |

Username do Phúc cung cấp: Huy `qhuyy04`, Tuấn `tuantran11022006-ops`, Kiệt `3ndly`; trạng thái collaborator/invitation chưa được xác minh. Lần rà remote trước đó không thấy nhánh/PR mới của Tuấn hoặc Kiệt. Các phần hoàn thiện trong working tree này là tích hợp theo phân công, không được ghi nhận là code do họ gửi.

Loader/split chung và BERT-base đã được kiểm tra trên seeds `42, 123, 2026`; các phần RoBERTa, DistilBERT, evaluator so sánh, demo và nội dung slide tích hợp đã được hoàn thiện trong working tree. Xem `results/team/`, `doc/team_errors.md`, `doc/slide_nhom.md` và trạng thái review mới nhất trong `doc/review.md`. Checkpoints và dữ liệu raw vẫn bị Git ignore.
