# Ẩn Ý — iSarcasmEval

Repo chung của nhóm 4 người cho bài toán phát hiện mỉa mai trên iSarcasmEval tiếng Anh, Subtask A. Repo hiện ở chế độ private; thành viên cần được Phúc mời và chấp nhận lời mời bằng tài khoản riêng trước khi clone.

## Bắt đầu tại đây

- [Cách tham gia nhóm và quy trình review trước khi push](doc/cach_lam.md)
- [Giao diện dữ liệu, split và prediction dùng chung](doc/giao_dien.md)
- [Phân công và yêu cầu đề tài](doc/phan_cong.md)
- [Rule của repo](AGENTS.md)
- [Trạng thái hiện tại của từng phần](doc/review.md)

## Kết quả Phúc

- **[Xem kết quả Phúc](https://github.com/PhucBeossss123/an-y/blob/phuc/baseline-zero-shot/results/phuc/README.md)** — bảng điểm, trạng thái, link tới prediction, prompt và từng run.
- [Báo cáo phương pháp, kết quả và phân tích lỗi](https://github.com/PhucBeossss123/an-y/blob/phuc/baseline-zero-shot/doc/bao_cao_phuc.md)
- [Chạy lại baseline và zero-shot](https://github.com/PhucBeossss123/an-y/blob/phuc/baseline-zero-shot/doc/cach_lam.md#chay-lai-phan-cua-phuc)

Kết quả Phúc hiện dùng split phát triển seed 42; split ID chung đã được chuẩn bị trên nhánh Huy và cần chạy lại kết quả Phúc theo split này trước khi so sánh nhóm. Official test đã chạy; các con số trên trang kết quả không thay thế kết quả chạy lại trên split nhóm.

## Nhánh và phần việc

| Thành viên | Nhánh | Phạm vi |
|---|---|---|
| Phúc | `phuc/baseline-zero-shot` | TF-IDF + SVM, zero-shot, thử prompt, báo cáo phần mình |
| Huy | `huy/data-bert` | Dữ liệu/split chung, BERT-base |
| Tuấn | `tuan/roberta-evaluation` | RoBERTa-base, metric chung, tổng hợp |
| Kiệt | `kiet/distilbert-demo` | DistilBERT, demo, ghép slide |

Tên tài khoản GitHub và lời mời của Huy, Tuấn, Kiệt còn chờ Phúc cung cấp/xác nhận. Chưa có code nào của các bạn được ghi nhận là đã nhận hoặc duyệt.

Loader/split chung và pipeline BERT-base có trong nhánh `huy/data-bert`; ba seed đã chạy đủ và kết quả validation/official-test được ghi tại `doc/cach_lam.md`. RoBERTa, DistilBERT, demo, metric chung và kết quả so sánh toàn nhóm vẫn **chờ bàn giao**. Không điền số liệu thay cho họ.
