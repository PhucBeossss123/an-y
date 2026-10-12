# Ẩn Ý — iSarcasmEval

Repo chung của nhóm 4 người cho bài toán phát hiện mỉa mai trên iSarcasmEval tiếng Anh, Subtask A. Visibility được xác minh gần nhất là private; trạng thái hiện tại chưa thể kiểm tra ngày 12/10/2026 vì GitHub CLI không đăng nhập được. Với repo private, thành viên cần được Phúc mời và chấp nhận lời mời bằng tài khoản riêng trước khi clone.

## Bắt đầu tại đây

- [Cách tham gia nhóm và quy trình review trước khi push](doc/cach_lam.md)
- [Giao diện dữ liệu, split và prediction dùng chung](doc/giao_dien.md)
- [Phân công và yêu cầu đề tài](doc/phan_cong.md)
- [Tạo/cập nhật task bằng Issue Form](.github/ISSUE_TEMPLATE/team-task.yml) · [Issue tracker](https://github.com/PhucBeossss123/an-y/issues)
- [Rule của repo](AGENTS.md)
- [Trạng thái hiện tại của từng phần](doc/review.md)

Visibility lần xác minh thành công gần nhất được ghi trong `doc/review.md` là private. Ngày 12/10/2026, trạng thái đăng nhập GitHub không hợp lệ nên chưa xác minh được visibility hoặc đồng bộ issue tracker; xem ghi chú mới nhất trong [hướng dẫn cộng tác](doc/cach_lam.md).

## Kết quả Phúc

- **[Xem kết quả Phúc](https://github.com/PhucBeossss123/an-y/blob/phuc/baseline-zero-shot/results/phuc/README.md)** — bảng điểm, trạng thái, link tới prediction, prompt và từng run.
- [Báo cáo phương pháp, kết quả và phân tích lỗi](https://github.com/PhucBeossss123/an-y/blob/phuc/baseline-zero-shot/doc/bao_cao_phuc.md)
- [Chạy lại baseline và zero-shot](https://github.com/PhucBeossss123/an-y/blob/phuc/baseline-zero-shot/doc/cach_lam.md#chay-lai-phan-cua-phuc)

Kết quả Phúc hiện dùng split phát triển tạm seed 42 vì split chung của Huy chưa được bàn giao. Official test đã chạy; các con số trên trang kết quả không thay thế kết quả chạy lại trên split nhóm.

## Nhánh và phần việc

| Thành viên | Nhánh | Phạm vi |
|---|---|---|
| Phúc | `phuc/baseline-zero-shot` | TF-IDF + SVM, zero-shot, thử prompt, báo cáo phần mình |
| Huy | `huy/data-bert` | Dữ liệu/split chung, BERT-base |
| Tuấn | `tuan/roberta-evaluation` | RoBERTa-base, metric chung, tổng hợp |
| Kiệt | `kiet/distilbert-demo` | DistilBERT, demo, ghép slide |

Tên tài khoản GitHub và lời mời của Huy, Tuấn, Kiệt còn chờ Phúc cung cấp/xác nhận. Chưa có code nào của các bạn được ghi nhận là đã nhận hoặc duyệt.

Các phần BERT, RoBERTa, DistilBERT, demo, metric chung và kết quả so sánh toàn nhóm đang **chờ thành viên bàn giao**. Không điền số liệu thay cho họ.
