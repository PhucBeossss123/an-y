# Giao diện dữ liệu và kết quả dùng chung

## Đầu vào

Mỗi mẫu có các trường:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `sample_id` | chuỗi | ID ổn định để truy về file và split nguồn |
| `text` | chuỗi | Văn bản tiếng Anh cần phân loại; giữ nguyên phủ định và emoji |
| `label` | số nguyên 0/1 | `0 = sincere/không mỉa mai`, `1 = sarcastic/mỉa mai` |

Chỉ đưa `text` vào mô hình. Không dùng nhãn, `rephrase` hay chú giải làm đặc trưng. Nhánh `huy/data-bert` ghi split seed 42 thành danh sách ID; loader trả `sample_id`, `text`, `label` và kiểm tra checksum trước khi nạp. Phần Phúc có thể dùng cùng danh sách ID để chạy lại các kết quả cần thiết.

## Đầu ra dự đoán

Tệp CSV tối thiểu có `sample_id`, `y_true`, `y_pred`; có thể thêm `selected_label` và `selected_score` cho zero-shot. Nhãn dự đoán luôn là số nguyên 0/1. Không diễn giải decision margin của SVM hoặc điểm NLI như xác suất đã hiệu chỉnh.

## Đánh giá

Báo Macro-F1 và F1 lớp sarcastic thành hai cột riêng, cùng precision/recall từng lớp và confusion matrix theo thứ tự nhãn `[0, 1]`. Chọn prompt, siêu tham số và mô hình bằng validation trước khi mở test. Mỗi bảng phải ghi tên split, số mẫu và trạng thái split.

## Trạng thái split hiện tại

- Official train: 3,468 mẫu; official test: 1,400 mẫu.
- Split chung hiện tại: 2,774 mẫu train và 694 mẫu validation, tách phân tầng một lần từ official train với seed 42; ID được lưu trong `data/splits/shared_split_seed42.csv`.
- Official test gồm 1,400 mẫu và không nằm trong file split.
- Audit split seed 42 phát hiện một text trùng qua fit/validation (`train:3067`, `train:3068`) và 5 nhóm train lặp có nhãn xung đột. Không có text trùng train/test; không mẫu nào bị tự động xóa hoặc đổi nhãn.
- BERT-base đã chạy đủ ba seed trên PyTorch CPU; kết quả theo seed và mean ± độ lệch chuẩn mẫu nằm trong `doc/cach_lam.md`. Checkpoint và prediction chi tiết ở `outputs/bert/` chỉ lưu cục bộ, không đưa vào Git.
- BERT/RoBERTa/DistilBERT hiện dùng cùng bộ seeds `42, 123, 2026`; bảng mean ± sample SD và prediction ID-only đã được xác minh tại `results/team/`. Demo local: `python scripts/demo.py`; tổng hợp slide tại `doc/slide_nhom.md`.
