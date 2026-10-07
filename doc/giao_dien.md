# Giao diện dữ liệu và kết quả dùng chung

## Đầu vào

Mỗi mẫu có các trường:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `sample_id` | chuỗi | ID ổn định để truy về file và split nguồn |
| `text` | chuỗi | Văn bản tiếng Anh cần phân loại; giữ nguyên phủ định và emoji |
| `label` | số nguyên 0/1 | `0 = sincere/không mỉa mai`, `1 = sarcastic/mỉa mai` |

Chỉ đưa `text` vào mô hình. Không dùng nhãn, `rephrase` hay chú giải làm đặc trưng. Phần Phúc hiện dùng adapter tạm đọc `tweet` từ train và file test chính thức; chưa coi split validation seed 42 là split chung. Huy bàn giao loader/split cố định thì cả nhóm thay adapter và chạy lại các kết quả cần thiết.

## Đầu ra dự đoán

Tệp CSV tối thiểu có `sample_id`, `y_true`, `y_pred`; có thể thêm `selected_label` và `selected_score` cho zero-shot. Nhãn dự đoán luôn là số nguyên 0/1. Không diễn giải decision margin của SVM hoặc điểm NLI như xác suất đã hiệu chỉnh.

## Đánh giá

Báo Macro-F1 và F1 lớp sarcastic thành hai cột riêng, cùng precision/recall từng lớp và confusion matrix theo thứ tự nhãn `[0, 1]`. Chọn prompt, siêu tham số và mô hình bằng validation trước khi mở test. Mỗi bảng phải ghi tên split, số mẫu và trạng thái split.

## Trạng thái split hiện tại

- Official train: 3,468 mẫu; official test: 1,400 mẫu.
- Validation của Phúc: 694 mẫu, tách phân tầng seed 42 từ official train để phát triển; **tạm, chưa dùng so sánh chính thức**.
- Không có text trùng giữa official train và official test. Có một text trùng qua fit/validation tạm và các nhóm trùng train có nhãn mâu thuẫn; xem `results/phuc/README.md` và `reports/data_audit.json`.
- Loader/split chung của Huy: **đang chờ bàn giao**.
