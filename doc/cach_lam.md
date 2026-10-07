# Cách làm và bàn giao phần Phúc

## Phạm vi / giao diện

Phần này chỉ gồm TF-IDF + Linear SVM và zero-shot NLI cho iSarcasmEval tiếng Anh, Subtask A. Đầu vào adapter là `sample_id`, `text`, `label`; `0` là sincere/không mỉa mai, `1` là sarcastic/mỉa mai. Train lấy text từ cột `tweet`; không dùng `rephrase` hoặc các cột chú giải làm đặc trưng. Dự đoán lưu `sample_id`, `y_true`, `y_pred`; score SVM không phải xác suất. Score zero-shot là điểm NLI chuẩn hóa giữa hai nhãn, không được xem là xác suất đã hiệu chỉnh.

## Dữ liệu và giả định hiện tại

Nguồn chính thức: `https://github.com/iabufarha/iSarcasmEval`; tải Subtask A tiếng Anh gồm `train/train.En.csv` và `test/task_A_En_test.csv`. Script ghi revision nguồn, SHA-256, README và LICENSE vào `data/raw/manifest.json`. Dữ liệu được tải cục bộ, không đưa vào Git.

**Chờ Huy:** loader và split chung. Cho đến khi bàn giao, adapter này chia phân tầng 80/20 trên train với seed 42; đây là split phát triển tạm, không phải split chung đã được nhóm duyệt. Test chính thức được giữ riêng. Khi nhận split đã chốt, thay adapter và chạy lại các thí nghiệm trước khi kết luận.

## Quy trình

1. Tạo `.venv`, cài `requirements.txt`, tải dữ liệu bằng `scripts/download_data.py`.
2. Baseline thử 6 cấu hình nhỏ (`C` 0.5/1/2, class weight none/balanced); chọn Macro-F1 validation, phân xử bằng F1 sarcastic. TF-IDF được fit trong pipeline trên train của split phát triển. Sau khi chọn, test script refit trên toàn bộ official train và đánh giá official test.
3. Zero-shot dùng `cross-encoder/nli-deberta-v3-small` (revision `fa2804872c3b4bd748f38c0185cc85775361e735`), nhãn gốc `sarcastic`/`sincere` và bốn biến thể đã lưu trong `config/prompts.json`. Model card xác nhận zero-shot Transformers pipeline, huấn luyện trên SNLI/MultiNLI và Apache-2.0. Cùng checkpoint/split/batch cho tất cả. Chọn prompt theo Macro-F1 validation, dùng F1 sarcastic khi hòa. Test chạy sau lựa chọn và so sánh prompt gốc với prompt được chọn trên cùng official test.
4. Metric tạm theo scikit-learn, chờ thay/đối chiếu với evaluator của Tuấn. Bảng validation và bảng test tách riêng.
5. Prediction lỗi được lưu theo ID; cần Phúc đọc từng lỗi, thêm giải thích có căn cứ và xem các nhóm phủ định, emoji, ngữ cảnh trước khi đưa nhận xét vào báo cáo.

## Tình trạng bằng chứng

- Repo ban đầu tại workspace trống, không có `.git`, code, tài liệu dự án, data, hay remote.
- Python 3.12.7 có tại Miniconda; Python alias mặc định không hoạt động. Máy có RTX 3050 với 4 GB VRAM; `nvidia-smi` xác nhận CUDA runtime 12.7. PyTorch trong `.venv` là bản CPU-only (`2.5.1+cpu`); thử cài wheel CUDA 12.4 chính thức nhưng dừng tải wheel 2.51 GB sau 2.5 phút không có tiến triển hiển thị.
- Git có cấu hình identity `HuynhtTranNhatPhuc`; GCM liệt kê `PhucBeossss123`, nhưng kiểm tra `/user` qua GitHub API trả HTTP 401. `gh` chưa cài; chưa tạo repo/remote đích.
- Superpowers không có trong tool catalog; áp dụng quy trình tương đương đọc hiểu → lập kế hoạch → thực hiện → kiểm thử → review theo prompt.
- Baseline validation và official test; 5 prompt zero-shot validation và prompt gốc/được chọn trên official test đã chạy. Metric thật được tóm tắt tại `doc/results_phuc.md`; file chi tiết tại thư mục `reports/`.
- Review: hoàn tất review local; log và lỗi đã sửa/chờ ghi ở `doc/review.md`. Chưa có code thành viên nào để review.

## Kết quả thực nghiệm

Baseline tốt nhất validation là `C=0.5`, `class_weight=balanced`; Macro-F1 `0.5525`, F1 sarcastic `0.2772`, confusion matrix `[[464,57],[136,37]]`. Kết quả chỉ dùng split phát triển tạm seed 42, cần chạy lại theo split chung của Huy. Tập train có 3,468 mẫu/867 sarcastic, validation tạm 694 mẫu/173 sarcastic, official test 1,400 mẫu/200 sarcastic. Audit phát hiện 20 dòng train thuộc các nhóm text lặp, trong đó 5 nhóm có nhãn xung đột; ID chi tiết ở `reports/data_audit.json`. Một text trùng qua fit/validation (`train:3067` và `train:3068`, cùng nhãn 0), không có text trùng giữa train/test. Không mẫu nào bị tự động xóa. Test có 6 dòng thuộc các nhóm text trùng nội bộ.

Zero-shot chọn `explicit_negation` bằng validation Macro-F1 `0.5488` (F1 sarcastic `0.2764`); prompt gốc đạt `0.4925` (`0.3149`). Trên official test, prompt gốc: Macro-F1 `0.4103`, F1 sarcastic `0.1083`; prompt chọn: `0.4476`, F1 sarcastic `0.0428`. Prompt chọn cao hơn Macro-F1 nhưng thấp hơn F1 sarcastic; lựa chọn không đổi theo test. Kết quả và ví dụ có trong `doc/results_phuc.md`.
