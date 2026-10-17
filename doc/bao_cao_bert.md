# Dữ liệu, phương pháp và kết quả BERT-base

## Dữ liệu

Thí nghiệm dùng iSarcasmEval tiếng Anh, Subtask A, bài toán phân loại nhị phân ý định mỉa mai. Văn bản đầu vào lấy từ trường `tweet` của tập train và trường `text` của official test; không dùng `rephrase` hoặc chú giải bổ sung làm đặc trưng. Nhãn `0` là sincere/không mỉa mai, `1` là sarcastic/mỉa mai.

Nguồn dữ liệu là revision `dfc708b53bde1bb571abfb5692f63231c2232195` của repo iSarcasmEval. Theo checksum đã ghi trong manifest, train có 3.468 mẫu (2.601 nhãn 0, 867 nhãn 1); official test có 1.400 mẫu (1.200 nhãn 0, 200 nhãn 1). Từ train, nhóm dùng một split phân tầng cố định 80/20 với seed chia `42`: 2.774 mẫu train-fit (2.080 nhãn 0, 694 nhãn 1) và 694 mẫu validation (521 nhãn 0, 173 nhãn 1). Official test được giữ riêng.

Kiểm toán phát hiện 20 dòng train thuộc các nhóm văn bản lặp, trong đó có 5 nhóm mang nhãn xung đột. Cặp `train:3067` và `train:3068` có cùng văn bản và cùng nhãn 0 nhưng nằm hai phía fit/validation. Không có văn bản trùng chính xác giữa official train và official test; official test có 6 dòng thuộc các nhóm lặp nội bộ. Không tự ý xóa hoặc sửa nhãn.

## Phương pháp và cấu hình

Fine-tune `bert-base-uncased`, khóa revision yêu cầu `86b5e0934494bd15c9632b12f734a8a67f723594`, với đầu phân loại nhị phân. Tokenizer tương ứng dùng truncation và giới hạn độ dài 128 token; padding động theo batch. Mô hình nhận văn bản và nhãn nhị phân, không nhận các cột chú giải.

Cấu hình huấn luyện: AdamW, learning rate `2e-5`, weight decay `0.01`, warmup tuyến tính 10%, tối đa 3 epoch; batch train 4, batch eval 8, tích lũy gradient 4 bước (batch hiệu dụng xấp xỉ 16). Mỗi seed đặt lại seed cho Python, NumPy và PyTorch. Chọn epoch theo Macro-F1 validation; nếu bằng nhau thì ưu tiên F1 lớp sarcastic. Sau khi chọn checkpoint bằng validation, đánh giá official test một lần. Báo cáo Macro-F1 và F1 riêng lớp sarcastic; độ lệch chuẩn giữa seed là độ lệch chuẩn mẫu (`ddof=1`).

Bộ seed báo cáo thống nhất theo kế hoạch chung: **42, 123, 2026**. ZIP ban đầu có kết quả 13, 42, 2026; đã chạy bổ sung seed 123 với cùng split, revision và siêu tham số. Seed 13 không được đưa vào bảng so sánh chung.

## Kết quả

| Seed | Epoch chọn | Validation Macro-F1 | Validation F1 sarcastic | Test Macro-F1 | Test F1 sarcastic |
|---:|---:|---:|---:|---:|---:|
| 42 | 2 | 0.6133 | 0.4362 | 0.5729 | 0.3465 |
| 123 | 3 | 0.5987 | 0.3673 | 0.5919 | 0.3112 |
| 2026 | 3 | 0.5884 | 0.3358 | 0.5963 | 0.3125 |
| **Trung bình ± SD mẫu** | — | **0.6002 ± 0.0125** | **0.3798 ± 0.0513** | **0.5870 ± 0.0124** | **0.3234 ± 0.0200** |

Seed 42 và 2026 lấy từ artifact BERT local đã có; seed 123 được chạy bổ sung bằng cùng pinned revision, split và siêu tham số. Đã xác minh lại ID, nhãn và metric của cả ba seed từ prediction CSV. Trên test seed 123, precision/recall sarcastic lần lượt là 0.2869/0.3400; ma trận nhầm lẫn theo thứ tự nhãn `[0,1]` là `[[1031,169],[132,68]]`. Trong 200 mẫu mỉa mai ở test, mô hình bắt đúng 68 mẫu.

### Năm lỗi BERT trên official test

Các ví dụ dưới đây lấy từ dự đoán seed 123; `test:N` là ID theo thứ tự dòng trong official test. Nhãn: `0 = sincere`, `1 = sarcastic`.

| ID mẫu | Nhãn thật | Dự đoán | Giải thích ngắn |
|---|---|---|---|
| `test:2` | 1 (sarcastic) | 0 (sincere) | Phàn nàn rằng chính quyền muốn người dân tiêm nhắc lại nhưng website không hoạt động; mỉa mai dựa vào tương phản giữa mục tiêu và trải nghiệm. |
| `test:21` | 1 (sarcastic) | 0 (sincere) | “Clean driver”/“never makes dirty moves” là lời khen trái ngược nhằm châm biếm tay đua; cần hiểu ngữ cảnh thể thao và ý đối lập. |
| `test:56` | 1 (sarcastic) | 0 (sincere) | Nói cảm thấy an toàn hơn vì cảnh sát không điều tra tội phạm sau khi xảy ra; sự vô lý trong quan hệ nguyên nhân-kết quả tạo mỉa mai. |
| `test:3` | 0 (sincere) | 1 (sarcastic) | Đây là lời khuyên thực dụng về đổi tên thiết bị Bluetooth để người kia nhớ; cụm “villainous pro tip” mang sắc thái đùa nên dễ kích hoạt dự đoán mỉa mai. |
| `test:15` | 0 (sincere) | 1 (sarcastic) | Người viết bực vì bị từ chối thuốc đau đầu do bị nghĩ là dưới 16 tuổi; nhấn “Sixteen. Six. Teen.” là nhấn mạnh cảm xúc, không nhất thiết là mỉa mai. |

### Hạn chế

- Dữ liệu lệch lớp: sarcastic chiếm 25% train và khoảng 14,3% official test. Seed 123 có recall sarcastic 0.3350 và precision 0.2876, cho thấy còn bỏ sót nhiều trường hợp và tạo nhiều false positive.
- Ý định mỉa mai thường phụ thuộc ngữ cảnh, kiến thức ngoài văn bản, người viết và sắc thái diễn đạt. BERT chỉ nhìn chuỗi token; câu khen mỉa, phủ định, hàm ý và châm biếm xã hội có thể giống câu nghĩa đen.
- Có mẫu văn bản trùng và nhãn xung đột trong train; một cặp trùng nằm ở hai phía fit/validation. Việc giữ nguyên theo manifest giúp tái lập split nhưng có thể làm điểm số dao động và khiến cùng nội dung có tín hiệu giám sát không nhất quán.
- Kết quả dựa trên một split train/validation cố định và một official test; ba seed thể hiện một phần biến thiên khởi tạo, không thay thế đánh giá trên bộ dữ liệu ngoài hoặc nhiều split độc lập.
- Lượt chạy bổ sung seed 123 dùng CPU-only nên tốn thời gian. Log ghi revision đã yêu cầu nhưng `revision_resolved` là `null`; không có SHA resolved riêng do thư viện không trả trường này trong lần chạy.
