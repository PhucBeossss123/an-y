# Kết quả Phúc — iSarcasmEval English Subtask A

**Trạng thái:** lượt chạy mới nhất dùng shared validation split seed 42, chọn cấu hình/prompt trên validation rồi đánh giá official test. Các run cũ bên dưới được giữ nguyên và vẫn mang nhãn provisional.

- [Báo cáo phương pháp và phân tích lỗi](../../doc/bao_cao_phuc.md)
- [Hướng dẫn nhóm và chạy lại](../../doc/cach_lam.md)
- [Artifact shared split mới nhất](shared_seed42_20261017/README.md)
- [CSV của run provisional đã đóng gói trước đó](summary.csv) · [So sánh prompt provisional](prompt_comparison.csv) · [Lỗi ví dụ](errors.csv)
- [Manifest/checksum dữ liệu](../../reports/data_manifest.json) · [Audit dữ liệu](../../reports/data_audit.json)

## Run provisional đã đóng gói trước đó

| Method/prompt | Split | n | Macro-F1 | F1 sarcastic | Precision / recall sarcastic |
|---|---|---:|---:|---:|---:|
| TF-IDF + SVM | provisional_validation_seed_42 | 694 | 0.5525 | 0.2772 | 0.3936 / 0.2139 |
| TF-IDF + SVM | official_test | 1400 | 0.5708 | 0.2826 | 0.2500 / 0.3250 |
| zero-shot `explicit_negation` | provisional_validation_seed_42 | 694 | 0.5488 | 0.2764 | 0.3725 / 0.2197 |
| zero-shot `original` | provisional_validation_seed_42 | 694 | 0.4925 | 0.3149 | 0.2554 / 0.4104 |
| zero-shot `original` | official_test | 1400 | 0.4103 | 0.1083 | 0.0766 / 0.1850 |
| zero-shot `explicit_negation` | official_test | 1400 | 0.4476 | 0.0428 | 0.0460 / 0.0400 |

Zero-shot prompt selection uses **validation Macro-F1 only** (F1 sarcastic is the tie-break). `explicit_negation` was selected. Test results are reported after the choice and were not used to change it.

## Trạng thái theo phương pháp

- **Run records bên dưới:** các artifact lịch sử được đóng gói từ validation tạm; giữ nguyên để tái tạo kết quả cũ.
- **Shared split:** kết quả SVM và zero-shot mới nhất, kèm predictions/metrics, nằm trong [`shared_seed42_20261017/`](shared_seed42_20261017/); BERT-base và trạng thái các phần nhóm khác xem `doc/cach_lam.md` và `doc/review.md`.
- **Các mô hình RoBERTa/DistilBERT, demo và metric tích hợp:** chưa có branch/PR hoặc kết quả từ Tuấn/Kiệt trong lần rà remote gần nhất.

## Các lượt chạy và artifact

Mỗi thư mục run lưu cấu hình/nguồn/split/seed, metric, prediction và confusion matrix. ID run có timestamp; khi đóng gói lần khác, thư mục cũ không bị ghi đè.

| Run | Method | Split | n | Macro-F1 | F1 sarcastic |
|---|---|---|---:|---:|---:|
| [`baseline-validation-20261007T095331664485Z`](runs/baseline-validation-20261007T095331664485Z/README.md) | baseline | provisional_validation_seed_42 | 694 | 0.5525 | 0.2772 |
| [`baseline-test-20261007T095331664485Z`](runs/baseline-test-20261007T095331664485Z/README.md) | baseline | official_test | 1400 | 0.5708 | 0.2826 |
| [`zero_shot-validation-explicit_negation-20261007T095331664485Z`](runs/zero_shot-validation-explicit_negation-20261007T095331664485Z/README.md) | zero_shot / explicit_negation | provisional_validation_seed_42 | 694 | 0.5488 | 0.2764 |
| [`zero_shot-validation-social_post-20261007T095331664485Z`](runs/zero_shot-validation-social_post-20261007T095331664485Z/README.md) | zero_shot / social_post | provisional_validation_seed_42 | 694 | 0.5016 | 0.3551 |
| [`zero_shot-validation-statement-20261007T095331664485Z`](runs/zero_shot-validation-statement-20261007T095331664485Z/README.md) | zero_shot / statement | provisional_validation_seed_42 | 694 | 0.4996 | 0.3618 |
| [`zero_shot-validation-original-20261007T095331664485Z`](runs/zero_shot-validation-original-20261007T095331664485Z/README.md) | zero_shot / original | provisional_validation_seed_42 | 694 | 0.4925 | 0.3149 |
| [`zero_shot-validation-ironic_or_literal-20261007T095331664485Z`](runs/zero_shot-validation-ironic_or_literal-20261007T095331664485Z/README.md) | zero_shot / ironic_or_literal | provisional_validation_seed_42 | 694 | 0.3072 | 0.4010 |
| [`zero_shot-test-original-20261007T095331664485Z`](runs/zero_shot-test-original-20261007T095331664485Z/README.md) | zero_shot / original | official_test | 1400 | 0.4103 | 0.1083 |
| [`zero_shot-test-explicit_negation-20261007T095331664485Z`](runs/zero_shot-test-explicit_negation-20261007T095331664485Z/README.md) | zero_shot / explicit_negation | official_test | 1400 | 0.4476 | 0.0428 |

Macro-F1 khác với F1 lớp sarcastic. Zero-shot score không phải xác suất hiệu chỉnh; SVM decision margin cũng không phải xác suất.
