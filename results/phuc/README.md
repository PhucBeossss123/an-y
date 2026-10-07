# Kết quả Phúc — iSarcasmEval English Subtask A

**Trạng thái:** baseline và zero-shot đã chạy; prompt được chọn trên validation; official test đã đánh giá. Validation hiện là split phát triển tạm seed 42, chưa phải split chung của nhóm.

- [Báo cáo phương pháp và phân tích lỗi](../../doc/bao_cao_phuc.md)
- [Hướng dẫn nhóm và chạy lại](../../doc/cach_lam.md)
- [Tổng hợp CSV](summary.csv) · [So sánh prompt validation](prompt_comparison.csv) · [Lỗi ví dụ](errors.csv)
- [Manifest/checksum dữ liệu](../../reports/data_manifest.json) · [Audit dữ liệu](../../reports/data_audit.json)

## Điểm chính

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

- **TF-IDF + SVM:** đã chạy 694 mẫu validation tạm và 1,400 mẫu official test; test model refit trên toàn bộ official train.
- **Zero-shot:** cả 5 prompt đã chạy trên 694 validation tạm; prompt gốc và prompt chọn chạy trên cùng 1,400 official test.
- **Chạy lại theo split chung:** đang chờ Huy bàn giao loader/split; các điểm validation hiện tại chưa phải kết quả so sánh chung.
- **Các mô hình BERT/RoBERTa/DistilBERT, demo và metric tích hợp:** chờ thành viên bàn giao; chưa có kết quả để so sánh.

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
