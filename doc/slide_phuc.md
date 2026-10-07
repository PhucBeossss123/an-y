# Nội dung hai slide phần Phúc

## Slide 1 — Dữ liệu, baseline và quy trình

- iSarcasmEval English Subtask A; phân loại sarcastic / sincere.
- Giao diện chung: `sample_id`, `text`, `label`; chỉ tweet vào đặc trưng; không dùng rephrase hay annotation.
- Baseline: word TF-IDF unigram/bigram + Linear SVM; vectorizer fit trên train; chọn cấu hình trên validation.
- Split hiện tại đang chờ Huy: validation tạm phân tầng seed 42; test chính thức được giữ riêng.
- Metric chính: Macro-F1 và F1 sarcastic (báo riêng), precision/recall và confusion matrix.

**Validation tạm:** 3,468 train → 2,774 fit / 694 validation (seed 42); 1,400 test chính thức.

| Kết quả | Macro-F1 | F1 sarcastic |
|---|---:|---:|
| TF-IDF + SVM validation | 0.5525 | 0.2772 |
| TF-IDF + SVM official test | 0.5708 | 0.2826 |

## Slide 2 — Zero-shot, prompt và lỗi

- Checkpoint NLI: `cross-encoder/nli-deberta-v3-small` (revision `fa2804872c3b4bd748f38c0185cc85775361e735`); zero-shot pipeline, không fine-tune.
- So sánh 5 cách diễn đạt; ánh xạ candidate label → `0/1` được khai báo tường minh.
- Chọn bằng Macro-F1 validation; sau đó so sánh prompt gốc và prompt được chọn trên cùng test.
- Bàn luận theo kết quả thực: phủ định, emoji, ngữ cảnh/nghĩa bề mặt; ví dụ phải có ID và nhãn thật/dự đoán.
- Giới hạn: score NLI không phải xác suất hiệu chỉnh; split tạm cần thay bằng split Huy.

| Prompt | Macro-F1 validation | F1 sarcastic validation |
|---|---:|---:|
| Gốc | 0.4925 | 0.3149 |
| Statement | 0.4996 | 0.3618 |
| Explicit negation (chọn) | 0.5488 | 0.2764 |
| Social post | 0.5016 | 0.3551 |
| Ironic or literal | 0.3072 | 0.4010 |

Official test: prompt gốc `0.4103 / 0.1083`, prompt được chọn `0.4476 / 0.0428` (Macro-F1 / F1 sarcastic). Lựa chọn được chốt trên validation; tăng Macro-F1 không đi kèm tăng F1 sarcastic. Ví dụ lỗi: sarcasm bề mặt tích cực (`test:84`) và sincere có emoji (`test:20`); xem `doc/results_phuc.md`.
