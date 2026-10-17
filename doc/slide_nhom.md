# Nội dung slide tích hợp nhóm — iSarcasmEval English Subtask A

## Slide 1 — Bài toán và dữ liệu

- Phân loại tweet tiếng Anh thành sincere/non-sarcastic (`0`) hoặc sarcastic (`1`).
- Nguồn: iSarcasmEval, revision `dfc708b53bde1bb571abfb5692f63231c2232195`; checksum gốc và split hash được lưu trong metrics.
- Chia train chính thức thành 2.774 fit / 694 validation (split phân tầng seed 42); giữ nguyên 1.400 mẫu official test.
- Chỉ dùng tweet/text và nhãn nhị phân; không đưa rephrase, rationale hay annotation vào đặc trưng.

## Slide 2 — Phương pháp và quy trình đánh giá

- Baseline: word TF-IDF unigram/bigram + Linear SVM; chọn cấu hình bằng validation.
- Zero-shot: NLI DeBERTa-v3-small, so sánh 5 prompt; chọn prompt bằng validation.
- Fine-tuning: BERT-base, RoBERTa-base, DistilBERT; cùng split, 3 epoch tối đa, max length 128, batch 4, gradient accumulation 4, learning rate `2e-5`, seeds 42/123/2026.
- Chọn epoch/checkpoint trên validation Macro-F1, tie-break bằng F1 sarcastic; chỉ sau đó mới chạy official test.
- Metric báo cáo riêng Macro-F1 và F1 sarcastic; mean ± sample SD (`ddof=1`) cho ba seed của mỗi encoder.

## Slide 3 — Baseline và zero-shot

| Phương pháp | Validation Macro-F1 / F1 sarcastic | Official test Macro-F1 / F1 sarcastic |
|---|---:|---:|
| TF-IDF + SVM (một cấu hình đã chọn) | 0.5525 / 0.2772 | 0.5708 / 0.2826 |
| Zero-shot `explicit_negation` (chọn trên validation) | 0.5488 / 0.2764 | 0.4476 / 0.0428 |
| Zero-shot prompt gốc | 0.4925 / 0.3149 | 0.4103 / 0.1083 |

Điểm SVM là decision margin; điểm NLI không phải xác suất đã hiệu chỉnh. Test cho thấy prompt có Macro-F1 cao hơn vẫn có F1 sarcastic thấp hơn.

## Slide 4 — So sánh ba encoder

| Encoder | Validation Macro-F1 | Validation F1 sarcastic | Test Macro-F1 | Test F1 sarcastic |
|---|---:|---:|---:|---:|
| BERT-base | 0.6002 ± 0.0125 | 0.3798 ± 0.0513 | 0.5870 ± 0.0124 | 0.3234 ± 0.0200 |
| RoBERTa-base | **0.6151 ± 0.0386** | 0.3777 ± 0.0832 | **0.6168 ± 0.0295** | **0.3542 ± 0.0723** |
| DistilBERT | 0.5597 ± 0.0438 | 0.2752 ± 0.0962 | 0.5717 ± 0.0236 | 0.2631 ± 0.0630 |

Demo chọn **RoBERTa seed 42** bằng mean validation Macro-F1 cao nhất của kiến trúc. Không chọn mô hình dựa trên test.

## Slide 5 — Lỗi chung của các encoder

- Bỏ lỡ tương phản/khen ngược: `test:9` (cả ba), `test:21` (cả ba), `test:2` (DistilBERT).
- Bị từ vựng cảm xúc hoặc emoji chi phối: `test:10`, `test:11`, `test:14`, `test:28`.
- Khó với ngữ cảnh/chủ đề và phàn nàn: `test:3`, `test:6`, `test:7`, `test:13`, `test:15`.
- Bảng nhãn thật/dự đoán và nhận xét từng ví dụ: [team error analysis](team_errors.md). Các ID trỏ tới official test; không chép tweet vào slide/repo.

## Slide 6 — Demo, kết luận và giới hạn

- Demo local nhập một câu tiếng Anh; có SVM, zero-shot và cả ba encoder. Khi đã đóng gói, model được validation chọn sẽ mặc định hiển thị.
- Chạy: `python scripts/demo.py`; địa chỉ mặc định `http://127.0.0.1:8765`.
- RoBERTa đạt điểm trung bình cao nhất trong so sánh ba encoder, nhưng F1 sarcastic vẫn 0.3542 trên test; không nên chỉ nhìn Macro-F1.
- Tập sarcastic nhỏ hơn nhiều (173/694 validation, 200/1.400 test); sarcasm còn phụ thuộc ngữ cảnh và tín hiệu pragmatics. Kết quả là nhãn cứng, không phải xác suất calibrated.

Nguồn số liệu và prediction ID-only: [`results/team/`](../results/team/). Checkpoint/raw data không được commit.
