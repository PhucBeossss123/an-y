# Báo cáo kết quả phần Phúc

## Trạng thái

Đã chạy baseline TF-IDF + LinearSVC và zero-shot NLI trên iSarcasmEval English Subtask A. Official test đã được đánh giá sau khi lựa chọn cấu hình trên validation. Các kết quả trong báo cáo dùng split phát triển tạm, phân tầng 80/20 từ official train với seed 42. Split chung của Huy hiện đã có trong `main`; cần chạy lại các thí nghiệm trên split này trước khi xem là kết quả so sánh cuối của nhóm.

| Phương pháp | Trạng thái | Dữ liệu |
|---|---|---|
| TF-IDF + SVM | Đã chạy validation, refit trên toàn bộ official train và chạy official test | 694 validation tạm; 1,400 test |
| Zero-shot, 5 prompt | Đã chạy cả 5 prompt trên validation; chọn theo Macro-F1 | 694 validation tạm |
| Zero-shot gốc và prompt chọn | Đã chạy cả hai trên cùng official test; không thay lựa chọn theo test | 1,400 test |
| Chạy lại trên split chung | Chưa chạy; chờ loader/split Huy bàn giao | — |

## Thiết lập

- Dữ liệu nguồn: [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval), revision `dfc708b53bde1bb571abfb5692f63231c2232195`; manifest/checksum tại [reports/data_manifest.json](../reports/data_manifest.json), license nguồn tại [reports/source_LICENSE.txt](../reports/source_LICENSE.txt).
- Train có 3,468 mẫu (2,601 sincere; 867 sarcastic). Official test có 1,400 (1,200 sincere; 200 sarcastic).
- Baseline: word TF-IDF unigram/bigram, `sublinear_tf=True`, `LinearSVC`; chọn trong 6 cấu hình nhỏ bằng Macro-F1 validation, F1 sarcastic làm tie-break. Cấu hình chọn: `C=0.5`, `class_weight=balanced`. Model được refit trên official train trước khi test.
- Zero-shot: `cross-encoder/nli-deberta-v3-small`, revision `fa2804872c3b4bd748f38c0185cc85775361e735`; Transformers `zero-shot-classification`, không fine-tune. Giữ cùng model, split, batch và thiết lập cho 5 prompt. Label map tường minh; NLI score không phải xác suất hiệu chỉnh.
- Runtime đã kiểm chứng: Python 3.12.7, pandas 2.2.3, scikit-learn 1.5.2, PyTorch 2.5.1+cpu, Transformers 4.46.3. GPU RTX 3050 không dùng được với wheel CPU-only; thí nghiệm chạy CPU.

## Baseline — chọn trên validation, đánh giá test một lần

| Split | n | Macro-F1 | F1 sarcastic | P/R sarcastic | Confusion matrix (nhãn 0,1) |
|---|---:|---:|---:|---:|---|
| Validation tạm, seed 42 | 694 | 0.5525 | 0.2772 | 0.3936 / 0.2139 | `[[464,57],[136,37]]` |
| Official test | 1,400 | 0.5708 | 0.2826 | 0.2500 / 0.3250 | `[[1005,195],[135,65]]` |

## Zero-shot — prompt selection trên validation

| Prompt | Macro-F1 | F1 sarcastic | P/R sarcastic | Được chọn |
|---|---:|---:|---:|---|
| `original` | 0.4925 | 0.3149 | 0.2554 / 0.4104 | Không |
| `statement` | 0.4996 | 0.3618 | 0.2758 / 0.5260 | Không |
| `explicit_negation` | **0.5488** | 0.2764 | 0.3725 / 0.2197 | **Có** |
| `social_post` | 0.5016 | 0.3551 | 0.2744 / 0.5029 | Không |
| `ironic_or_literal` | 0.3072 | **0.4010** | 0.2569 / 0.9133 | Không |

`explicit_negation` được chốt theo Macro-F1 cao nhất; không cần dùng tie-break. `ironic_or_literal` có F1 sarcastic cao nhất nhưng Macro-F1 thấp do recall sincere `0.1228`.

## Zero-shot — official test, không đổi prompt theo test

| Prompt | n | Macro-F1 | F1 sarcastic | P/R sarcastic | Confusion matrix |
|---|---:|---:|---:|---:|---|
| Gốc (`original`) | 1,400 | 0.4103 | 0.1083 | 0.0766 / 0.1850 | `[[754,446],[163,37]]` |
| Được chọn (`explicit_negation`) | 1,400 | 0.4476 | 0.0428 | 0.0460 / 0.0400 | `[[1034,166],[192,8]]` |

Trên test, prompt chọn cao hơn Macro-F1 `0.0373` nhưng F1 sarcastic thấp hơn `0.0656` so với prompt gốc. Đây là trade-off được báo lại; lựa chọn vẫn dựa duy nhất trên validation.

## Phân tích lỗi

Các ví dụ dưới đây là dự đoán sai thật trên official test. Nhận xét là đọc thủ công từng mẫu, không khẳng định nhãn nguồn không có sai sót.

| Method/prompt | ID | Thật → dự đoán | Nhóm | Nhận xét |
|---|---|---:|---|---|
| SVM | `test:2` | 1 → 0 | Ngữ cảnh/đối lập | Phàn nàn về website hỏng được diễn đạt như hệ quả của việc nhiều người muốn dùng; mô hình đọc như khiếu nại theo nghĩa bề mặt. |
| SVM | `test:4` | 0 → 1 | Emoji/câu ngắn | Câu tích cực chân thành kèm emoji bị gán mỉa mai. |
| SVM | `test:20` | 0 → 1 | Emoji/câu ngắn | Lời chúc ngủ ngon chân thành với emoji; emoji riêng lẻ không đủ làm dấu hiệu mỉa mai. |
| SVM | `test:21` | 1 → 0 | Khen bề mặt | Lời khen tài xế “never makes dirty moves” mang ý mỉa mai nhưng từ ngữ bề mặt tích cực. |
| SVM | `test:47` | 1 → 0 | Khen bề mặt | Khen chính trị gia “can't stop telling the truth” là phê bình mỉa mai; đặc trưng từ vựng tích cực gây nhầm. |
| Zero-shot `explicit_negation` | `test:1` | 0 → 1 | Thiếu ngữ cảnh | Mẫu chỉ có từ “Pinball!” nên thiếu ngữ cảnh hiển thị. |
| Zero-shot `explicit_negation` | `test:2` | 1 → 0 | Ngữ cảnh/đối lập | Mô hình đọc phát biểu về website theo nghĩa đen. |
| Zero-shot `explicit_negation` | `test:20` | 0 → 1 | Emoji/câu ngắn | Lời chúc ngủ ngon thân thiện kèm emoji bị dự đoán là mỉa mai. |
| Zero-shot `explicit_negation` | `test:21` | 1 → 0 | Khen bề mặt | Lời khen thể thao cường điệu được đọc là chân thành khi thiếu bối cảnh ngoài câu. |
| Zero-shot `explicit_negation` | `test:84` | 1 → 0 | Ngữ cảnh/đối lập | Khen việc cho trẻ ăn kẹo trước giờ ngủ dù hệ quả được kỳ vọng trái ngược. |

## Hạn chế và phần nhóm đang chờ

- Split validation seed 42 là tạm. Một văn bản trùng qua fit/validation (`train:3067`, `train:3068`, cùng nhãn 0); train có 20 dòng trong nhóm text lặp và 5 nhóm có nhãn xung đột. Không có text trùng giữa official train/test. Không tự ý xóa mẫu; chi tiết ở [audit dữ liệu](../reports/data_audit.json).
- Cần chạy lại phần Phúc trên split chung sau khi Huy bàn giao loader/split.
- Các lượt fine-tune BERT-base, RoBERTa-base, DistilBERT đủ ba seed, metric chung, demo thật, báo cáo/slide tích hợp toàn nhóm chưa được bàn giao nên chưa chạy hoặc so sánh.

## Nội dung slide phần Phúc

1. **Dữ liệu và baseline:** Subtask A, giao diện `sample_id/text/label`, TF-IDF + LinearSVC; nêu split validation tạm và điểm baseline 0.5525/0.2772 (validation), 0.5708/0.2826 (test), theo thứ tự Macro-F1/F1 sarcastic.
2. **Zero-shot và lỗi:** 5 prompt; chọn theo Macro-F1 validation; trên test prompt gốc 0.4103/0.1083, prompt chọn 0.4476/0.0428; lưu ý Macro-F1 tăng còn F1 sarcastic giảm. Minh họa câu mỉa mai có bề mặt tích cực và false positive có emoji.

## Tệp chi tiết và tái lập

- [Trang kết quả và các run đã đóng gói](../results/phuc/README.md)
- [Bảng tổng hợp CSV](../results/phuc/summary.csv)
- [So sánh prompt validation](../results/phuc/prompt_comparison.csv)
- [Các lỗi ví dụ](../results/phuc/errors.csv)
- [Hướng dẫn lệnh chạy lại](cach_lam.md)
- [Slide phần Phúc](slide_phuc.md)
