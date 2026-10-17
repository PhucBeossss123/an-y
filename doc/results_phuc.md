# Kết quả và phân tích lỗi phần Phúc

Trang kết quả có artifact theo từng lượt tại [`results/phuc/README.md`](../results/phuc/README.md); báo cáo bàn giao chuẩn tại [`bao_cao_phuc.md`](bao_cao_phuc.md). Tệp này tổng hợp lượt chạy mới trên shared split seed 42 ngày 17/10/2026. Prediction CSV và JSON metric nằm tại [`results/phuc/shared_seed42_20261017/`](../results/phuc/shared_seed42_20261017/); các run provisional cũ vẫn được giữ trong `results/phuc/runs/`.

## Cấu hình và trạng thái

- Dữ liệu: iSarcasmEval English Subtask A; raw source revision `dfc708b53bde1bb571abfb5692f63231c2232195`.
- Train source: 3,468 (2,601 sincere, 867 sarcastic); shared split seed 42 gồm 2,774 fit / 694 validation. Official test: 1,400 (1,200 sincere, 200 sarcastic).
- Đã chạy lại SVM và cả 5 zero-shot prompt trên shared validation; đã chốt lựa chọn rồi mới đánh giá official test. Kết quả đầy đủ và lệnh chạy được ghi trong `doc/cach_lam.md`.
- Baseline: word TF-IDF unigram/bigram + LinearSVC; grid validation 6 cấu hình; chọn `C=0.5`, `class_weight=balanced` theo Macro-F1 rồi F1 sarcastic. Model được nạp lại, fit toàn bộ official train sau khi chọn, và lưu để dự đoán.
- Zero-shot: `cross-encoder/nli-deberta-v3-small`, revision `fa2804872c3b4bd748f38c0185cc85775361e735`; pipeline `zero-shot-classification`, không fine-tune. Cả 5 prompt cùng model/split/batch; label mapping khai báo tường minh. `multi_label=False`, điểm chuẩn hóa theo hai candidate labels, không phải xác suất hiệu chỉnh.
- Runtime: Python 3.12.7, pandas 2.2.3, scikit-learn 1.5.2, PyTorch 2.5.1+cpu, Transformers 4.46.3; GPU không dùng được do wheel hiện tại CPU-only.
- Dữ liệu nguồn và metric chính thức được mô tả trong [repo iSarcasmEval](https://github.com/iabufarha/iSarcasmEval); license MIT được giữ tại `reports/source_LICENSE.txt`. Nghiên cứu nền tảng: Abu Farha et al. (2022), “SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic,” https://aclanthology.org/2022.semeval-1.111.
- Zero-shot model card: [`cross-encoder/nli-deberta-v3-small`](https://huggingface.co/cross-encoder/nli-deberta-v3-small), Apache-2.0; card documents Transformers zero-shot pipeline and SNLI/MultiNLI training.

## Baseline — chọn trên validation, đánh giá trên official test

| Split | n | Macro-F1 | F1 sarcastic | Precision sarcastic | Recall sarcastic | Confusion matrix (0,1) |
|---|---:|---:|---:|---:|---:|---|
| Shared validation, seed 42 | 694 | 0.5525 | 0.2772 | 0.3936 | 0.2139 | `[[464,57],[136,37]]` |
| Official test | 1,400 | 0.5708 | 0.2826 | 0.2500 | 0.3250 | `[[1005,195],[135,65]]` |

## Zero-shot prompt comparison

### Validation — prompt selection only

| ID | Candidate labels / template | Macro-F1 | F1 sarcastic | Precision/recall sarcastic |
|---|---|---:|---:|---:|
| original | `sarcastic` / `sincere`; “This text is {}.” | 0.4925 | 0.3149 | 0.2554 / 0.4104 |
| statement | `sarcastic statement` / `sincere statement`; “The author intended this as a {}.” | 0.4996 | 0.3618 | 0.2758 / 0.5260 |
| explicit_negation | `sarcastic` / `not sarcastic`; “The author is being {}.” | **0.5488** | 0.2764 | 0.3725 / 0.2197 |
| social_post | `a sarcastic social media post` / `a sincere social media post`; “This is {}.” | 0.5016 | 0.3551 | 0.2744 / 0.5029 |
| ironic_or_literal | `sarcastic or ironic` / `literal and sincere`; “The tone of this text is {}.” | 0.3072 | **0.4010** | 0.2569 / 0.9133 |

`explicit_negation` was selected because it had the highest Macro-F1; the tie-breaker was not needed. `ironic_or_literal` reached the highest sarcastic F1 but much lower Macro-F1 due to recall `0.1228` for the sincere class.

### Official test — no prompt changes after validation

| Prompt | n | Macro-F1 | F1 sarcastic | Precision/recall sarcastic | Confusion matrix (0,1) |
|---|---:|---:|---:|---:|---|
| Original | 1,400 | 0.4103 | 0.1083 | 0.0766 / 0.1850 | `[[754,446],[163,37]]` |
| Selected: explicit_negation | 1,400 | 0.4476 | 0.0428 | 0.0460 / 0.0400 | `[[1034,166],[192,8]]` |

On test the selected prompt improves Macro-F1 by `0.0373`, but sarcastic F1 falls by `0.0656`. This trade-off is reported without changing the validation selection.

## Examples of actual errors

Explanations below are manual readings of the listed text and outputs; they are not a claim that the dataset annotation is infallible.

### TF-IDF + SVM (official test)

| sample_id | True | Pred | Short explanation |
|---|---:|---:|---|
| `test:2` | 1 | 0 | “want people … so badly that the website doesn't even work” uses a causal mismatch to criticize the service; bag-of-words may read it as a complaint rather than sarcasm. |
| `test:4` | 0 | 1 | “I would date any of these men 🥺” is a sincere positive statement; emoji and short form may create a false sarcastic cue. |
| `test:20` | 0 | 1 | “nitey nite … done now with all the bad news” is a sincere sign-off with 🙈 emojis; emoji presence alone is insufficient. |
| `test:21` | 1 | 0 | “such a clean driver, he never makes dirty moves” is sarcastic praise whose intended meaning conflicts with its literal wording. |
| `test:47` | 1 | 0 | Repeated praise that the Prime Minister “can't stop telling the truth” is ironic political criticism; the model follows positive lexical cues. |

### Zero-shot (validation-selected prompt, official test)

| sample_id | True | Pred | Short explanation |
|---|---:|---:|---|
| `test:1` | 0 | 1 | The single word “Pinball!” lacks enough visible context; the model assigns sarcasm where the item is labeled sincere. |
| `test:2` | 1 | 0 | The criticism is expressed as an apparent statement about intent and a broken website; model takes the surface proposition literally. |
| `test:20` | 0 | 1 | Friendly night-time sign-off with 🙈 emojis; emoji does not imply sarcasm. |
| `test:21` | 1 | 0 | Hyperbolic “never makes dirty moves” reads like sincere sports praise without the external context. |
| `test:84` | 1 | 0 | “feeding children sweets before bedtime is an awesome way of getting lots of sleep” contradicts the expected consequence; positive wording wins over the pragmatic mismatch. |

Across these examples, both methods miss sarcasm conveyed by a positive surface form or contextual contradiction. The SVM and zero-shot classifier also produce a false positive on the same sincere emoji sign-off (`test:20`). These examples support checking emoji, negation, and context; they do not establish dataset-wide causal explanations.

## Data and split audit

- No exact text overlap was found between official train and official test.
- Train contains 20 rows belonging to exact-text duplicate groups, including five groups with conflicting labels; official test contains six repeated-text rows. One exact text appears on both sides of the temporary fit/validation split (`train:3067` and `train:3068`, both label 0). There is no exact text overlap between official train and test. These are disclosed and left untouched as required; Phúc's stored runs must be rerun on Huy's fixed split before final cross-model reporting.
