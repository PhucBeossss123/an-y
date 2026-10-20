# Hướng dẫn đọc và trình bày các mô hình

## Bài toán

Chương trình nhận một câu tiếng Anh và chọn một trong hai nhãn:

- `0`: sincere — chân thành/không mỉa mai.
- `1`: sarcastic — mỉa mai.

Khi huấn luyện, mô hình học từ câu và nhãn đã có. Khi thử câu mới, mô hình chỉ nhận nội dung câu rồi trả về nhãn dự đoán. Mô hình không nghe được giọng nói và không tự biết hoàn cảnh ngoài phần văn bản đã nhập.

## Mở file nào?

| Mô hình | File chính | Nói đơn giản cách hoạt động |
|---|---|---|
| TF-IDF + SVM | `src/an_y/baseline.py` | TF-IDF đổi từ/cặp từ thành các con số; SVM dùng các con số đó để chọn nhãn. Cách này khó nhận ra ý nghĩa phụ thuộc ngữ cảnh. |
| Zero-shot | `src/an_y/zero_shot.py` và `config/prompts.json` | Không huấn luyện lại trên nhãn của bài này. Mô hình NLI so câu với các mô tả như “người viết đang mỉa mai” hoặc “người viết chân thành”; prompt có thể làm kết quả thay đổi. |
| BERT-base | `src/an_y/bert.py` | BERT đã học ngôn ngữ trước đó, rồi được huấn luyện thêm bằng câu và nhãn của bài toán. Nó xét các từ trong ngữ cảnh của cả câu. |
| RoBERTa-base | `src/an_y/roberta.py` | Cùng hướng Transformer với BERT nhưng dùng mô hình nền và cách học khác; cũng được huấn luyện thêm trên dữ liệu bài toán. |
| DistilBERT | `src/an_y/distilbert.py` | Bản Transformer nhỏ hơn, được tinh chỉnh cho cùng bài toán. |

Trong từng file BERT/RoBERTa/DistilBERT, tìm lần lượt `Config`, `EncodedTexts`, `_train_seed`, `run_<tên_mô_hình>`, và `predict_<tên_mô_hình>`. Chúng tương ứng với cấu hình, chuyển câu thành đầu vào, huấn luyện một seed, chạy đủ thí nghiệm, và dự đoán câu mới. Các file dùng chung `src/an_y/data.py` để đọc đúng split và `src/an_y/metrics.py` để tính chỉ số.

## Vì sao kết quả khác nhau?

Mỗi mô hình biểu diễn câu và học ranh giới giữa hai nhãn theo cách khác nhau. SVM chủ yếu dựa vào từ/cặp từ; zero-shot phụ thuộc vào prompt; BERT, RoBERTa và DistilBERT học cách dùng ngữ cảnh từ dữ liệu đã gán nhãn. Mỉa mai còn phụ thuộc vào tình huống và giọng điệu, nên chỉ nhìn chữ đôi khi vẫn chưa đủ. Vì vậy, cùng một câu có thể cho nhãn khác nhau.

## Kết quả dự án

RoBERTa-base được chọn trong ba encoder vì có **Macro-F1 validation trung bình cao nhất: 0.6151** qua ba seed. Lựa chọn được chốt trên validation trước khi báo cáo official test. Macro-F1 cân bằng hai lớp; F1 sarcastic chỉ đo khả năng nhận ra lớp mỉa mai. Xem bảng và kết quả từng mô hình tại `results/team/README.md`.

Điểm SVM và điểm NLI zero-shot hiển thị trong demo là điểm thô, không phải xác suất đã hiệu chỉnh. Câu tự nhập chỉ dùng để minh họa; dự đoán của một câu không thay thế kết quả đánh giá trên validation/official test.

## Bản trình bày ngắn

> “Bọn em phân loại câu tiếng Anh thành chân thành hoặc mỉa mai. TF-IDF + SVM dựa nhiều vào từ và cặp từ; zero-shot so câu với các mô tả prompt; còn BERT, RoBERTa và DistilBERT được tinh chỉnh trên dữ liệu có nhãn. Các mô hình có cách biểu diễn câu khác nhau nên có thể dự đoán khác nhau. Bọn em chọn RoBERTa theo Macro-F1 trên validation, không chọn theo official test.”
