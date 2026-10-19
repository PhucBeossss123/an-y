# Phân tích lỗi chung trên official test

Đối chiếu dùng checkpoint có Macro-F1 validation cao nhất trong từng kiến trúc: BERT seed 42, RoBERTa seed 42, DistilBERT seed 123. Chỉ ID và nhãn được ghi vào tài liệu; tweet được đọc từ dữ liệu local để phân tích, không sao chép vào Git. Đây là nhận xét định tính, không phải taxonomy gán nhãn độc lập.

| Nhóm lỗi | ID / nhãn thật | BERT | RoBERTa | DistilBERT | Nhận xét |
|---|---|---:|---:|---:|---|
| Mỉa mai qua tương phản / khen ngược | `test:2` (1) | 1 | 1 | 0 | Khiếu nại về trang đăng ký tiêm chủng được diễn đạt như hệ quả trớ trêu; DistilBERT bỏ lỡ tín hiệu tương phản. |
| Mỉa mai qua tương phản / khen ngược | `test:9` (1) | 0 | 0 | 0 | Kế hoạch tự cải thiện bị đảo ngược bằng câu kết tỉnh dậy; cả ba mô hình bỏ lỡ cấu trúc đối lập. |
| Mỉa mai qua khen ngược | `test:21` (1) | 0 | 0 | 0 | Lời khen “lái xe sạch” đối lập với hành vi đua xe được nhắc tới; cả ba mô hình đọc theo nghĩa đen. |
| Từ ngữ cảm xúc/emoji lấn át lập trường | `test:10` (0) | 1 | 1 | 0 | Lời khen người làm PR và chuỗi emoji tích cực gây false positive ở BERT/RoBERTa. |
| Từ ngữ cảm xúc/emoji lấn át lập trường | `test:11` (0) | 1 | 1 | 1 | Chỉ trích truyền thông trực tiếp bị xem là mỉa mai do từ vựng tiêu cực mạnh. |
| Từ ngữ cảm xúc/emoji lấn át lập trường | `test:14` (0) | 1 | 0 | 0 | Lời nhờ chia sẻ cách xem F1TV và emoji cầu nguyện gây false positive riêng ở BERT. |
| Từ ngữ cảm xúc/emoji lấn át lập trường | `test:28` (0) | 1 | 1 | 1 | Câu nói vui về thời tiết lạnh kết thúc bằng yêu thích thời tiết; cả ba nhầm giọng vui với mỉa mai. |
| Ngữ cảnh/chủ đề làm lệch dự đoán | `test:3` (0) | 1 | 1 | 0 | Cụm “villainous pro tip” mang sắc thái đùa, nhưng nhãn gold là sincere/non-sarcastic. |
| Ngữ cảnh/chủ đề làm lệch dự đoán | `test:6` (0) | 0 | 1 | 1 | Nhận xét về nhạc Giáng Sinh có ngập ngừng và đánh giá tiêu cực; RoBERTa/DistilBERT nhầm thành sarcasm. |
| Ngữ cảnh/chủ đề làm lệch dự đoán | `test:7` (0) | 1 | 0 | 0 | Câu hỏi về chỉ số xG và phạt đền phụ thuộc ngữ cảnh bóng đá; BERT dự đoán sarcasm. |
| Ngữ cảnh/chủ đề làm lệch dự đoán | `test:13` (0) | 1 | 1 | 1 | Câu về Chelsea/West Ham có thể bị hiểu là khen ngược; cả ba dự đoán sarcasm dù nhãn gold là sincere. |
| Ngữ cảnh/chủ đề làm lệch dự đoán | `test:15` (0) | 1 | 1 | 1 | Khiếu nại bị từ chối thuốc đau đầu và nhấn mạnh tuổi “sixteen” bị cả ba gán sarcasm. |
| Phủ định trong câu mỉa mai | `test:127` (1) | 1 | 1 | 0 | Lời phản đối quyết định tiêm nhắc lại kết thúc bằng “I think not”; DistilBERT bỏ lỡ dấu hiệu phủ định. |
| Phủ định và khẳng định trái nghĩa | `test:176` (1) | 0 | 0 | 0 | Câu hỏi tu từ về năng lực chính quyền dùng phủ định rồi khẳng định mọi thứ “great”; cả ba mô hình đọc quá sát nghĩa bề mặt. |
| Phủ định mỉa mai nhấn mạnh | `test:246` (1) | 1 | 0 | 0 | Mẫu dùng “don't want” và “not at all” để thể hiện mong muốn trái với lời khen bề mặt; RoBERTa/DistilBERT bỏ lỡ ý mỉa mai. |
| Cấu trúc “not like ... or anything” | `test:281` (1) | 1 | 0 | 0 | Câu phủ định giả định mang ý chê bai nhân vật; RoBERTa/DistilBERT đọc như lời đồng tình. |

## Ví dụ lỗi theo kiến trúc

Nhãn trong ngoặc là `y_true → y_pred`; `0 = sincere`, `1 = sarcastic`.

- **BERT seed 42:** `test:3` (0→1, đùa có từ ngữ mỉa mai bề mặt); `test:7` (0→1, tham chiếu bóng đá); `test:9` (1→0, tương phản giữa dự định và hành động); `test:10` (0→1, lời khen/emoji); `test:11` (0→1, phê bình trực tiếp); `test:13` (0→1, ngữ cảnh thể thao); `test:14` (0→1, lời nhờ lịch sự kèm emoji); `test:15` (0→1, lời phàn nàn được nhấn mạnh).
- **RoBERTa seed 42:** `test:3` (0→1, từ ngữ đùa); `test:6` (0→1, đánh giá âm tính trực tiếp); `test:9` (1→0, tương phản); `test:10` (0→1, emoji/lời khen); `test:11` (0→1, phê bình trực tiếp); `test:13` (0→1, ngữ cảnh thể thao); `test:15` (0→1, phàn nàn); `test:21` (1→0, khen ngược về tay đua); `test:28` (0→1, giọng vui về thời tiết).
- **DistilBERT seed 123:** `test:2` (1→0, hệ quả trớ trêu); `test:6` (0→1, đánh giá âm tính trực tiếp); `test:9` (1→0, tương phản); `test:11` (0→1, phê bình trực tiếp); `test:13` (0→1, ngữ cảnh thể thao); `test:15` (0→1, phàn nàn); `test:21` (1→0, khen ngược); `test:28` (0→1, giọng vui về thời tiết).

Các ví dụ đều có trong file prediction ID-only dưới `results/team/<model>/seed_<seed>/official_test_predictions.csv`; không có text test được đưa vào kết quả đã commit.
