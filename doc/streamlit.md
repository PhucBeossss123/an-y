# Xem báo cáo bằng Streamlit

## Mở ứng dụng trên Windows

Chạy từ thư mục gốc repo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
$env:STREAMLIT_CONFIG_DIR = "$PWD\.streamlit"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Trên cùng máy, mở `http://localhost:8502`. Để máy khác trong cùng Wi-Fi/LAN truy cập, lấy IPv4 của máy chạy app bằng `ipconfig` (ví dụ `192.168.1.116`), rồi mở `http://192.168.1.116:8502` trên máy kia. Nếu Windows Firewall hỏi, cho phép Python/Streamlit trên mạng riêng. Cổng 8502 được dùng để không chiếm cổng 8501 có thể đang được SEIBEN sử dụng. Dừng app bằng `Ctrl+C` trong terminal; `STREAMLIT_CONFIG_DIR` lưu cấu hình trong thư mục dự án. Cách này chỉ dùng được trong cùng mạng nội bộ, không tạo link Internet công khai.

## Cách đọc báo cáo

- **Báo cáo:** kết luận chính, bốn chỉ số của mô hình được chọn, và bảng so sánh gọn trên official test. Đây là trang trình bày kết quả chính.
- **So sánh mô hình:** xem riêng validation và official test cho BERT, RoBERTa, DistilBERT. Mỗi điểm encoder là trung bình ± độ lệch chuẩn mẫu của ba seed.
- **Prompt zero-shot:** so sánh năm prompt trên validation, sau đó nêu riêng kết quả test của prompt đã chọn và prompt gốc.
- **Phân tích lỗi:** một số ví dụ đại diện từ official test, không liệt kê toàn bộ dự đoán.
- **Thử câu:** chạy mô hình với câu tiếng Anh tự nhập. Câu tự nhập không thuộc tập đánh giá.

## Phạm vi và diễn giải

Các trang báo cáo đọc kết quả đã lưu, không huấn luyện lại. Cấu hình được chọn từ validation trước khi báo cáo official test. Macro-F1 và F1 sarcastic được trình bày riêng. SVM decision margin và zero-shot NLI score là điểm thô, không phải xác suất đã hiệu chỉnh. Dữ liệu nguồn và checkpoint cần có cục bộ; chúng không được đưa vào Git.
