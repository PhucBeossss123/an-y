# Xem báo cáo bằng Streamlit

## Chạy trên Windows

Ứng dụng chạy trên máy cá nhân; địa chỉ `localhost:8502` chỉ hoạt động khi máy đang chạy Streamlit. Tắt máy sẽ dừng tiến trình. Sau khi đăng nhập Windows, mở ứng dụng theo một trong hai cách:

- Chạy `scripts/start_streamlit_hidden.vbs` từ File Explorer. Script tự dùng Python trong `.venv`, chờ dịch vụ sẵn sàng rồi mở trình duyệt.
- Hoặc chạy thủ công từ thư mục gốc repo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Nếu đây là lần đầu thiết lập môi trường, cài gói một lần:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

### Tự khởi động sau khi đăng nhập

Shortcut `iSarcasmEval Streamlit` đã được thêm vào Startup của tài khoản Windows hiện tại. Sau mỗi lần đăng nhập, Windows chạy ứng dụng nền và mở `http://localhost:8502` khi máy chủ sẵn sàng. Tắt máy sẽ dừng ứng dụng; nó được khởi động lại ở lần đăng nhập kế tiếp. Để tắt tự khởi động, nhấn `Win+R`, nhập `shell:startup`, nhấn `Enter`, rồi xóa shortcut đó. Nếu chuyển repo sang thư mục khác, cần tạo lại shortcut trỏ tới `scripts\start_streamlit_hidden.vbs` mới.

Script không cài lại gói khi khởi động. Nó phát hiện nếu dịch vụ đã chạy, ghi log tại `outputs/streamlit/stdout.log` và `outputs/streamlit/stderr.log`, và chờ tối đa 90 giây để ứng dụng sẵn sàng. Nếu trình duyệt không tự mở, chạy lại VBS; nếu vẫn lỗi, xem hai tệp log. Cổng 8502 giữ nguyên để không chiếm cổng 8501 có thể đang được SEIBEN sử dụng.

Trên cùng máy, mở `http://localhost:8502`. Để máy khác trong cùng Wi-Fi/LAN truy cập, lấy IPv4 của máy chạy app bằng `ipconfig` (ví dụ `192.168.1.116`), rồi mở `http://192.168.1.116:8502` trên máy kia. Nếu Windows Firewall hỏi, cho phép Python/Streamlit trên mạng riêng. Cách này chỉ dùng được trong cùng mạng nội bộ, không tạo link Internet công khai.

## Cách đọc báo cáo

- **Báo cáo:** kết luận chính, bốn chỉ số của mô hình được chọn, và bảng so sánh gọn trên official test. Đây là trang trình bày kết quả chính.
- **So sánh mô hình:** xem riêng validation và official test cho BERT, RoBERTa, DistilBERT. Mỗi điểm encoder là trung bình ± độ lệch chuẩn mẫu của ba seed.
- **Prompt zero-shot:** so sánh năm prompt trên validation, sau đó nêu riêng kết quả test của prompt đã chọn và prompt gốc.
- **Phân tích lỗi:** một số ví dụ đại diện từ official test, không liệt kê toàn bộ dự đoán.
- **Thử câu:** chạy mô hình với câu tiếng Anh tự nhập. Câu tự nhập không thuộc tập đánh giá.

## Phạm vi và diễn giải

Các trang báo cáo đọc kết quả đã lưu, không huấn luyện lại. Cấu hình được chọn từ validation trước khi báo cáo official test. Macro-F1 và F1 sarcastic được trình bày riêng. SVM decision margin và zero-shot NLI score là điểm thô, không phải xác suất đã hiệu chỉnh. Dữ liệu nguồn và checkpoint cần có cục bộ; chúng không được đưa vào Git.
