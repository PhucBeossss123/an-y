# Ẩn Ý

Phần việc của Phúc cho đề tài 11: baseline TF-IDF + Linear SVM và zero-shot Subtask A trên iSarcasmEval tiếng Anh.

## Thiết lập trên Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Nếu `py` không có, dùng Python 3.12 đã cài để tạo môi trường ảo. Không cài thư viện vào môi trường hệ thống.

`requirements.txt` cài PyTorch CPU để thiết lập Windows gọn và ổn định. Nếu muốn dùng GPU, cài wheel CUDA 12.4 chính thức theo [hướng dẫn phiên bản PyTorch](https://docs.pytorch.org/get-started/previous-versions/): `python -m pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1`. Wheel này rất lớn; có thể giữ CPU để chạy.

## Tải dữ liệu và chạy

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\audit_data.py
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs
.\.venv\Scripts\python.exe -m an_y.zero_shot validate --data-dir data\raw --output-dir outputs
```

Dữ liệu ở `data/raw/` và model cache không được commit. Script tải đúng hai CSV tiếng Anh cần cho Subtask A, README và LICENSE từ revision GitHub đã resolve; `manifest.json` lưu revision và SHA-256. Zero-shot dùng `cross-encoder/nli-deberta-v3-small`, checkpoint NLI có hướng dẫn pipeline Transformers và giấy phép Apache-2.0. Nếu Huy bàn giao loader/split dùng chung, thay adapter tạm theo `doc/cach_lam.md` và chạy lại validation.

Sau khi chọn prompt chỉ từ validation, chạy test một lần:

```powershell
.\.venv\Scripts\python.exe -m an_y.zero_shot test --data-dir data\raw --output-dir outputs
```

Lệnh test đòi `outputs/selected_prompt.json` do lệnh `validate` tạo. Đánh giá baseline test dùng:

```powershell
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs --split test
```

Cuối cùng, xác minh và đóng gói các prediction/metric nhỏ đã chạy:

```powershell
.\.venv\Scripts\python.exe scripts\verify_outputs.py
.\.venv\Scripts\python.exe scripts\package_results.py
```

`reports/` chứa metric, bảng prompt, prediction không kèm raw text, data manifest và thông báo license. `data/raw/`, model cache và checkpoint không được commit.

## Google Colab (nếu cần GPU)

Đưa source vào Colab rồi cài `requirements-colab.txt`; file này giữ PyTorch có sẵn của runtime để không thay wheel CUDA bằng wheel CPU:

```python
%pip install -r requirements-colab.txt
%pip install -e .
!python scripts/download_data.py
!python scripts/audit_data.py
!python -m an_y.baseline --data-dir data/raw --output-dir outputs
!python -m an_y.zero_shot validate --data-dir data/raw --output-dir outputs
```

Sau khi kiểm tra/chọn prompt từ validation, chạy hai lệnh test ở trên. Không dùng test để đổi prompt.

## Kết quả

Metric, prediction, bảng thử prompt, log và lỗi được ghi trong `outputs/`. Hãy điền các số liệu đã chạy vào `doc/cach_lam.md` và `doc/slide_phuc.md`; không commit dữ liệu, model hoặc cache.
