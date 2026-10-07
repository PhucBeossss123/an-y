# Cách làm và bàn giao phần Phúc

## Phạm vi / giao diện

Phần này chỉ gồm TF-IDF + Linear SVM và zero-shot NLI cho iSarcasmEval tiếng Anh, Subtask A. Đầu vào adapter là `sample_id`, `text`, `label`; `0` là sincere/không mỉa mai, `1` là sarcastic/mỉa mai. Train lấy text từ cột `tweet`; không dùng `rephrase` hoặc các cột chú giải làm đặc trưng. Dự đoán lưu `sample_id`, `y_true`, `y_pred`; score SVM không phải xác suất. Score zero-shot là điểm NLI chuẩn hóa giữa hai nhãn, không được xem là xác suất đã hiệu chỉnh.

## Dữ liệu và giả định hiện tại

Nguồn chính thức: `https://github.com/iabufarha/iSarcasmEval`; tải Subtask A tiếng Anh gồm `train/train.En.csv` và `test/task_A_En_test.csv`. Script ghi revision nguồn, SHA-256, README và LICENSE vào `data/raw/manifest.json`. Dữ liệu được tải cục bộ, không đưa vào Git.

**Chờ Huy:** loader và split chung. Cho đến khi bàn giao, adapter này chia phân tầng 80/20 trên train với seed 42; đây là split phát triển tạm, không phải split chung đã được nhóm duyệt. Test chính thức được giữ riêng. Khi nhận split đã chốt, thay adapter và chạy lại các thí nghiệm trước khi kết luận.

## Quy trình

1. Tạo `.venv`, cài `requirements.txt`, tải dữ liệu bằng `scripts/download_data.py`.
2. Baseline thử 6 cấu hình nhỏ (`C` 0.5/1/2, class weight none/balanced); chọn Macro-F1 validation, phân xử bằng F1 sarcastic. TF-IDF được fit trong pipeline trên train của split phát triển. Sau khi chọn, test script refit trên toàn bộ official train và đánh giá official test.
3. Zero-shot dùng `cross-encoder/nli-deberta-v3-small` (revision `fa2804872c3b4bd748f38c0185cc85775361e735`), nhãn gốc `sarcastic`/`sincere` và bốn biến thể đã lưu trong `config/prompts.json`. Model card xác nhận zero-shot Transformers pipeline, huấn luyện trên SNLI/MultiNLI và Apache-2.0. Cùng checkpoint/split/batch cho tất cả. Chọn prompt theo Macro-F1 validation, dùng F1 sarcastic khi hòa. Test chạy sau lựa chọn và so sánh prompt gốc với prompt được chọn trên cùng official test.
4. Metric tạm theo scikit-learn, chờ thay/đối chiếu với evaluator của Tuấn. Bảng validation và bảng test tách riêng.
5. Prediction lỗi được lưu theo ID; cần Phúc đọc từng lỗi, thêm giải thích có căn cứ và xem các nhóm phủ định, emoji, ngữ cảnh trước khi đưa nhận xét vào báo cáo.

## Tình trạng bằng chứng

- Workspace ban đầu trống; repo private `PhucBeossss123/an-y` hiện đã tồn tại, có remote `origin` và nhánh `main`/`phuc/baseline-zero-shot`.
- Python 3.12.7 có tại Miniconda; Python alias mặc định không hoạt động. Máy có RTX 3050 với 4 GB VRAM; `nvidia-smi` xác nhận CUDA runtime 12.7. PyTorch trong `.venv` là bản CPU-only (`2.5.1+cpu`); thử cài wheel CUDA 12.4 chính thức nhưng dừng tải wheel 2.51 GB sau 2.5 phút không có tiến triển hiển thị.
- GitHub CLI đã xác minh `PhucBeossss123`; repo private là `https://github.com/PhucBeossss123/an-y`. Commit và SHA nhánh được ghi trong `doc/review.md`/trang kết quả.
- Superpowers không có trong tool catalog; đã dùng quy trình tương đương đọc hiểu → lập kế hoạch → thực hiện → kiểm thử → review theo prompt.
- Baseline validation và official test; 5 prompt zero-shot validation và prompt gốc/prompt chọn trên official test đã chạy. Kết quả đọc được trong `results/phuc/README.md`; báo cáo tại `doc/bao_cao_phuc.md`; file chi tiết tại `reports/` và từng run trong `results/phuc/runs/`.
- Review: hoàn tất review phần Phúc. Username và lời mời Huy/Tuấn/Kiệt đang chờ; chưa có code thành viên nào để review.

## Kết quả thực nghiệm

Baseline tốt nhất validation là `C=0.5`, `class_weight=balanced`; Macro-F1 `0.5525`, F1 sarcastic `0.2772`, confusion matrix `[[464,57],[136,37]]`. Kết quả chỉ dùng split phát triển tạm seed 42, cần chạy lại theo split chung của Huy. Tập train có 3,468 mẫu/867 sarcastic, validation tạm 694 mẫu/173 sarcastic, official test 1,400 mẫu/200 sarcastic. Audit phát hiện 20 dòng train thuộc các nhóm text lặp, trong đó 5 nhóm có nhãn xung đột; ID chi tiết ở `reports/data_audit.json`. Một text trùng qua fit/validation (`train:3067` và `train:3068`, cùng nhãn 0), không có text trùng giữa train/test. Không mẫu nào bị tự động xóa. Test có 6 dòng thuộc các nhóm text trùng nội bộ.

Zero-shot chọn `explicit_negation` bằng validation Macro-F1 `0.5488` (F1 sarcastic `0.2764`); prompt gốc đạt `0.4925` (`0.3149`). Trên official test, prompt gốc: Macro-F1 `0.4103`, F1 sarcastic `0.1083`; prompt chọn: `0.4476`, F1 sarcastic `0.0428`. Prompt chọn cao hơn Macro-F1 nhưng thấp hơn F1 sarcastic; lựa chọn không đổi theo test. Kết quả và ví dụ có trong `doc/results_phuc.md`.

## Cách tham gia repo nhóm

Repo private: <https://github.com/PhucBeossss123/an-y>. Mỗi người dùng tài khoản GitHub riêng. Username của Huy, Tuấn và Kiệt đang chờ Phúc cung cấp để mời; chưa gửi lời mời và chưa xác minh quyền trên máy của các bạn.

### Nhận lời mời và clone

1. Phúc mở repo → **Settings** → **Collaborators** → **Add people**, tìm đúng username do thành viên xác nhận và gửi lời mời. Không dùng chung tài khoản. Thành viên phải chấp nhận lời mời khi đăng nhập đúng tài khoản trước khi clone repo private. Collaborator của repo cá nhân có quyền ghi; review trước push là quy trình nhóm, không phải khóa kỹ thuật. Hiện chưa cấu hình branch protection và không tuyên bố GitHub tự chặn push chưa duyệt.
2. Clone nhánh chung `main`, vào thư mục repo và tạo nhánh theo phân công. Nếu đã clone repo hoặc nhánh tồn tại, kiểm tra trước bằng `git status` và `git branch -a`, không tạo trùng một cách mù quáng.

```powershell
git clone --branch main https://github.com/PhucBeossss123/an-y.git
Set-Location an-y
git switch -c huy/data-bert
```

Tên nhánh: Huy `huy/data-bert`; Tuấn `tuan/roberta-evaluation`; Kiệt `kiet/distilbert-demo`. Chỉ sửa phần sở hữu đã ghi trong `doc/phan_cong.md`. Không commit trực tiếp lên `main` hoặc sửa phần của người khác.

Để nhận thay đổi mới đã được Phúc tích hợp vào `main`, trước hết không được có thay đổi local chưa commit; sau đó cập nhật `main`, rồi đưa phần chung vào nhánh riêng và gửi lại phiên bản đã kiểm tra để review:

```powershell
git status
git fetch origin
git switch main
git pull --ff-only origin main
git switch huy/data-bert
git merge main
```

Thay `huy/data-bert` bằng nhánh của mình. Chạy lại kiểm tra, commit local thay đổi tích hợp và tạo bundle phiên bản mới; không push commit mới chỉ dựa trên lần duyệt trước.

### Tạo môi trường và dữ liệu

Từ thư mục gốc repo, dùng Python 3.12 trên Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\audit_data.py
```

Không commit `.venv`, `data/raw`, cache hoặc checkpoint. Phần baseline/zero-shot của Phúc có lệnh riêng ở [Chạy lại phần của Phúc](#chay-lai-phan-cua-phuc). Loader/split chung và các model còn lại đang chờ bàn giao; không giả định demo hoặc chín lượt fine-tune đã chạy.

### Commit local và gửi bản review

Trước khi code, lưu commit nền đang có:

```powershell
$baseSha = git rev-parse HEAD
```

Chạy kiểm tra phù hợp, xem `git diff`, rồi commit local các file đã hoàn thành. Commit local được phép trước review; **không push**. Ghi lại `$baseSha`, `git rev-parse HEAD` làm `head_sha`, lệnh kiểm tra, kết quả và lỗi còn lại. Tạo bundle ngoài repo với tên phiên bản mới mỗi lần gửi:

```powershell
git bundle create ..\huy-review-v1.bundle huy/data-bert
git bundle verify ..\huy-review-v1.bundle
git rev-parse huy/data-bert
```

Bundle chỉ chứa commit; không bao gồm thay đổi chưa commit, secret, `.venv`, cache, dữ liệu hoặc model lớn. Thành viên tự gửi bundle và thông tin review cho Phúc qua kênh nhóm. Không gửi bundle cũ đè lên bản trước.

### Phúc mở và review bundle

Ví dụ khi `huy-review-v1.bundle` nằm cạnh thư mục clone hiện tại; thay tên người và phiên bản tương ứng:

```powershell
git bundle verify ..\huy-review-v1.bundle
git clone --branch huy/data-bert ..\huy-review-v1.bundle ..\review-huy-v1
Set-Location ..\review-huy-v1
git rev-parse HEAD
git diff <base_sha>..<head_sha> --stat
git diff <base_sha>..<head_sha>
```

Phúc đối chiếu `HEAD` đúng `head_sha`, đọc diff trong thư mục review riêng, cài môi trường/chạy kiểm tra phù hợp và ghi phiên bản, lỗi, mức ảnh hưởng, kết quả vào `doc/review.md`. Thành viên sửa phần mình và gửi bundle phiên bản mới nếu có lỗi.

Chỉ khi Phúc xác nhận rõ `Duyệt push <tên> <head_sha>` mới được push. Sau xác nhận, thành viên đối chiếu chính xác SHA rồi push nhánh riêng:

```powershell
git rev-parse HEAD
git push -u origin huy/data-bert
git ls-remote origin refs/heads/huy/data-bert
```

Nếu SHA đã đổi sau review, kể cả do sửa conflict/rebase, dừng push và gửi review lại. Phúc duyệt push và Phúc duyệt tích hợp vào `main` là hai bước khác nhau; Phúc chủ động review PR trước khi tích hợp. Không tự tạo/merge PR thay Phúc.

<a id="chay-lai-phan-cua-phuc"></a>
## Chạy lại phần của Phúc

Đứng ở thư mục gốc repo và kích hoạt `.venv` hoặc gọi trực tiếp Python trong `.venv`. Dữ liệu tải chưa được commit; tải theo manifest trước khi chạy:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\audit_data.py
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs --split validation
.\.venv\Scripts\python.exe -m an_y.zero_shot validate --data-dir data\raw --output-dir outputs
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs --split test
.\.venv\Scripts\python.exe -m an_y.zero_shot test --data-dir data\raw --output-dir outputs
.\.venv\Scripts\python.exe scripts\verify_outputs.py
.\.venv\Scripts\python.exe scripts\package_results.py
```

Validation phải chạy trước test: nó chọn baseline, lưu `outputs/svm.joblib` và `outputs/selected_prompt.json`. Test không dùng để sửa lựa chọn. Các lệnh trên tái tạo split phát triển tạm seed 42 và official test; **chưa thay thế việc chạy lại trên split chung của Huy**.

Dự đoán một câu sau khi đã chạy baseline validation và tải model zero-shot lần đầu:

```powershell
.\.venv\Scripts\python.exe -m an_y.predict baseline --text "I just love waiting in traffic for hours."
.\.venv\Scripts\python.exe -m an_y.predict zero-shot --text "I just love waiting in traffic for hours." --prompt-id selected
```

Lệnh in JSON ra terminal, không tự ghi prediction vào bảng thực nghiệm. Bằng chứng đã chạy hiện có trong `results/phuc/`; `outputs/` chứa model cục bộ nhưng bị `.gitignore` loại khỏi Git.
