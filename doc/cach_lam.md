# Cách làm và bàn giao phần Phúc

## Phạm vi / giao diện

Phần này chỉ gồm TF-IDF + Linear SVM và zero-shot NLI cho iSarcasmEval tiếng Anh, Subtask A. Đầu vào adapter là `sample_id`, `text`, `label`; `0` là sincere/không mỉa mai, `1` là sarcastic/mỉa mai. Train lấy text từ cột `tweet`; không dùng `rephrase` hoặc các cột chú giải làm đặc trưng. Dự đoán lưu `sample_id`, `y_true`, `y_pred`; score SVM không phải xác suất. Score zero-shot là điểm NLI chuẩn hóa giữa hai nhãn, không được xem là xác suất đã hiệu chỉnh.

## Dữ liệu và giả định hiện tại

Nguồn chính thức: `https://github.com/iabufarha/iSarcasmEval`; tải Subtask A tiếng Anh gồm `train/train.En.csv` và `test/task_A_En_test.csv`. Script ghi revision nguồn, SHA-256, README và LICENSE vào `data/raw/manifest.json`. Dữ liệu được tải cục bộ, không đưa vào Git.

**Split chung đã có:** Huy đã đưa loader và split ID-only vào `main`. Kết quả SVM/zero-shot hiện ghi trong tài liệu và `results/phuc/` vẫn dùng split phát triển tạm 80/20 seed 42; cần chạy lại các thí nghiệm Phúc theo split chung trước khi so sánh mô hình hoặc chốt kết luận. Official test vẫn được giữ riêng.

## Quy trình

1. Tạo `.venv`, cài `requirements.txt`, tải dữ liệu bằng `scripts/download_data.py`.
2. Baseline thử 6 cấu hình nhỏ (`C` 0.5/1/2, class weight none/balanced); chọn Macro-F1 validation, phân xử bằng F1 sarcastic. TF-IDF được fit trong pipeline trên train của split chung. Sau khi chọn, test script refit trên toàn bộ official train và đánh giá official test.
3. Zero-shot dùng `cross-encoder/nli-deberta-v3-small` (revision `fa2804872c3b4bd748f38c0185cc85775361e735`), nhãn gốc `sarcastic`/`sincere` và bốn biến thể đã lưu trong `config/prompts.json`. Model card xác nhận zero-shot Transformers pipeline, huấn luyện trên SNLI/MultiNLI và Apache-2.0. Cùng checkpoint/split/batch cho tất cả. Chọn prompt theo Macro-F1 validation, dùng F1 sarcastic khi hòa. Test chạy sau lựa chọn và so sánh prompt gốc với prompt được chọn trên cùng official test.
4. Metric tạm theo scikit-learn, chờ thay/đối chiếu với evaluator của Tuấn. Bảng validation và bảng test tách riêng.
5. Prediction lỗi được lưu theo ID; cần Phúc đọc từng lỗi, thêm giải thích có căn cứ và xem các nhóm phủ định, emoji, ngữ cảnh trước khi đưa nhận xét vào báo cáo.

## Tình trạng bằng chứng

- Repo `PhucBeossss123/an-y` hiện public, có remote `origin` và nhánh `main`/`phuc/baseline-zero-shot`.
- Python 3.12.7 có tại Miniconda; Python alias mặc định không hoạt động. Máy có RTX 3050 với 4 GB VRAM; `nvidia-smi` xác nhận CUDA runtime 12.7. PyTorch trong `.venv` là bản CPU-only (`2.5.1+cpu`); thử cài wheel CUDA 12.4 chính thức nhưng dừng tải wheel 2.51 GB sau 2.5 phút không có tiến triển hiển thị.
- Repo public đã được xác minh qua GitHub API trước đó. CLI hiện không xác thực được; trạng thái collaborator không được xác nhận trong lần review này.
- Superpowers không có trong tool catalog; đã dùng quy trình tương đương đọc hiểu → lập kế hoạch → thực hiện → kiểm thử → review theo prompt.
- Baseline validation và official test; 5 prompt zero-shot validation và prompt gốc/prompt chọn trên official test đã chạy. Kết quả đọc được trong `results/phuc/README.md`; báo cáo tại `doc/bao_cao_phuc.md`; file chi tiết tại `reports/` và từng run trong `results/phuc/runs/`.
- Review: PR #7 của Huy đã được review, sửa và merge. Không thấy branch/PR của Kiệt hoặc Tuấn trong lần rà remote gần nhất. Kết quả Phúc đã lưu vẫn là kết quả tạm, cần chạy lại trên split chung.
- Kiểm tra sau tích hợp: `python -m compileall -q src scripts` đạt; `scripts/check_line_limits.py` đạt (16 file, tối đa 500 dòng; file dài nhất `src/an_y/bert.py` có 323 dòng); lệnh `--help` của baseline và zero-shot chạy được, có `--validation-split {shared,provisional}`; `git diff --check` sạch. Chưa chạy lại thí nghiệm SVM/zero-shot trên split chung.

## Kết quả thực nghiệm

Baseline tốt nhất validation là `C=0.5`, `class_weight=balanced`; Macro-F1 `0.5525`, F1 sarcastic `0.2772`, confusion matrix `[[464,57],[136,37]]`. Kết quả chỉ dùng split phát triển tạm seed 42, cần chạy lại theo split chung của Huy. Tập train có 3,468 mẫu/867 sarcastic, validation tạm 694 mẫu/173 sarcastic, official test 1,400 mẫu/200 sarcastic. Audit phát hiện 20 dòng train thuộc các nhóm text lặp, trong đó 5 nhóm có nhãn xung đột; ID chi tiết ở `reports/data_audit.json`. Một text trùng qua fit/validation (`train:3067` và `train:3068`, cùng nhãn 0), không có text trùng giữa train/test. Không mẫu nào bị tự động xóa. Test có 6 dòng thuộc các nhóm text trùng nội bộ.

Zero-shot chọn `explicit_negation` bằng validation Macro-F1 `0.5488` (F1 sarcastic `0.2764`); prompt gốc đạt `0.4925` (`0.3149`). Trên official test, prompt gốc: Macro-F1 `0.4103`, F1 sarcastic `0.1083`; prompt chọn: `0.4476`, F1 sarcastic `0.0428`. Prompt chọn cao hơn Macro-F1 nhưng thấp hơn F1 sarcastic; lựa chọn không đổi theo test. Kết quả và ví dụ có trong `doc/results_phuc.md`.

## Cách tham gia repo nhóm

Repo public: <https://github.com/PhucBeossss123/an-y>. Mỗi người dùng tài khoản GitHub riêng. Username do Phúc cung cấp: Huy `qhuyy04`, Tuấn `tuantran11022006-ops`, Kiệt `3ndly`. Trạng thái collaborator/invitation chưa được xác minh.

### Nhận lời mời và clone

1. Phúc mở repo → **Settings** → **Collaborators** → **Add people**, chọn đúng username đã được thành viên xác nhận và gửi lời mời nếu cần quyền push. Không dùng chung tài khoản. Thành viên phải chấp nhận lời mời bằng tài khoản riêng trước khi push. Collaborator của repo cá nhân có quyền ghi; review trước push là quy trình nhóm, không phải khóa kỹ thuật. Trạng thái branch protection chưa xác minh.
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

Không commit `.venv`, `data/raw`, cache hoặc checkpoint. Phần baseline/zero-shot của Phúc có lệnh riêng ở [Chạy lại phần của Phúc](#chay-lai-phan-cua-phuc). Loader/split và pipeline BERT được ghi bên dưới; lượt BERT ba seed đã chạy đầy đủ trên PyTorch CPU. Các model còn lại chưa được chạy trong môi trường này.

### Loader/split chung và BERT-base — nhánh `huy/data-bert`

- `scripts/create_shared_split.py` xác minh SHA-256 của official train/test theo `data/raw/manifest.json`, rồi tạo `data/splits/shared_split_seed42.csv`. File này chỉ có `sample_id` và tên split, không chứa text, rephrase hoặc annotation. Cấu hình hiện tại là một lần tách phân tầng 80/20 từ official train (seed 42); official test giữ nguyên riêng.
- `an_y.data.load_shared_split()` kiểm tra checksum file split, kiểm tra ID đủ/trùng/ngoài nguồn và chỉ trả `sample_id`, `text`, `label`. Nhãn giữ `0 = sincere`, `1 = sarcastic`.
- `scripts/train_bert.py` chạy BERT-base uncased với ba seed `42, 123, 2026`, max length 128, batch train 4, gradient accumulation 4, 3 epoch, AdamW learning rate `2e-5`. Mỗi seed chọn epoch theo validation Macro-F1, dùng F1 sarcastic để phân xử hòa; sau khi chốt checkpoint theo validation mới đánh giá official test. Bảng tóm tắt dùng độ lệch chuẩn mẫu (ddof=1) qua ba seed.
- Checkpoint và metric/prediction sinh ra nằm trong `outputs/bert/` và bị Git ignore. Model Hugging Face `bert-base-uncased` được khóa ở revision `86b5e0934494bd15c9632b12f734a8a67f723594`; có thể thay revision bằng `--revision <commit-sha>`. Run này lưu revision được yêu cầu; thư viện trả `revision_resolved: null`, nên không ghi nhận một resolved SHA riêng trong metrics.
- Hàm `an_y.bert.predict_bert(text, checkpoint_dir)` dự đoán một câu, trả nhãn 0/1 và không gọi đầu ra là xác suất hiệu chỉnh.

Lệnh chạy:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\create_shared_split.py
.\.venv\Scripts\python.exe scripts\train_bert.py --seeds 42 123 2026
```

Tại lần bàn giao này split đã được tạo và kiểm tra: nguồn iSarcasmEval revision `dfc708b53bde1bb571abfb5692f63231c2232195`; SHA-256 train `fde737b69caffbcf81ebe79858ad16ac21881573c308343cd5219c4f7a800ae7`; SHA-256 official test `0ea25619ca3358ecc8440daac1b9bba4d18455bf5b76d874d8dd6a959e72fc49`; SHA-256 split `f1a5118715a5387dc45dd86f797f86885ff60b3b308521aa8005da8befe1abaf`. Phân bố fit `2774` (2080 nhãn 0, 694 nhãn 1), validation `694` (521 nhãn 0, 173 nhãn 1), official test `1400`. Audit ghi nhận một text trùng qua fit/validation (`train:3067`, `train:3068`) và 5 nhóm text lặp có nhãn xung đột trong official train; không xóa hoặc đổi nhãn. Máy này có PyTorch `2.5.1+cpu`, không có CUDA trong PyTorch. Lượt pilot seed 13 epoch 1 trước đây là lần chạy chưa hoàn chỉnh. Lần chạy BERT hoàn chỉnh trước đó dùng seed `13, 42, 2026`; kết quả so sánh hiện tại dùng seed 42 và 2026 từ lần chạy đó, cộng với lần chạy bổ sung seed 123:

```powershell
.\.venv\Scripts\python.exe -u scripts\train_bert.py --seeds 123 --output-dir outputs\bert\seed123_20261014
```

Mỗi seed chọn epoch trên validation rồi mới chấm official test. Kết quả dưới đây là Macro-F1 / F1 sarcastic; mỗi dòng test có 1,400 mẫu:

| Seed | Epoch chọn | Validation Macro-F1 | Validation F1 sarcastic | Test Macro-F1 | Test F1 sarcastic |
|---:|---:|---:|---:|---:|---:|
| 42 | 2 | 0.6133 | 0.4362 | 0.5729 | 0.3465 |
| 123 | 3 | 0.5967 | 0.3630 | 0.5916 | 0.3095 |
| 2026 | 3 | 0.5884 | 0.3358 | 0.5963 | 0.3125 |
| Trung bình ± SD mẫu | — | 0.5995 ± 0.0127 | 0.3783 ± 0.0519 | 0.5869 ± 0.0124 | 0.3228 ± 0.0206 |

Metrics và predictions chi tiết của seed 123 được lưu cục bộ, không đưa lên Git; checkpoint cũng không đưa lên Git. Seed 123 chọn epoch theo validation Macro-F1 rồi mới đánh giá official test. Kết quả seed 42 và 2026 lấy từ lần chạy trước. Metrics revision_requested ghi revision model; revision_resolved là null do thư viện không trả SHA resolved trong run.

Bộ seed so sánh chung là `42, 123, 2026`. Seed 123 được chạy bổ sung; seed 42 và 2026 lấy từ lần chạy trước. SD mẫu được tính từ số liệu seed đã làm tròn trong tài liệu, nên các giá trị tổng hợp là xấp xỉ.
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
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs --split validation --validation-split shared
.\.venv\Scripts\python.exe -m an_y.zero_shot validate --data-dir data\raw --output-dir outputs --validation-split shared
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs --split test
.\.venv\Scripts\python.exe -m an_y.zero_shot test --data-dir data\raw --output-dir outputs
.\.venv\Scripts\python.exe scripts\verify_outputs.py
.\.venv\Scripts\python.exe scripts\package_results.py
```

Validation phải chạy trước test: nó chọn baseline và prompt trên split chung, lưu `outputs/svm.joblib` và `outputs/selected_prompt.json`. Test không dùng để sửa lựa chọn. Các lệnh trên chạy lại theo split ID-only chung và official test. Để tái tạo bộ kết quả lịch sử trong `results/phuc/`, truyền `--validation-split provisional`; các kết quả cũ vẫn được đánh dấu là tạm.

Dự đoán một câu sau khi đã chạy baseline validation và tải model zero-shot lần đầu:

```powershell
.\.venv\Scripts\python.exe -m an_y.predict baseline --text "I just love waiting in traffic for hours."
.\.venv\Scripts\python.exe -m an_y.predict zero-shot --text "I just love waiting in traffic for hours." --prompt-id selected
```

Lệnh in JSON ra terminal, không tự ghi prediction vào bảng thực nghiệm. Bằng chứng đã chạy hiện có trong `results/phuc/`; `outputs/` chứa model cục bộ nhưng bị `.gitignore` loại khỏi Git.

## Rà soát code — 13/10/2026

- Phần ứng dụng `src/an_y` có 603 dòng tính cả dòng trống và khoảng 493 dòng lệnh/code; file `threadpoolctl.py` trong `.venv/Lib/site-packages` là thư viện cài ngoài, không phải code dự án nên không chỉnh sửa.
- Tách riêng bước chọn cấu hình SVM trên validation và bước đánh giá test; tách đọc dữ liệu, chọn prompt, chấm prompt và chạy test thành các hàm có tên theo nhiệm vụ. Loader chỉ trả `sample_id`, `text`, `label`, kiểm tra nhãn `0/1`, và không đưa rephrase/chú giải vào đặc trưng.
- Đã chạy `& .\.venv\Scripts\python.exe -m compileall -q src\an_y` (đạt) và `git diff --check` (đạt). Không chạy lại huấn luyện hoặc suy luận model; không tạo kết quả thực nghiệm mới.

## Đồng bộ GitHub hằng ngày lúc 17:00

- Tác vụ Windows `iSarcasmEval - sync Phuc branch` chạy `scripts/sync_phuc_branch.ps1` lúc 17:00 hằng ngày bằng tài khoản Windows đang đăng nhập. Script giới hạn ở nhánh `phuc/baseline-zero-shot` và remote `origin`; nó chỉ đẩy commit đã có, không tự commit file. Nó fast-forward thay đổi remote nếu working tree tracked sạch, và dừng khi nhánh lệch nhau.
- Có thể chạy thủ công từ repo bằng `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync_phuc_branch.ps1`. Máy cần đang bật và đăng nhập lúc chạy; tác vụ đã tạo trên Windows ngày 13/10/2026.

## Giới hạn độ dài file code

- Mỗi file code Python hoặc PowerShell trong `src/`, `scripts/` và `tests/` được giới hạn tối đa **500 dòng**. `scripts/check_line_limits.py` kiểm tra giới hạn này; thư viện cài trong `.venv/` nằm ngoài phạm vi. GitHub Actions chạy kiểm tra này trên mỗi push và pull request.
- Chạy cục bộ bằng `& .\.venv\Scripts\python.exe scripts\check_line_limits.py`. Kết quả hiện tại: đạt; 13 file code, file dài nhất `scripts/package_results.py` có 263 dòng.
