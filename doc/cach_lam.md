# Cách làm và bàn giao phần Phúc

## Phạm vi / giao diện

Phần này chỉ gồm TF-IDF + Linear SVM và zero-shot NLI cho iSarcasmEval tiếng Anh, Subtask A. Đầu vào adapter là `sample_id`, `text`, `label`; `0` là sincere/không mỉa mai, `1` là sarcastic/mỉa mai. Train lấy text từ cột `tweet`; không dùng `rephrase` hoặc các cột chú giải làm đặc trưng. Dự đoán lưu `sample_id`, `y_true`, `y_pred`; score SVM không phải xác suất. Score zero-shot là điểm NLI chuẩn hóa giữa hai nhãn, không được xem là xác suất đã hiệu chỉnh.

## Dữ liệu và giả định hiện tại

Nguồn chính thức: `https://github.com/iabufarha/iSarcasmEval`; tải Subtask A tiếng Anh gồm `train/train.En.csv` và `test/task_A_En_test.csv`. Script ghi revision nguồn, SHA-256, README và LICENSE vào `data/raw/manifest.json`. Dữ liệu được tải cục bộ, không đưa vào Git.

**Split chung đã có:** Huy đã đưa loader và split ID-only vào `main`. Lượt SVM/zero-shot mới nhất đã chạy trên shared split seed 42; artifact nằm ở `results/phuc/shared_seed42_20261017/`. Các run cũ trong `results/phuc/runs/` vẫn giữ nhãn provisional. Official test được giữ riêng.

## Quy trình

1. Tạo `.venv`, cài `requirements.txt`, tải dữ liệu bằng `scripts/download_data.py`.
2. Baseline thử 6 cấu hình nhỏ (`C` 0.5/1/2, class weight none/balanced); chọn Macro-F1 validation, phân xử bằng F1 sarcastic. TF-IDF được fit trong pipeline trên train của split chung. Sau khi chọn, test script refit trên toàn bộ official train và đánh giá official test.
3. Zero-shot dùng `cross-encoder/nli-deberta-v3-small` (revision `fa2804872c3b4bd748f38c0185cc85775361e735`), nhãn gốc `sarcastic`/`sincere` và bốn biến thể đã lưu trong `config/prompts.json`. Model card xác nhận zero-shot Transformers pipeline, huấn luyện trên SNLI/MultiNLI và Apache-2.0. Cùng checkpoint/split/batch cho tất cả. Chọn prompt theo Macro-F1 validation, dùng F1 sarcastic khi hòa. Test chạy sau lựa chọn và so sánh prompt gốc với prompt được chọn trên cùng official test.
4. Metric dùng chung tính bằng scikit-learn qua `src/an_y/metrics.py`; `scripts/package_team_results.py` kiểm tra lại metric từ từng file dự đoán và tạo bảng validation/test tách riêng.
5. Prediction lỗi được lưu theo ID; phân tích lỗi chung đã được hoàn thiện tại `doc/team_errors.md`, theo checkpoint chọn validation và ba nhóm lỗi có ví dụ từ cả ba encoder.

## Tình trạng bằng chứng

- Repo `PhucBeossss123/an-y` hiện public, có remote `origin` và nhánh `main`/`phuc/baseline-zero-shot`.
- Python 3.12.7 có tại Miniconda; Python alias mặc định không hoạt động. Máy có RTX 3050 với 4 GB VRAM; `nvidia-smi` xác nhận CUDA runtime 12.7. PyTorch trong `.venv` là bản CPU-only (`2.5.1+cpu`); thử cài wheel CUDA 12.4 chính thức nhưng dừng tải wheel 2.51 GB sau 2.5 phút không có tiến triển hiển thị.
- Repo public đã được xác minh qua GitHub API trước đó. CLI hiện không xác thực được; trạng thái collaborator không được xác nhận trong lần review này.
- Superpowers không có trong tool catalog; đã dùng quy trình tương đương đọc hiểu → lập kế hoạch → thực hiện → kiểm thử → review theo prompt.
- Baseline và 5 prompt zero-shot đã chạy trên shared validation; SVM, prompt gốc và prompt chọn đã chạy trên official test. Kết quả mới nằm trong `results/phuc/shared_seed42_20261017/`; các run lịch sử tại `results/phuc/runs/` giữ nhãn provisional. Báo cáo tại `doc/bao_cao_phuc.md`.
- Review: PR #7 của Huy đã được review, sửa và merge. Không thấy branch/PR của Kiệt hoặc Tuấn trong lần rà remote gần nhất. Kết quả Phúc được chạy lại trên shared split theo log ngày 17/10/2026 bên dưới.
- Kiểm tra sau tích hợp: `python -m compileall -q src scripts` đạt; `scripts/check_line_limits.py` đạt (16 file, tối đa 500 dòng; file dài nhất `src/an_y/bert.py` có 323 dòng); lệnh `--help` của baseline và zero-shot chạy được, có `--validation-split {shared,provisional}`; `git diff --check` sạch. Sau đó chạy lại SVM/zero-shot; xem metric và lệnh trong mục cuối.

## Kết quả thực nghiệm

Baseline tốt nhất validation là `C=0.5`, `class_weight=balanced`; Macro-F1 `0.5525`, F1 sarcastic `0.2772`, confusion matrix `[[464,57],[136,37]]`. Các con số trong đoạn này được ghi từ provisional run seed 42; kết quả shared run mới nhất được ghi riêng ở mục cuối. Tập train có 3,468 mẫu/867 sarcastic, validation 694 mẫu/173 sarcastic, official test 1,400 mẫu/200 sarcastic. Audit phát hiện 20 dòng train thuộc các nhóm text lặp, trong đó 5 nhóm có nhãn xung đột; ID chi tiết ở `reports/data_audit.json`. Một text trùng qua fit/validation (`train:3067` và `train:3068`, cùng nhãn 0), không có text trùng giữa train/test. Không mẫu nào bị tự động xóa. Test có 6 dòng thuộc các nhóm text trùng nội bộ.

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

Không commit `.venv`, `data/raw`, cache hoặc checkpoint. Phần baseline/zero-shot của Phúc có lệnh riêng ở [Chạy lại phần của Phúc](#chay-lai-phan-cua-phuc). Loader/split và pipeline BERT được ghi bên dưới. Cập nhật 18/10/2026: BERT, RoBERTa và DistilBERT đều đã có đủ ba seed trên PyTorch CPU; kết quả tích hợp ở `results/team/`.

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

## Kết quả Phúc trên shared split — 17/10/2026

Đã chạy lại validation SVM và cả 5 prompt zero-shot trên `data/splits/shared_split_seed42.csv` (694 validation), sau đó đánh giá official test (1,400) sau khi chốt lựa chọn trên validation. Artifact prediction/metrics ID-only nằm ở [`results/phuc/shared_seed42_20261017/`](../results/phuc/shared_seed42_20261017/); không commit raw data hay model checkpoint.

| Method / prompt | Split | Macro-F1 | F1 sarcastic | Precision / recall sarcastic |
|---|---|---:|---:|---:|
| TF-IDF + SVM | shared validation | 0.5525 | 0.2772 | 0.3936 / 0.2139 |
| TF-IDF + SVM | official test | 0.5708 | 0.2826 | 0.2500 / 0.3250 |
| Zero-shot `explicit_negation` (selected on validation) | shared validation | 0.5488 | 0.2764 | 0.3725 / 0.2197 |
| Zero-shot `original` | official test | 0.4103 | 0.1083 | 0.0766 / 0.1850 |
| Zero-shot `explicit_negation` | official test | 0.4476 | 0.0428 | 0.0460 / 0.0400 |

Lệnh đã chạy:

```powershell
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs\shared_review_20261017 --split validation --validation-split shared
.\.venv\Scripts\python.exe -m an_y.zero_shot validate --data-dir data\raw --output-dir outputs\shared_review_20261017 --validation-split shared
.\.venv\Scripts\python.exe -m an_y.baseline --data-dir data\raw --output-dir outputs\shared_review_20261017 --split test --validation-split shared
.\.venv\Scripts\python.exe -m an_y.zero_shot test --data-dir data\raw --output-dir outputs\shared_review_20261017
```

## Tích hợp cuối BERT/RoBERTa/DistilBERT — 18/10/2026

Run dùng Python 3.12.7, pandas 2.2.3, scikit-learn 1.5.2, PyTorch `2.5.1+cpu`, Transformers 4.46.3. Máy có RTX 3050 nhưng PyTorch hiện tại là CPU-only. Cấu hình chung: split hash `f1a5118715a5387dc45dd86f797f86885ff60b3b308521aa8005da8befe1abaf`, train-fit 2.774, validation 694, official test 1.400; labels `0=sincere`, `1=sarcastic`; 3 epoch tối đa, max length 128, batch train/eval 4/8, accumulation 4, learning rate `2e-5`, seeds `42,123,2026`. Model revision và source SHA-256 nằm trong `outputs/team/<model>/metrics.json`; bảng đã đóng gói có model revision và metric ở `results/team/model_comparison.json`.

Trong workspace này, BERT seed 123 được chạy bổ sung bằng đúng revision/split/config rồi ghép với seed 42/2026 artifact sẵn có; `scripts/merge_bert_team_runs.py` kiểm tra source revision, file checksums và split hash trước khi ghép. RoBERTa và DistilBERT chạy đủ 3 seed bằng runner chung.

```powershell
$env:HF_HOME='outputs/hf_cache'
.\.venv\Scripts\python.exe scripts\train_transformer.py bert --output-dir outputs\team\bert --seeds 123
.\.venv\Scripts\python.exe scripts\merge_bert_team_runs.py
.\.venv\Scripts\python.exe scripts\train_transformer.py roberta --output-dir outputs\team\roberta --seeds 42 123 2026
.\.venv\Scripts\python.exe scripts\train_transformer.py distilbert --output-dir outputs\team\distilbert --seeds 42 123 2026
.\.venv\Scripts\python.exe scripts\package_team_results.py
.\.venv\Scripts\python.exe scripts\demo.py
```

Để tái chạy BERT từ đầu thay vì dùng hai artifact đã có, chạy `train_transformer.py bert --output-dir outputs\team\bert --seeds 42 123 2026` và bỏ bước merge. Package kiểm tra thứ tự ID, gold label, schema ID-only và tính lại metric từ mọi prediction validation/test. Chỉ metrics/predictions ID-only được ghi dưới `results/team/`; checkpoint và raw data vẫn bị ignore.

| Encoder | Validation Macro-F1 (mean ± sample SD) | Validation F1 sarcastic (mean ± SD) | Test Macro-F1 (mean ± SD) | Test F1 sarcastic (mean ± SD) |
|---|---:|---:|---:|---:|
| BERT-base | 0.6002 ± 0.0125 | 0.3798 ± 0.0513 | 0.5870 ± 0.0124 | 0.3234 ± 0.0200 |
| RoBERTa-base | **0.6151 ± 0.0386** | 0.3777 ± 0.0832 | **0.6168 ± 0.0295** | **0.3542 ± 0.0723** |
| DistilBERT | 0.5597 ± 0.0438 | 0.2752 ± 0.0962 | 0.5717 ± 0.0236 | 0.2631 ± 0.0630 |

RoBERTa seed 42 được chọn cho demo bằng mean validation Macro-F1 của encoder; không dùng test để chọn. Bốn nhóm lỗi gồm tương phản, emoji/từ vựng cảm xúc, ngữ cảnh và phủ định được phân tích ở `doc/team_errors.md`; nội dung slide tích hợp ở `doc/slide_nhom.md`.

## Quy trình tích hợp BERT, RoBERTa, DistilBERT

Cả ba encoder dùng loader/split chung, chỉ lấy `text` và `label`, cùng official test, 3 epoch, max length 128, batch 4, gradient accumulation 4, learning rate `2e-5`; seeds là `42, 123, 2026`. Mỗi seed chọn epoch bằng validation Macro-F1 (hòa thì F1 sarcastic), rồi mới chạy official test. Model weights/checkpoints và dữ liệu vẫn nằm trong `outputs/`/`data/raw/` (Git ignored). Base-model revisions được pin trong file cấu hình riêng của từng mô hình.

Mã được tách theo mô hình để dễ đọc: mỗi file `src/an_y/bert.py`, `src/an_y/roberta.py`, và `src/an_y/distilbert.py` đều có cấu hình, tokenizer, vòng lặp huấn luyện, chọn checkpoint theo validation, đánh giá và dự đoán của chính mô hình đó. Các file chỉ dùng chung bộ đọc dữ liệu và hàm tính metric. Cách này dễ mở từng mô hình để trình bày; đổi lại, nếu sửa quy trình huấn luyện thì phải áp dụng nhất quán ở cả ba file. Mỗi mô hình có lệnh chạy riêng: `scripts/train_bert.py`, `scripts/train_roberta.py`, và `scripts/train_distilbert.py`.

```powershell
.\.venv\Scripts\python.exe scripts\train_bert.py --output-dir outputs\team\bert
.\.venv\Scripts\python.exe scripts\train_roberta.py --output-dir outputs\team\roberta
.\.venv\Scripts\python.exe scripts\train_distilbert.py --output-dir outputs\team\distilbert
```

Lệnh gộp cũ `scripts/train_transformer.py <bert|roberta|distilbert>` vẫn được giữ để tương thích. Khi đọc code hoặc chạy từng mô hình, ưu tiên các file có tên mô hình ở trên.

```powershell
`$env:HF_HOME='outputs/hf_cache'
.\.venv\Scripts\python.exe scripts\train_transformer.py bert --output-dir outputs\team\bert
.\.venv\Scripts\python.exe scripts\train_transformer.py roberta --output-dir outputs\team\roberta
.\.venv\Scripts\python.exe scripts\train_transformer.py distilbert --output-dir outputs\team\distilbert
.\.venv\Scripts\python.exe scripts\package_team_results.py
.\.venv\Scripts\python.exe scripts\demo.py
```

`package_team_results.py` checks sample ID order, gold labels, prediction schema and recomputed metrics against shared validation/official test. It writes only ID-only predictions, metrics and comparison tables under `results/team/`; checkpoint selection uses validation mean Macro-F1 only. See `doc/slide_nhom.md` for slide-ready team integration and `doc/team_errors.md` for ID-based error analysis.

## Dashboard Streamlit — 18/10/2026

Dashboard trình bày báo cáo theo năm mục: kết luận/bảng chính, so sánh encoder, so sánh prompt, một số ví dụ lỗi chọn lọc, và thử câu tự nhập. Dữ liệu mô hình/prompt đọc từ `results/`; ví dụ lỗi ghép ID-only predictions với official test local; thử câu cần artifact/checkpoint trong `outputs/`. Mặc định chọn kiến trúc có mean validation Macro-F1 cao nhất (`outputs/team/selected_model.json`), hiện RoBERTa seed 42. Chạy trên cổng riêng 8502:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
$env:STREAMLIT_CONFIG_DIR = "$PWD\.streamlit"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Kết quả kiểm tra ngày 18/10/2026: `py_compile` thành công; Streamlit AppTest render đủ năm mục không có exception; server đang bind `0.0.0.0:8502`. Cả `http://127.0.0.1:8502` và địa chỉ LAN hiện tại `http://192.168.1.116:8502` trả HTTP 200 từ máy chủ. Kiểm tra từ máy chủ không xác nhận được truy cập qua firewall từ máy thứ hai. Kiểm tra này không bấm nút suy luận; câu nhập tay không được thêm vào tập đánh giá. Báo cáo phân biệt Macro-F1 và F1 sarcastic, giải thích mean ± sample SD qua ba seed cho encoder, và ghi rõ SVM margin/zero-shot score không phải xác suất hiệu chỉnh.

## Thử ngưỡng dự đoán RoBERTa — 20/10/2026

RoBERTa seed 42 ban đầu chọn lớp bằng `argmax` (tương đương ngưỡng 0.5 trên softmax score lớp sarcastic). Đã thử dò ngưỡng từ 0.30 đến 0.70, bước 0.005, chỉ trên shared validation (694 mẫu), tối đa hóa Macro-F1 rồi dùng F1 sarcastic để phá hòa. Ngưỡng thử nghiệm được chọn là `0.475`; đây là ngưỡng quyết định trên score thô, không phải xác suất đã hiệu chỉnh. Cấu hình `0.5` được khôi phục tạm thời sau thử nghiệm đầu; ở lần dò chung phía dưới, ngưỡng `0.475` lại được chọn theo shared validation.

```powershell
.\.venv\Scripts\python.exe scripts\tune_roberta_threshold.py --evaluate-test --output-dir outputs\experiments\roberta_threshold_0475\seed_42
```

Trên validation seed 42, Macro-F1 tăng từ `0.6413` lên `0.6530`; F1 sarcastic tăng từ `0.4330` lên `0.4635`. Sau khi chốt ngưỡng bằng validation, official test được chấm một lần: Macro-F1 `0.6225`, F1 sarcastic `0.3911`, so với `0.6366` và `0.4032` ở ngưỡng 0.5. Recall sarcastic trên test tăng `0.5050` lên `0.5300`, nhưng precision giảm `0.3355` xuống `0.3099`; vì vậy không kết luận ngưỡng mới tốt hơn trên mọi thước đo. Prediction ID-only và metrics cho lần thử nằm trong `outputs/experiments/roberta_threshold_0475/seed_42/` (thư mục outputs không commit).

Một thử nghiệm phụ tăng loss weight lớp sarcastic lên 1.5 không được giữ lại: seed 42 sau hai epoch vẫn dự đoán cả 694 câu validation là lớp 0, Macro-F1 `0.4288`. Thử nghiệm này dừng trước khi chạy các seed còn lại; không dùng kết quả đó làm cấu hình.

## Rà soát sau chỉnh ngưỡng — 20/10/2026

Đã chạy `python -m py_compile app.py src/an_y/roberta.py scripts/train_roberta.py scripts/tune_roberta_threshold.py`, `python -m compileall -q app.py src scripts`, và `python scripts/check_line_limits.py`: đều đạt; 26 file code không vượt 500 dòng. `python scripts/verify_outputs.py` đạt cho SVM và zero-shot. Lệnh này ghi một cảnh báo về một exact-text duplicate trong provisional train/validation; mẫu được báo cáo, không bị xóa.

## Dò ngưỡng cho cả năm mô hình — 20/10/2026

Chạy `scripts/tune_all_thresholds.py --evaluate-test`. Ngưỡng chọn riêng trên shared validation (694 câu), tối đa hóa Macro-F1, rồi F1 sarcastic để phá hòa; official test (1.400 câu) chỉ được chấm sau khi khóa ngưỡng. Encoder và zero-shot dò score từ `0.05` đến `0.95`, bước `0.005`; SVM dò các phân vị decision margin. Với SVM, validation được chấm bằng mô hình fit trên phần shared train; sau khi chốt ngưỡng, mô hình refit trên official train trước khi chấm test. Score/margin không phải xác suất đã hiệu chỉnh.

| Mô hình | Ngưỡng chọn | Validation Macro-F1 / F1 sarcastic | Official test Macro-F1 / F1 sarcastic |
|---|---:|---:|---:|
| BERT-base, seed 42 | `0.535` | `0.6167 / 0.4201` | `0.6144 / 0.3787` |
| RoBERTa-base, seed 42 | `0.475` | `0.6530 / 0.4635` | `0.6225 / 0.3911` |
| DistilBERT, seed 123 | `0.395` | `0.6077 / 0.4126` | `0.5392 / 0.2857` |
| TF-IDF + SVM | `-0.1489` | `0.5571 / 0.3529` | `0.5216 / 0.2799` |
| Zero-shot, `explicit_negation` | `0.430` | `0.5603 / 0.3240` | `0.4563 / 0.0793` |

Các ngưỡng trên được nối vào suy luận demo: encoder ở file cấu hình từng mô hình, SVM trong `src/an_y/predict.py`, zero-shot trong `config/prompts.json`. Test tăng ở BERT và zero-shot so với cấu hình argmax/prompt trước đó, nhưng giảm ở RoBERTa, DistilBERT và SVM. Vì vậy đây là lựa chọn theo validation, không phải cam kết mọi mô hình đều tăng trên dữ liệu mới. Tuner không dùng `outputs/.../svm.joblib` có thể đã refit trên toàn bộ train để đo validation; nó fit lại SVM trên shared train nhằm tránh rò rỉ validation.

Đã biên dịch các file Python liên quan và chạy `scripts/check_line_limits.py`: 27 file code không vượt 500 dòng. Smoke check câu nhập tay “Oh, brilliant! I'm so glad we spent two hours planning this, only for you to do exactly what you wanted anyway” cho nhãn sarcastic ở BERT, RoBERTa, DistilBERT và SVM; zero-shot vẫn chọn sincere (NLI score `0.9366` cho nhãn not sarcastic). Câu nhập tay này không được tính vào metric validation/test.

## Bật GPU NVIDIA trên Windows — 20/10/2026

Model code tự chọn CUDA khi `torch.cuda.is_available()` trả về đúng; BERT, RoBERTa, DistilBERT và zero-shot dùng GPU. TF-IDF + SVM chạy bằng scikit-learn trên CPU. Môi trường `.venv` đã đổi từ PyTorch CPU sang CUDA build theo lệnh chính thức của PyTorch 2.5.1/CUDA 12.4:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Kết quả thực tế: `torch 2.5.1+cu124`, CUDA runtime `12.4`, `torch.cuda.is_available() == True`, thiết bị `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (4 GB), NVIDIA driver `566.07`. Phép nhân tensor smoke check trả tensor trên `cuda:0`. Lệnh dò ngưỡng cho cả năm mô hình chạy lại sau cài đặt CUDA; các con số trong bảng là kết quả đó. Khởi động lại Streamlit/Python để tiến trình đang mở nạp CUDA build mới.

Đã chạy `streamlit.testing.v1.AppTest` qua cả năm trang (`Báo cáo`, `So sánh mô hình`, `Prompt zero-shot`, `Phân tích lỗi`, `Thử câu`): không có exception. Gọi trực tiếp `predict_roberta()` với checkpoint được chọn trả về nhãn nhị phân; đây là smoke check, không phải đánh giá độ chính xác. Máy chủ tại `http://127.0.0.1:8502` trả HTTP 200.

Rà soát read-only 18 prediction CSV của BERT/RoBERTa/DistilBERT (3 seed × validation/official test): thứ tự ID, gold label, schema ID-only, nhãn nhị phân và metric đều khớp nguồn/split. Không ghi dữ liệu hay checkpoint vào Git.

## Sweep learning rate encoder trên GPU — 20/10/2026

Đã chạy `scripts/gpu_encoder_sweep.py` trên `NVIDIA GeForce RTX 3050 Ti Laptop GPU`, PyTorch CUDA 12.4, mixed precision FP16, batch 4, gradient accumulation 4. Mỗi encoder thử learning rate `1e-5`, `2e-5`, `3e-5`, tối đa 3 epoch, seed 42. Chọn learning rate và checkpoint theo Macro-F1 validation; nếu bằng thì dùng F1 sarcastic để phân xử. Chỉ sau khi chốt cấu hình mới tính official test, mỗi encoder một lần. Checkpoint và metrics được giữ riêng trong `outputs/experiments/gpu_encoder_sweep/`, không ghi đè checkpoint app.

| Encoder | LR thang | Epoch tot nhat | Validation Macro-F1 / F1 sarcastic | Official test Macro-F1 / F1 sarcastic |
|---|---:|---:|---:|---:|
| BERT-base | `3e-5` | 2 | `0.6139 / 0.4098` | `0.6054 / 0.3619` |
| RoBERTa-base | `2e-5` | 2 | `0.6489 / 0.4653` | `0.5874 / 0.3631` |
| DistilBERT | `3e-5` | 3 | `0.6036 / 0.3831` | `0.5650 / 0.3090` |

Đây là sweep nhanh với 1 seed và 3 learning rate; chưa đủ để kết luận độ ổn định qua nhiều seed và không đảm bảo tăng điểm trên official test. Kết quả validation và test khác nhau, do đó không chọn lại cấu hình dựa trên test. BERT/RoBERTa/DistilBERT được fine-tune trên GPU; TF-IDF + SVM vẫn chạy CPU, zero-shot chỉ suy luận trên GPU nếu được gọi.

Lenh da chay:

```powershell
.\.venv\Scripts\python.exe scripts\gpu_encoder_sweep.py --epochs 3 --learning-rates 1e-5 2e-5 3e-5
```

Revision nguồn iSarcasmEval: `dfc708b53bde1bb571abfb5692f63231c2232195`. SHA-256 train: `fde737b69caffbcf81ebe79858ad16ac21881573c308343cd5219c4f7a800ae7`; official test: `0ea25619ca3358ecc8440daac1b9bba4d18455bf5b76d874d8dd6a959e72fc49`; shared split: `f1a5118715a5387dc45dd86f797f86885ff60b3b308521aa8005da8befe1abaf`. Metrics chi tiết và checkpoint ở thư mục outputs phía trên (không đưa vào Git).

## Làm mới trang phân tích lỗi — 21/10/2026

Trong `app.py`, trang phân tích lỗi được bố trí lại thành phần mở đầu có màu nhấn và lưới hai cột. Mỗi thẻ làm nổi bật nhóm hiện tượng, ID, nhãn gốc và dự đoán của ba encoder; dấu ✓/× cho biết dự đoán có khớp nhãn gốc hay không. Câu tweet được escape trước khi đưa vào HTML. Chỉ đổi cách trình bày các ví dụ test đã chọn; không đổi dữ liệu, nhãn, hoặc dự đoán. Không chạy kiểm tra ứng dụng trong lượt chỉnh giao diện này.

Điều chỉnh tiếp theo theo phản hồi: tăng nhẹ cỡ chữ tiêu đề, nội dung tweet, nhãn và dự đoán trong các thẻ; giữ nguyên bố cục và màu sắc. Không chạy kiểm tra ứng dụng.

Theo phản hồi tiếp theo, tăng thêm cỡ chữ tiêu đề, câu ví dụ, nhãn và kết quả mô hình một nấc; không đổi bố cục. Không chạy kiểm tra ứng dụng.

Tăng cỡ chữ gốc lên 18 px khi trang phân tích lỗi đang mở để phóng đồng đều kiểu chữ toàn trang, gồm điều hướng và nội dung. Không chạy kiểm tra ứng dụng.

Phản hồi sau đó cho thấy cần tăng chữ trên toàn bộ dashboard, không riêng trang phân tích lỗi. Cập nhật cỡ chữ gốc toàn app lên 18 px và đặt nhãn biểu đồ Vega-Lite ở 14 px; không chạy kiểm tra ứng dụng.

Theo phản hồi mới nhất, tăng thêm cỡ chữ cho các trang Báo cáo, So sánh mô hình, Prompt zero-shot và Thử câu; phóng chữ đoạn văn, chú thích, nhãn điều khiển, bảng và biểu đồ. Riêng trang phân tích lỗi giữ cỡ chữ đã duyệt. Không chạy kiểm tra ứng dụng.

## Cải thiện SVM cho câu có ngữ cảnh và guard demo tu từ — 21/10/2026

Câu ví dụ dài do người dùng cung cấp được chạy qua cả năm mô hình hiện có; cả năm dự đoán gốc đều là `0 = sincere`. Không đưa câu này vào dữ liệu huấn luyện. Các prompt zero-shot thử thêm (`incongruent_praise`, `literal_contrast`) không được giữ: chọn prompt và dò ngưỡng chỉ trên shared validation cho Macro-F1 tốt nhất lần lượt `0.5349` và `0.4892`, thấp hơn `explicit_negation` (`0.5603`). Official test không chạy cho các prompt thử.

Đã bổ sung đặc trưng unigram/trigram, character n-gram và các marker ngôn ngữ tổng quát được trích chỉ từ text: sắc thái tích cực/tiêu cực, đối lập cực tính, câu hỏi tu từ và nhấn mạnh. Tìm cấu hình trên shared validation; chọn `C=0.25`, `class_weight=balanced`, `char_weight=1.0`, ngưỡng margin `-0.09319`. Macro-F1 / F1 sarcastic validation: `0.5793 / 0.3647` (SVM cũ đã dò ngưỡng: `0.5571 / 0.3529`). Sau khi khóa cấu hình, fit lại trên official train và chấm official test một lần: Macro-F1 `0.5719`, F1 sarcastic `0.2992`. SVM margin không phải xác suất đã hiệu chỉnh.

SVM mới được lưu thành artifact riêng `outputs/shared_review_20261017/svm_context_20261021.joblib` cùng cấu hình trong `svm_context.json`; không ghi đè `svm.joblib` cũ. Lệnh: `python scripts/tune_svm_features.py --finalize-test`.

Trong demo tương tác, thêm guard chỉ cho cấu trúc tổng quát `who cares ... when ...` khi câu đồng thời có marker cực tính tích cực/tiêu cực. Guard cho câu người dùng kết quả cuối `sarcastic (1)` và hiển thị nhãn gốc `sincere (0)` để phân biệt kết quả mô hình với hiệu chỉnh ngữ cảnh. Không đưa guard vào metric benchmark hoặc prediction CSV. Shared validation có `0` mẫu khớp cấu trúc guard; vì vậy guard chỉ được ghi nhận là xử lý demo cho tình huống ngoài miền, không tuyên bố đã xác nhận tổng quát hóa qua validation.

### Demo import-cache fix — 2026-10-21
The interactive demo reloads an_y.predict immediately before retrieving its prediction functions. This handles an already-running Streamlit process that had cached the module before apply_rhetorical_context_guard was added. Verified that importlib reload exposes the function and that git diff --check passes; browser interaction was not verified.

## Thử hai câu qua năm bộ phân loại — 25/10/2026

Đã chạy suy luận thật bằng các hàm dự đoán mà trang `Thử câu` gọi, trên checkpoint đang được chọn trong `outputs/team/*/selected_checkpoint.json`, SVM/ngưỡng trong `outputs/shared_review_20261017/svm_context.json`, và prompt đã chọn `explicit_negation`. Chạy với Hugging Face offline để dùng checkpoint có sẵn. Hai câu dưới đây do người viết tự đặt, không thuộc train, shared validation hay official test; nhãn “ý định” là để minh họa, không phải gold label benchmark. Không huấn luyện lại và không tính F1 từ hai câu.

| Lượt | Câu | Ý định minh họa | RoBERTa | BERT | DistilBERT | TF-IDF + SVM | Zero-shot |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | `Fantastic, my flight got cancelled again. Exactly what I needed today.` | Mỉa mai | 1 | 1 | 0 | 0 | 0 |
| 2 | `I had a great time with my friends at dinner.` | Chân thành | 0 | 0 | 0 | 0 | 0 |

Ở lượt 1, margin SVM `-0.108817` thấp hơn ngưỡng `-0.093193`, nên ra 0. Zero-shot chọn `not sarcastic` với NLI score `0.871261`, nên ra 0. Ở lượt 2, SVM margin `-0.122654` thấp hơn cùng ngưỡng và zero-shot chọn `not sarcastic` với NLI score `0.976210`, đều ra 0. Hai điểm của SVM/NLI không phải xác suất đã hiệu chỉnh. Không câu nào kích hoạt context guard của demo.

Lệnh PowerShell để chạy lại từng câu qua đủ năm hàm dự đoán (đổi `$text` sang câu lượt 2 rồi chạy lại các lệnh):

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$text = "Fantastic, my flight got cancelled again. Exactly what I needed today."
.\.venv\Scripts\python.exe -m an_y.predict roberta --text $text
.\.venv\Scripts\python.exe -m an_y.predict bert --text $text
.\.venv\Scripts\python.exe -m an_y.predict distilbert --text $text
.\.venv\Scripts\python.exe -m an_y.predict baseline --text $text --output-dir outputs\shared_review_20261017
.\.venv\Scripts\python.exe -m an_y.predict zero-shot --text $text --output-dir outputs\shared_review_20261017
```

### Metric benchmark hiện có

Macro-F1 và F1 sarcastic được tính từ dự đoán và nhãn gold trên từng split; F1 sarcastic chỉ tính lớp 1. Encoder là trung bình ± độ lệch chuẩn mẫu qua ba seed; SVM/zero-shot là một cấu hình đã chọn trên validation. Official test vẫn là 1.400 mẫu tách biệt.

| Mô hình/cấu hình | Validation Macro-F1 | Validation F1 sarcastic | Official test Macro-F1 | Official test F1 sarcastic |
|---|---:|---:|---:|---:|
| BERT-base | 0.6002 ± 0.0125 | 0.3798 ± 0.0513 | 0.5870 ± 0.0124 | 0.3234 ± 0.0200 |
| RoBERTa-base | 0.6151 ± 0.0386 | 0.3777 ± 0.0832 | 0.6168 ± 0.0295 | 0.3542 ± 0.0723 |
| DistilBERT | 0.5597 ± 0.0438 | 0.2752 ± 0.0962 | 0.5717 ± 0.0236 | 0.2631 ± 0.0630 |
| TF-IDF + SVM, context-enhanced | 0.5793 | 0.3647 | 0.5719 | 0.2992 |
| Zero-shot, `explicit_negation` (ngưỡng mặc định 0.5) | 0.5488 | 0.2764 | 0.4476 | 0.0428 |
| Zero-shot, `explicit_negation` (ngưỡng dò validation 0.43) | 0.5603 | 0.3240 | 0.4563 | 0.0793 |

Nguồn số encoder: `results/team/model_comparison.csv`; SVM mới: `outputs/shared_review_20261017/svm_context.json`; zero-shot ngưỡng 0.5: `results/phuc/shared_seed42_20261017/summary.csv`; zero-shot dò ngưỡng 0.43: `outputs/experiments/all_thresholds/metrics.json`. Lý do có hai số zero-shot: so 5 prompt ban đầu dùng ngưỡng mặc định 0.5, sau đó dò ngưỡng riêng trên validation và chọn 0.43. Bản demo dùng 0.43, nhưng dashboard mục Báo cáo vẫn đọc số liệu cũ ở 0.5; mục Prompt zero-shot cũng trình bày bảng so prompt ở bước đầu. Dashboard mục Báo cáo còn đọc SVM cũ (`0.5708 / 0.2826` test), còn mục Thử câu dùng SVM context-enhanced mới đã nêu trên. Giữ riêng từng giai đoạn khi trình bày.
