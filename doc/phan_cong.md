# PHÂN CÔNG ĐỀ TÀI 11 — NHÓM 4 NGƯỜI

> **Lưu ý phiên bản:** mục 6 ghi lại yêu cầu ở lượt phân công ban đầu (chỉ chia task, chưa tạo repo/push). Với việc triển khai hiện tại, quyền và quy trình Git được cập nhật bởi Prompt_An_Y mới do Phúc cung cấp; xem `AGENTS.md` và `doc/cach_lam.md`.

Ngày lập: 07/10/2026. Hạn nội bộ hoàn tất: 19/10/2026, trước ngày 20/10/2026 (giờ Việt Nam).

Đề tài: **Phát hiện mỉa mai (Sarcasm) trên mạng xã hội với iSarcasmEval**, phần tiếng Anh.

Căn cứ: yêu cầu chung và mục 11 trong tài liệu giảng viên do người dùng gửi; tên thành viên lấy từ ảnh đính kèm. Bản phân công lấy Huỳnh Trần Nhật Phúc là leader theo ngữ cảnh và dòng được tô trong ảnh. Các mốc tháng 10/2026 là cách hiểu của yêu cầu “trước ngày 20”.

## 1. Phạm vi phải làm đúng

- Dùng iSarcasmEval tiếng Anh, bài toán phân loại nhị phân mỉa mai/không mỉa mai. Không tự đổi dữ liệu, ngôn ngữ hoặc giảm dữ liệu của bài chính.
- Baseline: TF-IDF + SVM.
- Zero-shot: pipeline `zero-shot-classification`, nhãn gốc `sarcastic` / `sincere`, không fine-tune mô hình của nhánh này.
- Fine-tune đủ **BERT-base, RoBERTa-base và DistilBERT**; mỗi người phụ trách trọn vẹn một kiến trúc. Đây là bộ ba theo bảng gợi ý tiếng Anh trong tài liệu. Phần riêng của đề nhắc RoBERTa nhưng yêu cầu chung vẫn bắt buộc đủ ba kiến trúc.
- Mỗi kiến trúc chạy ít nhất ba seed, đề xuất dùng chung `42`, `123`, `2026`: tổng cộng ít nhất chín lượt fine-tune hoàn chỉnh. Báo cáo từng lượt và mean ± std.
- Đánh giá Macro-F1 và F1 riêng lớp sarcastic; bổ sung precision/recall theo lớp và ma trận nhầm lẫn để phân tích.
- Phân tích ít nhất ba nhóm lỗi; phải có phủ định và emoji, nhóm thứ ba có thể là câu phụ thuộc ngữ cảnh hoặc nghĩa bề mặt trái ý định. Đây là các nhóm cần kiểm tra, không phải kết luận có sẵn.
- Nâng cao: prompt engineering cho zero-shot, so sánh trước/sau và đối chiếu với các mô hình fine-tune. Không mặc định prompt mới sẽ tốt hơn; báo cáo kết quả thực tế.
- Có demo Gradio hoặc Streamlit dùng mô hình tốt nhất trong ba kiến trúc, chọn bằng kết quả validation.
- Có hai báo cáo tiến độ, báo cáo cuối, slide và bảng đóng góp theo yêu cầu giảng viên. Các hạn dưới đây là hạn nội bộ, không thay thế lịch nộp chính thức.

## 2. Phân công cân bằng theo công sức dự kiến

Mỗi người khoảng 25% công sức, gồm cả code, thí nghiệm, phân tích và tài liệu. Đây là mục tiêu phân bổ, không phải số đóng góp thực tế đã được xác nhận. Không chia đều bằng số file hoặc số dòng code.

| Thành viên | Phần chính phải sở hữu trọn vẹn | Phần việc chung được giao | Bàn giao cho leader |
|---|---|---|---|
| **Huỳnh Trần Nhật Phúc — leader** | TF-IDF + SVM; zero-shot với hai nhãn sarcastic/sincere; thử 3–5 cách diễn đạt prompt/nhãn và đánh giá phần nâng cao | Chốt giao diện dữ liệu/kết quả, điều phối, review code, duyệt bản đủ điều kiện push; viết nhận xét cuối về baseline/zero-shot và kết quả chung | Code chạy baseline/zero-shot; cấu hình prompt; kết quả trước/sau; phần báo cáo của mình; danh sách lỗi review và trạng thái duyệt từng người |
| **Lê Quốc Huy** | BERT-base: chuẩn bị tokenizer riêng, fine-tune đủ ba seed, đánh giá, lưu mô hình/tokenizer và viết hàm dự đoán | Chuẩn bị dữ liệu dùng chung: nguồn tải, thống kê nhãn, kiểm tra chất lượng dữ liệu, bộ chia train/validation/test cố định và bộ đọc dữ liệu | Bộ dữ liệu/split thống nhất và thống kê trước 08/10; code BERT; ba bộ kết quả/prediction; checkpoint có thể chạy; phần báo cáo dữ liệu và BERT |
| **Trần Hoàng Tuấn** | RoBERTa-base: tokenizer riêng, fine-tune đủ ba seed, đánh giá, lưu mô hình/tokenizer và viết hàm dự đoán | Viết bộ tính chỉ số dùng chung; tổng hợp bảng so sánh, mean ± std; tập hợp ví dụ lỗi từ cả nhóm; ghép các phần báo cáo do từng người viết | Code RoBERTa; ba bộ kết quả/prediction; checkpoint có thể chạy; bộ đánh giá; bảng so sánh chung; phần báo cáo RoBERTa và bản tổng hợp |
| **Nguyễn Trần Anh Kiệt** | DistilBERT: tokenizer riêng, fine-tune đủ ba seed, đánh giá, lưu mô hình/tokenizer và viết hàm dự đoán | Xây demo Gradio/Streamlit đọc mô hình qua giao diện chung; kết nối mô hình thắng validation dù đó là BERT/RoBERTa; soạn hướng dẫn chạy demo, ảnh/video demo và ghép slide | Code DistilBERT; ba bộ kết quả/prediction; checkpoint có thể chạy; demo nhập câu tiếng Anh và xem dự đoán; phần báo cáo DistilBERT/demo và slide tổng hợp |

**Phân bổ tài liệu:** mỗi người tự viết phương pháp, cấu hình, kết quả, hạn chế của phần mình và chuẩn bị khoảng hai slide. Tuấn ghép báo cáo; Kiệt ghép slide; Phúc kiểm tra và duyệt. Huy, Tuấn, Kiệt tự sửa lỗi trong phần mình khi nhận review. Phúc không nhận làm lại toàn bộ code hoặc viết thay các phần còn thiếu.

**Phân tích lỗi:** mỗi chủ mô hình cung cấp ví dụ có `sample_id`, nhãn thật, dự đoán và giải thích ngắn; tối thiểu năm lỗi tiêu biểu nếu có đủ lỗi. Phúc làm tương tự cho baseline/zero-shot. Tuấn ghép và đối chiếu cả ba kiến trúc trên cùng các mẫu, phân thành ít nhất ba nhóm lỗi có ví dụ thực tế.

## 3. Thống nhất để bốn người làm song song

1. Huy chốt dữ liệu và split trước 20:00 ngày 08/10. Giữ tập test chính thức; nếu cần tạo validation từ train, dùng một lần chia phân tầng cố định, công bố seed chia và danh sách ID. Không chia lại tập dữ liệu theo seed huấn luyện của mỗi người. Nếu phát hiện trùng lặp hoặc vấn đề nhãn, ghi lại và thống nhất cách xử lý trước khi thay đổi.
2. Định dạng dữ liệu chung: `sample_id`, `text`, `label`; quy ước nhãn `0 = không mỉa mai`, `1 = mỉa mai`. Giữ phủ định và emoji; không xóa những dấu hiệu này bằng bước làm sạch áp dụng đại trà.
3. Tokenizer và bước mã hóa thuộc trách nhiệm mỗi chủ kiến trúc. Tất cả nhận cùng văn bản đầu vào và cùng split; không ép ba mô hình dùng chung tokenizer.
4. Phúc và Tuấn chốt cách tính chỉ số ngay 08–09/10. Mỗi lần chạy ghi kiến trúc/checkpoint, seed, split, siêu tham số, môi trường, Macro-F1 và F1 sarcastic. Tuấn dùng cùng một công thức std và ghi rõ trong báo cáo.
5. Đầu ra dự đoán chung: `sample_id`, `y_true`, `y_pred`; lưu tách theo phương pháp và seed. Có thể thêm điểm dự đoán nếu phương pháp hỗ trợ; không bắt baseline SVM phải trả xác suất.
6. Mỗi kiến trúc cung cấp hàm dự đoán một câu và cách nạp model/tokenizer. Kiệt bắt đầu demo với đầu ra mẫu theo giao diện đã chốt; sau đó nối checkpoint thật. Đầu ra mẫu chỉ phục vụ dựng giao diện, không được dùng làm kết quả thí nghiệm.
7. Chọn prompt, siêu tham số, checkpoint và kiến trúc cho demo dựa trên validation. Chốt lựa chọn trước khi báo cáo trên test; không dùng test để chỉnh prompt hoặc chọn mô hình thắng.
8. Mọi người kiểm tra khả năng chạy GPU trước 09/10. Chạy thử nhỏ chỉ để xác nhận pipeline, sau đó chạy đầy đủ dữ liệu bài chính. Nếu thiếu tài nguyên, báo leader ngay để điều chỉnh batch size/độ dài đầu vào có giải thích hoặc lịch dùng máy; không tự bỏ seed, bỏ kiến trúc hay đổi dataset.

## 4. Lịch nội bộ và điều kiện hoàn thành

| Hạn, giờ Việt Nam | Đầu việc phải xong | Người chịu trách nhiệm |
|---|---|---|
| **07/10** | Nhận nhiệm vụ; cùng đọc phần chung, mục 11 và bài báo nền tảng; Phúc chốt đầu ra bàn giao | Cả nhóm |
| **08/10, 20:00** | Dữ liệu/split và quy ước nhãn được chia sẻ; phân bố nhãn và ví dụ mẫu có sẵn | Huy chủ trì; Phúc duyệt; cả nhóm xác nhận |
| **09/10, 20:00** | Ba pipeline fine-tune chạy thử thành công; baseline/zero-shot có bản đầu; bộ metric dùng chung và khung demo chạy được | Mỗi người phần mình; Tuấn metric; Kiệt demo |
| **11/10, 20:00** | Bản nội bộ báo cáo tiến độ 1: tóm tắt bài báo, thống kê dữ liệu, số liệu baseline/zero-shot, tiến độ seed và vướng mắc của ba kiến trúc | Mỗi người viết phần mình; Tuấn ghép; Phúc duyệt |
| **12/10, 20:00** | Mỗi kiến trúc xong ít nhất một seed đầy đủ; các seed còn lại đã có lịch chạy; Phúc có kết quả prompt trên validation | Cả nhóm |
| **15/10, 20:00** | **Hạn gửi bản chạy được để review**: đủ chín lượt fine-tune, metric/prediction, code, cấu hình, hướng dẫn và báo cáo phần mình; demo đã nối model thật | Cả nhóm |
| **16–17/10** | Phúc review code và chạy thử; người phụ trách sửa phần mình; Tuấn lập báo cáo tiến độ 2 từ đủ kết quả, lỗi, demo và phần nâng cao | Phúc review; ba thành viên sửa và bổ sung |
| **18/10, 20:00** | Hết lỗi chặn; chạy tích hợp thành công; demo dùng model được chọn; hoàn thiện báo cáo/slide; Phúc review lại bản đã sửa | Cả nhóm |
| **19/10, trước 20:00** | Thành viên push đúng bản Phúc đã duyệt; kiểm tra bản trên GitHub chạy được; tổng duyệt thuyết trình và chốt bàn giao | Cả nhóm, Phúc duyệt cuối |

Không đợi đến 19/10 mới gửi code lần đầu. Bản gửi 15/10 phải chạy được, không chỉ là notebook đang viết hoặc lời báo “đã xong”. Lịch này dự kiến cần mỗi người chủ động chạy thí nghiệm song song; tiến độ thực tế phụ thuộc tài nguyên và lỗi phát sinh.

## 5. Gói bàn giao của từng người

- Code hoặc notebook chạy lại được, hướng dẫn lệnh/các bước và danh sách thư viện cần dùng.
- Cấu hình thực nghiệm, nguồn dữ liệu, split, seed và log/kết quả thực tế; không tự điền số liệu chưa chạy.
- Với mỗi kiến trúc: đủ ba seed, kết quả riêng từng seed, mean ± std, prediction và cách truy cập/nạp checkpoint cần cho demo hoặc tái hiện.
- Ví dụ lỗi kèm giải thích, phần báo cáo và slide do chính người phụ trách viết.
- Danh sách lỗi còn tồn tại và điều kiện máy đã dùng. Mỗi người phải giải thích được code và kết quả của mình.

“Xong” nghĩa là leader có thể làm theo hướng dẫn để kiểm tra phần việc, thấy đủ bằng chứng kết quả và xác nhận đáp ứng đề bài. Việc kiểm tra không nhất thiết phải huấn luyện lại toàn bộ chín lượt từ đầu.

## 6. Những yêu cầu đã ghi lại cho bước sau

**Yêu cầu của người dùng:** hiện tại tập trung chia task. Phần hướng dẫn GitHub và cách review code sẽ làm ở bước tiếp theo, không tự tạo repo hoặc đẩy code trong bước phân công.

- Nhóm gồm bốn người, Phúc là leader; công việc phải được chia cân bằng.
- Hoàn tất trước ngày 20; kế hoạch hiện đặt ngày chốt là 19/10/2026 và hạn gửi review là 15/10/2026.
- Mỗi thành viên sẽ tự push phần được phân công lên GitHub.
- **Phúc phải được xem, chạy thử và review code trước khi thành viên push lên GitHub.** Không diễn giải yêu cầu này thành chỉ review trước khi merge vào nhánh chính.
- Luồng dự kiến: thành viên gửi code/notebook hoặc gói file để Phúc xem → Phúc phản hồi → thành viên sửa → Phúc duyệt đúng phiên bản → thành viên mới push. Bản sửa sau khi duyệt cần được kiểm tra lại trước khi push.
- Cần hướng dẫn Phúc từng bước về nhận code, đọc thay đổi, chạy thử, kiểm tra tính đúng, ghi nhận lỗi, yêu cầu sửa và xác nhận bản được duyệt.
- Sau đó mới hướng dẫn tạo/cấu hình repo, chia nhánh, cách push và kiểm soát tích hợp. Chưa có URL repo, tài khoản GitHub hoặc môi trường máy của từng người trong yêu cầu hiện tại.
- Chưa đặt nhắc lịch tự động. Mục này là ghi chú yêu cầu để tiếp tục công việc ở lượt sau.

Nguồn tài liệu: `Pasted text.txt` do người dùng đính kèm (yêu cầu chung, mục 11, báo cáo tiến độ/cuối kỳ) và ảnh danh sách nhóm. Đây là kế hoạch phân công; chưa có code hoặc kết quả thực nghiệm được tạo hay xác minh.

## 7. Task, đầu vào, đầu ra và nghiệm thu

**Cập nhật theo yêu cầu ngày 12/10/2026:** phần trên mục 6 ghi lại yêu cầu ở lượt phân công ban đầu. Hướng dẫn GitHub và quy trình dưới đây đã được bổ sung; khi có khác biệt, tuân theo `AGENTS.md` hiện tại và hướng dẫn cộng tác tại [`doc/cach_lam.md`](cach_lam.md).

Tạo issue từ [GitHub Issue Form](../.github/ISSUE_TEMPLATE/team-task.yml). Issue tracker: <https://github.com/PhucBeossss123/an-y/issues>. Kiểm tra issue đã có trước khi tạo để tránh trùng. Tên nhánh dưới đây là đề xuất, không khẳng định nhánh đã được tạo trên remote.

| Người phụ trách | Mục tiêu / phạm vi | Đầu vào và phụ thuộc | Bàn giao và tiêu chí nghiệm thu | Nhánh đề xuất |
|---|---|---|---|---|
| Huỳnh Trần Nhật Phúc (`PhucBeossss123`) | Baseline TF-IDF + SVM; zero-shot sarcastic/sincere; prompt engineering; review, duyệt phiên bản và tổng kết | Dữ liệu/split chung và evaluator của nhóm từ Huy/Tuấn; code, kết quả hiện có trong `src/an_y`, `reports/`, `results/phuc/` | Chạy lại trên split nhóm sau khi chốt; ghi cấu hình và lệnh; Macro-F1 riêng với sarcastic F1; validation dùng chọn cấu hình, test chỉ báo cáo; prompt comparison và phân tích lỗi có ví dụ thực | `phuc/baseline-zero-shot` (đã tồn tại cục bộ và trên origin khi kiểm tra workspace) |
| Lê Quốc Huy (GitHub username: chờ cung cấp) | Loader/dữ liệu/split dùng chung và BERT-base, ba seed `42`, `123`, `2026` | Nguồn iSarcasmEval tiếng Anh chính thức; quy ước `sample_id`, `text`, `label`; giữ official test; không dùng rephrase/annotation làm feature | Manifest revision/checksum; thống kê nhãn; split cố định; loader được các thành viên dùng; mỗi seed có cấu hình, lệnh, metric, prediction và log; bàn giao code/checkpoint ngoài Git và phần báo cáo. Nghiệm thu khi Phúc tái hiện được giao diện và đối chiếu được artifact | `huy/data-bert` (đề xuất; chưa xác minh trên remote) |
| Trần Hoàng Tuấn (GitHub username: chờ cung cấp) | RoBERTa-base ba seed; evaluator dùng chung; mean ± std; so sánh và tổng hợp báo cáo | Split/loader của Huy; quy ước đầu ra chung; kết quả của ba chủ mô hình | Metric cùng công thức, tách Macro-F1 và sarcastic F1; confusion matrix; kết quả từng seed và mean ± std; phân tích tối thiểu ba nhóm lỗi, gồm phủ định và emoji; bảng chỉ dùng kết quả có artifact | `tuan/roberta-evaluation` (đề xuất; chưa xác minh trên remote) |
| Nguyễn Trần Anh Kiệt (GitHub username: chờ cung cấp) | DistilBERT ba seed; demo Gradio/Streamlit; hướng dẫn demo và ghép slide | Split/loader chung; prediction interface đã thống nhất; checkpoint thắng được chọn trên validation từ ba kiến trúc | Kết quả ba seed có cấu hình, lệnh và artifact; demo chạy với mô hình được chọn trên validation; hướng dẫn tái chạy; phần báo cáo và slide có số liệu thực, không dùng output mẫu làm kết quả | `kiet/distilbert-demo` (đề xuất; chưa xác minh trên remote) |

### Trạng thái được xác minh trong workspace ngày 12/10/2026

- Nhánh đang mở là `phuc/baseline-zero-shot`, sạch tại thời điểm kiểm tra; remote `origin` trỏ tới `PhucBeossss123/an-y`. Có `main` và nhánh Phúc ở local/remote.
- Code, kết quả baseline và zero-shot của Phúc đã có trong repo. Validation là split tạm seed 42; cần chạy lại khi Huy bàn giao split chung.
- Tài liệu review hiện ghi Huy, Tuấn và Kiệt chưa bàn giao code/kết quả và username GitHub còn chờ. Đây là trạng thái theo bằng chứng lưu trong repo, chưa phải xác nhận tiến độ mới trực tiếp từ thành viên.
- Hạn gửi bản chạy được để review là 15/10/2026, 20:00 giờ Việt Nam. Review/sửa 16–17/10; tích hợp 18/10; hoàn tất trước 20:00 ngày 19/10. Chưa đánh dấu các mốc này hoàn thành.
- Mỗi issue cần ghi người phụ trách, nhánh đề xuất, phụ thuộc, đầu ra, acceptance checklist, hạn review/hoàn tất, trạng thái và SHA được Phúc duyệt. Thành viên gửi ZIP/patch/bundle trước; chỉ push đúng SHA sau khi Phúc duyệt rõ ràng. Nếu sửa sau duyệt, gửi review lại trước lần push kế tiếp.

### Điều kiện cộng tác GitHub

Repo public (nếu được xác minh sau khi đổi visibility) chỉ cho phép xem và clone; không tự cấp quyền push. Muốn push trực tiếp, mỗi thành viên cần được thêm làm collaborator và chấp nhận lời mời bằng tài khoản riêng đã xác minh. Không đoán username, không tự mời người chưa xác minh. Review trước push là quy trình nhóm; branch protection không thay thế quy trình đó. Thành viên không push hoặc merge vào `main` theo quy ước nhóm.
