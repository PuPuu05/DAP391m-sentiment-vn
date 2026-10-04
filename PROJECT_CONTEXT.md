# PROJECT_CONTEXT — DAP391m FA26 · Vietnamese Fine-grained Emotion Detection (ViGoEmotions)

Cập nhật: 04/10/2026. Nguồn: các phiên trao đổi giữa Phúc và Claude (chat), kèm kiểm chứng trực tiếp trên PDF paper, repo và dữ liệu.
Dành cho Claude Code làm việc trong repo của project. **Đọc hết file này trước khi đề xuất thay đổi về RQ, thiết kế thí nghiệm hoặc phạm vi.**
File này thay thế mọi giả định trong file context cũ nếu hai bên mâu thuẫn (xem mục 3.2).

## 0. Nhãn độ tin cậy dùng trong file

| Nhãn | Nghĩa |
|---|---|
| [PAPER] | Đã đối chiếu trực tiếp với PDF ViGoEmotions (EACL 2026, tr. 2805–2831) |
| [DATA] | Tính trực tiếp từ train/val/test csv; tái lập bằng `data_audit.py` |
| [REPO] | Đọc từ repo github.com/ricardo-tran/ViGoEmotions |
| [SUY LUẬN] | Suy ra từ dữ kiện đã có, chưa có tài liệu xác nhận |
| [ĐỀ XUẤT] | Claude (chat) đề xuất, Phúc chưa chốt |
| [CHƯA KIỂM CHỨNG] | Chưa tra cứu hoặc chưa xác nhận |
| [CẦN QUYẾT ĐỊNH] | Phúc phải quyết trước khi làm tiếp |

---

## 1. Tóm tắt trạng thái (1 phút)

1. Project môn DAP391m (FPT University, FA26), làm theo nhóm, **hạn nộp 09/11/2026**. Dataset ViGoEmotions, backbone ViSoBERT, ablation 5 config (A0, A, B, C, D) quanh ASL và per-label threshold.
2. **Quyết định ngày 04/10:** Phúc không tự trói mình vào nhãn "applied" hay "research". Nhắm tới hội nghị nếu kết quả có tiềm năng; **mức tối thiểu là hoàn thành project cuối kỳ**. RQ được chọn theo tiêu chí: tạo ra hiểu biết mới, kiểm chứng được, bảo vệ được trước 09/11.
3. Framing "triage phản hồi khách hàng" (tỉ lệ chi phí r, ngân sách đọc Recall@k) đã **bị hạ khỏi trung tâm**: dữ liệu không phải phản hồi khách hàng, và tỉ lệ bình luận "vận hành" phụ thuộc mạnh vào nguồn (2%–34%).
4. **RQ trung tâm chưa đóng băng.** Đề xuất hiện tại (mục 7): nghiên cứu hiệu quả của ASL và per-label threshold khác nhau thế nào theo (a) tính chất nhãn và (b) nguồn dữ liệu. Phần theo nguồn cần Phúc quyết định vì cross-platform đang nằm trong Out of Scope.
5. Dữ liệu đã kiểm tra: ánh xạ nhãn đúng; **nguồn của từng câu khôi phục được từ `id`**; nhóm "rare" gần như không tồn tại (mất cân bằng chỉ 4,52 lần); grief gần như là tập con của sadness; neutral không loại trừ nhãn khác.
6. Đã có siêu tham số từ notebook gốc (mục 4.3b, do Phúc trích ngày 04/10, chưa được đối chiếu lại với notebook). **Notebook gốc không set seed** nên con số 61,50 là một lần chạy không tái lập chính xác được. Còn phải xác nhận 4 điểm (mục 4.3b) rồi tái lập config A.

---

## 2. Quyết định hướng đi và lịch sử

### 2.1 Lập trường đã nêu của Phúc

- Nghiêng về applied Data Science nhưng không loại trừ research; làm nghiên cứu ở trường giúp tiếp xúc nhiều mối quan hệ tốt (nói ở chat trước, khoảng 11–12/09). Không muốn khóa cứng vào một domain, sẽ thu hẹp dần vào những năm cuối. Đã yêu cầu **không push back quá mức khi bàn về hướng research**.
- Cải thiện con số là việc của researcher, còn nhóm là data science; muốn RQ nhấn vào quyết định thay vì chỉ một bảng số.
- **04/10:** "nếu không nộp được hội nghị cũng không sao, minimum là hoàn thành project cuối kỳ là đủ; nếu có tiềm năng thì cố gắng hoàn thành để nộp hội nghị". Hỏi việc dùng Claude xuyên suốt quá trình và đặt RQ có khả năng cao nộp được hội nghị, **không nhất thiết theo applied**.

### 2.2 Diễn biến (theo thứ tự)

1. RQ2 gốc: "ASL và per-label threshold cải thiện Macro F1 bao nhiêu?" → Phúc coi là hướng chạy theo con số.
2. Đề xuất đổi sang quyết định triển khai: chọn config theo tỉ lệ chi phí r = c_FN/c_FP. Một công cụ khác (nguồn không rõ) phản hồi bổ sung protocol (tune threshold lại theo từng r, bootstrap, v.v.). Phúc vẫn thấy hướng này hàn lâm: rigor dày, còn r là con số không có cơ sở thực.
3. Đề xuất đổi sang "ngân sách đọc" (Recall@k trên hàng đợi ưu tiên). Phúc hỏi có xa chủ đề "nhận dạng cảm xúc" không → có, ở trọng tâm đánh giá → đề xuất cấu trúc "lõi 28 nhãn + lớp ứng dụng".
4. Phúc hỏi cấu trúc đó cho "cái mới" gì → **novelty ở đây là bằng chứng mới, không phải phương pháp mới** (theo Out of Scope, không đề xuất kiến trúc mới).
5. **Quyết định 04/10** (mục 2.1).
6. Phúc dán một phản hồi (nguồn ngoài, chưa rõ) chọn RQ "đặc tính nhãn vs lợi ích của can thiệp". Claude đánh giá: hướng hợp lý nhưng có lỗi regression-to-mean và công suất thống kê thấp (mục 8.4).
7. Phúc gửi PDF paper → xem lại paper từ đầu (mục 4).
8. Phúc có quyền truy cập HF, gửi các thư mục trong repo → audit dữ liệu (mục 5) → phát hiện làm đổi nhiều giả định (mục 6).

### 2.3 Trạng thái các hướng đã bàn

| Hướng | Trạng thái | Lý do |
|---|---|---|
| Tỉ lệ chi phí r, winner-region | Hạ xuống phụ lục, tùy chọn | r không có cơ sở thực tế; tính hậu kỳ được từ xác suất đã lưu, không cần train lại |
| Ngân sách đọc Recall@k | Hạ | p(ops) = 25,5% và phụ thuộc cơ cấu nguồn; đọc 10% bình luận thì recall tối đa ≈ 39,5% |
| Streamlit dashboard | Giữ như deliverable của môn học | Không còn là "minh chứng trung tâm" |
| Đặc tính nhãn vs hiệu quả can thiệp | Ứng viên RQ trung tâm | Rẻ, dùng lại đúng pipeline; novelty vừa |
| Hiệu quả can thiệp theo nguồn / cross-platform | Ứng viên mới, **cần quyết định** | Dữ liệu cho phép khôi phục nguồn từ id; khác biệt giữa nguồn rất lớn |
| Dấu vết gán nhãn của LLM | **Không khả thi** với dữ liệu công khai | Bản HF chỉ có id, text, labels |
| Phân tích taxonomy/confusion thuần mô tả | Chỉ làm phân tích hỗ trợ | Paper gốc đã có (mục 4) |

---

## 3. Spec project gốc và các chỗ cần sửa

### 3.1 Spec giữ nguyên

- **Đề tài:** An Applied Multi-label Emotion Detection System for Vietnamese Social Media Feedback (tên có thể đổi theo RQ cuối).
- **Dataset:** ViGoEmotions (EACL 2026), 20.664 bình luận, 28 nhãn (27 cảm xúc + neutral, taxonomy GoEmotions), multi-label. Split train/val/test = 16.531/2.066/2.067.
- **Backbone:** ViSoBERT (chính), PhoBERT (đối chứng). Scenario 1 (giữ emoji, chuẩn hóa theo luật).
- **Ablation:** tất cả dùng ViSoBERT, Scenario 1.

| Config | Loss | Threshold |
|---|---|---|
| A0 | BCE không trọng số | 0,5 |
| A | BCE + pos_weight = (N − n_i)/n_i | 0,5 (tái lập baseline paper) |
| B | ASL | 0,5 |
| C | BCE + pos_weight | per-label, grid search trên val |
| D | ASL | per-label, grid search trên val |

- ASL: γ+ = 0, γ− = 4, clip = 0,05 làm điểm xuất phát. Threshold grid [0,10; 0,90] bước 0,05, objective là F1 từng nhãn, **chỉ tune trên val**. Seeds tối thiểu 13, 42, 2026.
- **Metric:** Macro F1 (chính), Weighted F1, per-label F1, operational recall (anger, disappointment, fear, grief).
- **Deliverables môn học:** Streamlit dashboard; checkpoint + `thresholds.json`; bảng ablation (5 config × ≥3 seeds); error analysis 100–200 câu; báo cáo 10–12 trang, Springer format.
- **Out of Scope (file gốc):** calibration, multimodal, LLM-based augmentation, kiến trúc mới, cross-platform, user study.

### 3.2 Các chỗ trong file context gốc cần sửa

1. **Dữ liệu không phải phản hồi khách hàng.** [PAPER] 13.743 bình luận mới được thu từ 144 bài đăng công khai về đời sống, tin tức, giải trí và chuyện cá nhân (crawl 53.890, giữ 13.743), cộng 6.921 câu UIT-VSMEC gán nhãn lại. Mô tả "customer feedback" là giả định về miền dữ liệu, chưa kiểm chứng. Nhóm anger/disappointment/fear/grief được chọn theo trực giác, không có dữ liệu nghiệp vụ. Phải nêu rõ là giả định.
2. **Ngưỡng rare < 2% (< 413 mẫu) cho nhóm rỗng.** [DATA] nhãn ít nhất là relief với 635 mẫu train (3,84%). Xem mục 5.4 về cách xử lý.
3. **Tiêu chí tái lập A "sai số ≤ 0,5%" là tùy tiện.** [PAPER] paper chỉ báo một lần chạy, không có SD (notebook gốc không set seed, mục 4.3b); ViSoBERT hơn CafeBERT chỉ 0,33 điểm (61,50 so với 61,17). Đổi thành: "nằm trong khoảng biến thiên giữa các seed", báo cả dev (62,33) lẫn test (61,50), và ghi rõ threshold 0,5 là giả định.
4. **PhoBERT pretrain trên 140GB** (Wikipedia, news, OSCAR-2301) [PAPER], không phải 20GB.
5. **Nền tảng có cả Reddit và X**, và khoảng 1/3 corpus (6.921/20.664 = 33,5%) là UIT-VSMEC [PAPER + DATA].
6. **Siêu tham số ASL không đến từ paper ViGoEmotions** (paper không dùng ASL, focal hay threshold, theo tìm từ khóa). Chúng là giá trị mặc định thường dùng của ASL gốc [CHƯA KIỂM CHỨNG nguồn cụ thể].
7. Các số 16.531/2.066/2.067, amusement 2.868 và relief 635 là số train [DATA]; paper chỉ nêu tỉ lệ 8:1:1 và số đếm toàn corpus.
8. **Novelty claim phủ định** ("chưa có công trình nào…") **chưa được tra cứu**; không đưa vào báo cáo trước khi tìm tài liệu.
9. **Danh sách nhãn khó/mơ hồ trong file gốc (disapproval, neutral, realization) thiếu disappointment.** [PAPER] disappointment có F1 49,88 (thấp thứ 6) và Kappa 0,425 (thấp thứ 2). Trong nhóm vận hành, chỉ disappointment vừa khó vừa mơ hồ; fear, grief và anger thì không.
10. **Cross-platform đang nằm trong Out of Scope** → [CẦN QUYẾT ĐỊNH] (mục 7).

---

## 4. Kết quả xem lại paper ViGoEmotions [PAPER]

### 4.1 Corpus và gán nhãn
- 20.664 bình luận = 6.921 UIT-VSMEC (gán nhãn lại) + 13.743 mới (Facebook, YouTube, Reddit, TikTok, Threads, X). Split 8:1:1.
- 3 LLM đề xuất nhãn (Gemini 2.0 Flash, Llama-3-70B, Gemma 3); 3 annotator là sinh viên đại học, người Việt bản ngữ. **Annotator rà soát tập nhãn LLM đề xuất sẵn** (giữ, thêm, bỏ), không gán độc lập từ đầu (Phụ lục C.3). Hệ quả: Kappa có thể bị đẩy lên do neo vào đề xuất của LLM, và còn phụ thuộc tần suất nhãn.
- Mô tả cách chốt nhãn **không nhất quán**: phần mô tả quy trình ở thân bài (Mục 3) nói giao của các annotator, có majority khi cần; Phụ lục C.3 nói nhãn nào có ít nhất hai annotator chọn thì giữ.
- Bảng so khớp LLM với người (Table 2): khớp hoàn toàn 8.436 (40,82%); chỉ bị bỏ bớt 3.459 (16,74%); chỉ được thêm 4.645 (22,48%); vừa thêm vừa bỏ 4.124 (19,96%). Tổng 20.664. Đây là mức người sửa đề xuất của LLM, **không phải độ đồng thuận giữa người với người**, và không phải trần hiệu năng của mô hình.

### 4.2 Kappa theo nhãn (Table 8)
Cao nhất đến thấp nhất: fear 0,7148; gratitude 0,6911; pride 0,6849; embarrassment 0,6402; … anger 0,5959; … grief 0,5379; … nervousness 0,4920; disgust 0,4908; neutral 0,4583; **disappointment 0,4250**; **disapproval 0,3400**.

### 4.3 Mô hình và kết quả
- ViSoBERT Scenario 1: dev Macro F1 62,33 (Table 3); **test Macro F1 61,50, Weighted F1 63,26** (Table 4). CafeBERT test 61,17 / 62,74.
- Setup huấn luyện nêu trong paper: 12 epochs, AdamW, lr 5×10⁻⁵, linear scheduler với warmup 1 epoch, dropout 20%, một tầng FC 28 đầu ra, BCEWithLogits + pos_weight_i = (N − n_i)/n_i.
- **Không tìm thấy trong văn bản paper** (tìm theo từ khóa): batch size, max length, seed, quy tắc chọn checkpoint, threshold, từ điển chuẩn hóa teencode. Chỉ báo một lần chạy, không có SD. Các giá trị này lấy từ notebook ở mục 4.3b.

### 4.3b Siêu tham số từ notebook gốc [REPO, theo bản trích của Phúc ngày 04/10; Claude chưa tự đọc notebook]

| Tham số | Giá trị |
|---|---|
| Checkpoint | `uitnlp/visobert` |
| `max_len` | 128 (mặc định của Dataset class); có comment kiểm tra thử 200 |
| `batch_size` | 32 |
| Epochs | 12 |
| Optimizer / lr | AdamW, 5e-5 |
| Scheduler | Linear, warmup = 1 epoch (`num_warmup_steps = len(train_loader)`) |
| Dropout | hidden 0,1 và attention 0,1 (từ config model), thêm `Dropout(p=0.2)` trước FC |
| Head | `Linear(768, 28)`, không activation |
| Loss | `BCEWithLogitsLoss` + `pos_weight = (N − n_i)/n_i` |
| Threshold | 0,5 (không thấy code đổi) |
| Chọn checkpoint | Lưu khi **val Macro F1 (ở ngưỡng 0,5)** tốt nhất (`best_f1`) |
| Seed | **Không set** (không thấy `set_seed`, `torch.manual_seed`, `random.seed`) |

Pipeline chuẩn hóa Scenario 1 (theo thứ tự): `remove_duplicate_chars` → `remove_duplicate_emoji` → `replace_teencode` (dùng `model/docs/teencode4.txt`) → giữ emoji. Các file liên quan trong `model/docs/`: `teencode4.txt`, `emojis.json`, `patterns.json`.

**Phải xác nhận trước khi tái lập (chưa rõ từ bản trích):**
1. `max_len` thực sự được truyền vào lúc huấn luyện là 128 hay 200 (xem điểm gọi Dataset, đừng chỉ dựa vào giá trị mặc định).
2. Đầu vào của FC là gì: `[CLS]`, `pooler_output` hay mean pooling.
3. Điểm test được đo từ checkpoint `best_f1` nạp lại, hay từ epoch cuối.
4. Cả ba file `teencode4.txt`, `emojis.json`, `patterns.json` được dùng ở hàm nào, và thứ tự hàm có đúng như trên không.

**Hệ quả:**
- Seed không set nghĩa là khởi tạo FC head và thứ tự batch là ngẫu nhiên, nên 61,50 là một lần chạy. Việc paper không báo SD là đúng như vậy; **không suy ra đó là nguyên nhân** vì paper không nói lý do.
- Chọn checkpoint bằng val Macro F1 ở ngưỡng 0,5 nghĩa là val đã dùng để chọn epoch; sau đó tune threshold per-label trên cùng val (C, D) thì val bị dùng hai lần, kết quả trên val lạc quan. Test không bị ảnh hưởng. Ghi rõ trong báo cáo.
- Với A0, B (xác suất có phân phối khác), tiêu chí "val Macro F1 ở 0,5" có thể chọn epoch khác với epoch tốt nhất ở threshold đã tune. Giữ **cùng một quy tắc chọn checkpoint cho mọi config** để so sánh công bằng; nếu còn thời gian, chạy thêm một phân tích độ nhạy với quy tắc khác.
- File csv trong `corpus/` là **văn bản thô** [DATA]: 432 câu có ≥3 chữ cái lặp liền, 1.347 câu có emoji lặp liền, câu đầu test còn teencode chưa chuẩn hóa. Vậy bước chuẩn hóa phải áp dụng trong pipeline, cho cả train, val và test, bằng cùng một hàm.
- Độ dài [DATA]: trung vị 11 từ; chỉ 0,18% câu (37) dài hơn 128 từ và 0,03% dài hơn 160 từ; token nhiều hơn từ nên tỉ lệ bị cắt thật hơi cao hơn, cần đo bằng tokenizer. Khác biệt giữa 128 và 200 nhỏ, nhưng vẫn phải khớp với giá trị đã dùng khi tái lập.
- Per-label (Table 5, macro avg khớp 61,50):

| Nhãn | Precision | Recall | F1 |
|---|---:|---:|---:|
| anger | 60,20 | 62,43 | 61,30 |
| disappointment | 47,27 | 52,79 | 49,88 |
| fear | 70,30 | 72,45 | 71,36 |
| grief | 58,04 | 75,58 | 65,66 |
| macro avg | 59,08 | 64,40 | 61,50 |

F1 thấp nhất: disapproval 35,78; neutral 44,05; realization 46,94; desire 48,24; confusion 49,43; disappointment 49,88.
- **Baseline đã lệch về recall** (macro recall 64,40 > macro precision 59,08; grief recall 75,58 so với precision 58,04) do pos_weight. [SUY LUẬN] threshold tối ưu F1 có thể nâng ngưỡng lên và làm giảm recall; hướng của C − A và A − A0 là câu hỏi thực nghiệm mở.

### 4.4 Những gì paper đã làm và chưa làm
- **Đã có:** ma trận nhầm lẫn theo top-3 nhãn (Table 12), tương quan đồng xuất hiện (Figure 5, Table 9), từ gây nhầm (Table 13), FP/FN theo nhãn (Figure 7), phân tích số nhãn dự đoán so với số nhãn gold (Table 6). → phân tích taxonomy/confusion thuần mô tả **gần như không còn novelty**.
- **Chưa có** (theo tìm từ khóa): ASL, focal loss, threshold theo nhãn, phương sai theo seed, kết quả theo nền tảng.
- **Limitations tự nhận:** ít nền tảng; gán nhãn mang tính chủ quan với nhãn trùng nghĩa; mô hình yếu với mỉa mai và ngữ cảnh tinh tế.

### 4.5 Repo và giấy phép [REPO]
- Thư mục `annotation/`: `llm_guideline_official.md` và 3 notebook (Gemini 2.0 Flash; Llama-3-70B-8192 qua Groq; Gemma-3-27b-it). **Chỉ có code và prompt, không có kết quả gán nhãn.** Mỗi notebook chỉ chạy thử 10 câu; zero-shot cùng prompt tiếng Việt; temperature 0,5, top_p 0,9; Gemini/Gemma top_k 27, max_output_tokens 5000; Llama max_tokens ngẫu nhiên 45–50 (paper nói max_tokens 45–50 cho cả ba mô hình, notebook Gemini và Gemma khác). Cách gộp ba LLM thành một tập đề xuất không có trong code. File làm việc `comment_25.03.16.csv` có 25.960 dòng (khác 20.664, cơ chế lọc chưa rõ).
- Thư mục `corpus/`: `train.csv`, `val.csv`, `test.csv`, `dataset_V1.xlsx` (3 sheet, **giống hệt** 3 csv), `label_dict.json`.
- Thư mục `model/`: notebook Jupyter huấn luyện và thư mục `model/docs/` (`teencode4.txt`, `emojis.json`, `patterns.json`). Siêu tham số ở mục 4.3b. README không nêu giấy phép cho code và các file từ điển; khi copy vào project, ghi rõ nguồn.
- **Giấy phép:** dữ liệu gated trên Hugging Face (`uitnlp/vigoemotions`); chỉ dùng cho nghiên cứu và giáo dục phi lợi nhuận; **cấm dùng thương mại, cấm phân phối lại**; được công bố tóm tắt và phân tích nếu không dựng lại được dữ liệu từ đó.

---

## 5. Audit dữ liệu [DATA] (chạy `data_audit.py` để tái lập)

### 5.1 Xác nhận
- Kích thước split khớp: 16.531 / 2.066 / 2.067. Cột: `id`, `text`, `labels` (chuỗi dạng `"[0, 1, 21]"`, phải parse).
- **Tổng nhãn khớp Figure 4 của paper** (amusement 3.569, sadness 3.484, annoyance 3.336, anger 1.963, disappointment 1.930, fear 960, grief 930, relief 753, neutral 1.031) → `label_dict.json` đúng, mọi phân tích theo nhãn an toàn.
- 39.446 lượt nhãn, 1,909 nhãn/câu. Số nhãn mỗi câu: 1 → 5.431, 2 → 7.706, 3 → 2.915, 4 → 438, 5 → 41 (train).
- Trùng nguyên văn với train: val 0, test 1. Trùng sau chuẩn hóa (bỏ hoa thường và dấu câu): val 61, test 54 (khoảng 3%) → rò rỉ nhẹ, nên ghi vào báo cáo.

### 5.2 Nguồn của từng câu khôi phục được từ `id`
- Tiền tố chữ: `tik` (TikTok), `you` (YouTube), `x`, `red` (Reddit), `thr` (Threads). `id` là số thuần: 13.444 câu.
- **[SUY LUẬN]** id số ≤ 6926 là UIT-VSMEC (đúng 6.921 câu, khớp số VSMEC của paper); id số > 6927 là Facebook thu mới (6.523 câu; 6.523 + 7.220 câu có tiền tố = 13.743 khớp số câu thu mới). Chưa có tài liệu xác nhận việc gán "id số lớn = Facebook".

| Nguồn | Số câu | Test | Có ≥1 nhãn vận hành | anger | disappointment | fear | grief |
|---|---:|---:|---:|---:|---:|---:|---:|
| VSMEC | 6.921 | 692 | 33,4% | 13,6% | 12,9% | 6,0% | 4,2% |
| Facebook mới | 6.523 | 672 | 33,5% | 14,6% | 10,3% | 5,5% | 7,0% |
| TikTok | 4.508 | 436 | 11,4% | 0,4% | 5,7% | 2,2% | 3,5% |
| YouTube | 1.960 | 198 | 10,5% | 1,4% | 4,6% | 3,8% | 1,1% |
| X | 524 | 46 | 2,3% | 0,2% | 0,2% | 1,5% | 0,4% |
| Reddit | 167 | 14 | 18,0% | 10,8% | 8,4% | 0,6% | 0,6% |
| Threads | 61 | 9 | 4,9% | 0 | 4,9% | 0 | 0 |

- Khoảng 97% anger và 80–81% fear, grief, disappointment nằm ở VSMEC + Facebook (65% dữ liệu).
- Tỉ lệ nguồn gần như bằng nhau giữa train/val/test → **test là in-distribution**; mọi nhận định theo nguồn trên split hiện tại không phải là kiểm tra dịch chuyển miền.
- Chỉ VSMEC, Facebook, TikTok, YouTube đủ lớn trong test để báo cáo riêng. Reddit 14, Threads 9, X 46 thì quá nhỏ.

### 5.3 Bình luận có nhãn vận hành
- Có ≥1 nhãn vận hành: 5.263/20.664 = **25,5%** (train 25,28%, val 27,15%, test 25,30%). Tổng lượt nhãn vận hành 5.783 (khớp tổng 4 nhãn ở Figure 4).
- 4.765 bình luận chỉ có 1 nhãn vận hành, 476 có 2, 22 có 3.
- Hệ quả: đọc 10% bình luận thì recall tối đa 0,10/0,253 ≈ 39,5%.
- Số mẫu test mỗi nhãn vận hành: anger 189, disappointment 197, fear 98, grief 86. Relief chỉ có 60 mẫu dương trong test → F1 theo nhãn rất nhiễu.

### 5.4 Nhóm "rare" gần như không tồn tại
- 12 nhãn nằm trong khoảng 635–821 mẫu train; tỉ lệ nhãn nhiều nhất / ít nhất chỉ **4,52**.
- 7 nhãn thấp nhất (train): relief 635, surprise 651, confusion 676, realization 687, remorse 707, nervousness 728, **fear 745**. Grief (747) đứng thứ 8, cách ranh giới đúng 2 mẫu.
- Hệ quả: quy tắc "bottom quartile" đưa fear vào nhóm rare, trái với điều muốn tách (fear quan trọng vì nghiệp vụ, không vì hiếm); ranh giới tùy ý. **Dùng số mẫu train như biến liên tục (Spearman), bỏ metric nhị phân "rare-label Macro F1".** Nếu vẫn cần nhóm, gọi là "low-frequency" và ghi danh sách nhãn cùng số mẫu ngay trong protocol.
- Mất cân bằng nhẹ như vậy có thể làm ASL ít tác dụng [SUY LUẬN, chưa kiểm tra]. Khác biệt giữa các nguồn lớn hơn nhiều so với khác biệt tần suất nhãn.

### 5.5 Đồng xuất hiện
- P(sadness | grief) = **0,83** → grief gần như là tập con của sadness; P(sadness | disappointment) = 0,60; P(annoyance | anger) = 0,44; P(nervousness | fear) = 0,43; P(disappointment | grief) = 0,22.
- Phân tích grief phải so với sadness, đừng coi nó là lớp tách biệt.

### 5.6 Neutral và nhiễu nhãn
- **Neutral không loại trừ:** 306/1.031 bình luận neutral (29,7%) còn mang nhãn khác, trái với guideline gán nhãn. → Giữ sigmoid độc lập cho cả 28 đầu ra, **không hậu xử lý neutral**; ghi vào error analysis.
- Có 759 cặp bình luận trùng sau chuẩn hóa: chỉ 41,9% cặp có cùng tập nhãn (Jaccard TB 0,66), nhưng 94,1% trong 170 nhóm nhất quán về việc có nhãn vận hành hay không. **Chưa xem nội dung các câu đó**, nên chênh lệch có thể do cùng văn bản đứng dưới các bài khác nhau chứ không hẳn là nhiễu nhãn. Chỉ dùng làm tham khảo.

---

## 6. Hệ quả thiết kế (tóm tắt)

1. Không dùng "phản hồi khách hàng" như một sự thật về dữ liệu; nếu giữ, chỉ như kịch bản giả định ghi rõ trong báo cáo.
2. p(ops) và tần suất từng nhãn vận hành phụ thuộc mạnh vào nguồn → mọi con số tổng hợp (Recall@k, chi phí) chỉ có nghĩa với đúng cơ cấu nguồn này.
3. Mất cân bằng nhãn nhẹ (4,52×) → không kỳ vọng ASL cải thiện lớn; kết quả "không có cải thiện ổn định" là kết quả có thể xảy ra và vẫn viết được.
4. Baseline A đã thiên về recall; **hướng ảnh hưởng của threshold per-label lên recall nhóm vận hành là mở**, đừng giả định nó tăng.
5. Annotation-artifact RQ không làm được từ dữ liệu công khai.
6. Nguồn dữ liệu là biến có thật và đo được → mở lại khả năng phân tích theo nguồn.

---

## 7. Các RQ ứng viên

| RQ | Novelty | Chi phí thêm | Rủi ro | Ghi chú |
|---|---|---|---|---|
| **R1. Đặc tính nhãn (support, độ khó, Kappa, đồng xuất hiện) vs hiệu quả ASL và threshold** | Vừa | Thấp (dùng 5 config, xác suất lưu sẵn) | Chỉ 28 nhãn → công suất thấp; kết quả có thể là "không có quan hệ ổn định" | Ứng viên chính đã được bàn nhiều |
| **R2. Hiệu quả can thiệp theo nguồn dữ liệu** (mức 1: F1 theo nguồn trên test hiện có) | Vừa–cao | Rất thấp, không train lại | Test per nguồn nhỏ; nguồn lẫn với chủ đề và quy trình gán nhãn (VSMEC gán riêng) | Nên làm kèm R1 |
| **R3. Dịch chuyển nền tảng** (mức 2: train VSMEC + Facebook, test TikTok/YouTube hoặc leave-one-source-out) | Vừa–cao | Cần split mới và train lại | Anger gần như không có ở TikTok (≈18 mẫu trên toàn TikTok); cần [CẦN QUYẾT ĐỊNH] vì ngoài Out of Scope | Paper gốc không có kết quả theo nền tảng (theo tìm từ khóa) |
| R4. Ổn định theo seed và preprocessing | Thấp–vừa | 5 seeds nếu đủ compute | Dễ thành báo cáo kỹ thuật | Bắt buộc báo cáo (RQ3 gốc) nhưng không làm trung tâm |
| R5. Taxonomy/confusion | Thấp | Thấp | Paper đã làm | Phân tích hỗ trợ cho error analysis |
| R6. Dấu vết gán nhãn của LLM | Cao | — | **Không khả thi** với bản công khai | Cần dữ liệu từ tác giả (corresponding author: kietnv@uit.edu.vn); không xây kế hoạch dựa vào điều này |
| Lớp ứng dụng (triage, dashboard) | Thấp | Thấp | Lệch chủ đề nếu để thành trung tâm | Chỉ là deliverable hạ lưu |

**Khuyến nghị hiện tại [ĐỀ XUẤT, chưa chốt]:** R1 + R2 làm một RQ trung tâm duy nhất:
> Hiệu quả của ASL và per-label threshold (so với BCE + pos_weight) thay đổi thế nào theo tính chất của nhãn và theo nguồn dữ liệu trên ViGoEmotions? Hai kỹ thuật bổ sung hay chồng lấn nhau?

R3 là bước mở rộng chỉ khi mốc 25/10 cho tín hiệu tốt và Phúc đồng ý đưa vào phạm vi. Lớp triage/dashboard giữ như deliverable hạ lưu.

**Ước lượng tầm vóc [ĐỀ XUẤT]:** bài đánh giá thực nghiệm trên một benchmark, một backbone, hợp workshop hoặc hội nghị khu vực. Chưa kiểm tra hạn nộp hay chính sách của hội nghị cụ thể nào; nhiều nơi yêu cầu khai báo việc dùng AI hỗ trợ, cần đọc trước khi nộp.

---

## 8. Protocol thí nghiệm

### 8.1 Cấu trúc huấn luyện
- **C dùng lại checkpoint của A, D dùng lại checkpoint của B**, chỉ khác threshold → chỉ cần train **A0, A, B**, mỗi loại × số seed (3 seeds: 9 lần train; 5 seeds nếu đủ compute).
- Seeds 13, 42, 2026 (thêm 2027, 2028 nếu có). **Mọi config dùng cùng ngân sách tuning** và cùng seed list; không chạy nhiều seed hơn cho config "hứa hẹn".
- **Ngay khi train xong, lưu xác suất sigmoid trên val và test** (kèm id, `.npz`) cho mọi run. Mọi phân tích sau đó (threshold, ΔF1, chi phí, theo nguồn) chạy trên file này, không cần train lại.
- Thiết kế 2×2 (BCE + pos_weight hay ASL) × (0,5 cố định hay threshold per-label): B − A là hiệu ứng loss, C − A là hiệu ứng threshold trên cùng xác suất, D − B là hiệu ứng threshold sau ASL. A0 là đối chứng cho pos_weight.
- Tương tác quan sát được (không gọi là causal): I_l = [F1_l(D) − F1_l(B)] − [F1_l(C) − F1_l(A)].

### 8.2 Threshold
- Tune **chỉ trên val**, test dùng một lần cho báo cáo cuối. Mọi lựa chọn (config, threshold, vùng chi phí nếu có) xác định trên val.
- Mỗi config (A0, A, B) có xác suất khác phân phối, nên **không dùng threshold của config này cho config khác**.
- Val chỉ có 2.066 câu; nhãn thấp tần suất chỉ có vài chục mẫu dương → threshold nhiễu. Nếu cần kiểm tra độ ổn định: bootstrap **theo bình luận** (không theo từng lượt nhãn), 200–500 lần, tune lại mỗi lần, chỉ cho 4 nhãn vận hành và vài cấu hình đại diện; không dùng bootstrap để chọn threshold cuối.
- Nếu làm phân tích chi phí: threshold tối ưu lý thuyết là 1/(1 + r) khi xác suất calibrate (≈ 0,09 với r = 10; ≈ 0,05 với r = 20), thấp hơn cận dưới 0,10 của grid → mở grid xuống khoảng 0,02 và đếm số threshold chạm biên. Tune lại threshold cho từng r và cho cả A0, A, B để so sánh công bằng.

### 8.3 Tái lập config A
- Dùng đúng setup ở mục 4.3 và 4.3b, **sau khi xác nhận 4 điểm còn mở** ở 4.3b. Copy `teencode4.txt`, `emojis.json`, `patterns.json` vào `configs/` (ghi nguồn); đặt `max_len` theo giá trị thực sự dùng khi huấn luyện, không chỉ theo mặc định.
- **Set seed tường minh** (Python, NumPy, PyTorch, DataLoader generator); notebook gốc không set, nên đây là khác biệt có chủ đích so với bản gốc, phải ghi trong báo cáo.
- Tiêu chí tái lập: paper chỉ có một lần chạy không seed, nên 61,50 là một mẫu. Coi là tái lập được nếu 61,50 nằm trong khoảng [min, max] của các seed của A, hoặc trong mean ± 2·SD. Với 3 seed tiêu chí này yếu; nếu compute cho phép, chạy A với 5 seed. Báo cả dev (62,33) và test (61,50).
- Dùng **cùng một quy tắc chọn checkpoint** (val Macro F1 ở 0,5) cho A0, A và B.

### 8.4 Phân tích R1: tránh các lỗi đã được chỉ ra
- **Regression to the mean:** không dùng "F1 của config A" làm biến giải thích cho ΔF1 = F1(B) − F1(A), vì F1(A) nằm ở cả hai vế → tương quan âm giả. Dùng F1 từng nhãn của paper (Table 5, một lần chạy độc lập của tác giả) hoặc F1 của A0 làm độ khó; vẫn còn cùng test set nên không loại hẳn.
- Biến giải thích: số mẫu train (liên tục), Kappa (có caveat neo vào LLM), độ khó độc lập, mức đồng xuất hiện. Kết quả: ΔF1, ΔP, ΔR theo nhãn.
- **Chỉ 28 nhãn:** dùng Spearman, scatter có tên từng điểm, bảng đầy đủ 28 nhãn × config × seed trong phụ lục. **Không** chạy hồi quy nhiều biến, không claim nhân quả, không chọn nhãn "đẹp" sau khi thấy kết quả. Chọn 4–6 nhãn tương phản theo tiêu chí khóa trước (ví dụ fear: Kappa cao và F1 cao; disapproval: cả hai thấp; relief: ít mẫu nhất).
- **Công suất:** test chỉ 2.067 câu, nhiều nhãn chỉ 60–110 mẫu dương. Tính SD giữa seed theo từng nhãn **trước**, rồi mới diễn giải. Với 3 seed, chênh lệch nhỏ hơn SD ghi là "không phân biệt được"; không dùng từ "statistically significant"; so sánh theo cặp cùng seed; báo tỉ lệ seed cùng dấu.
- Kappa chỉ là bằng chứng về mức đồng thuận gán nhãn, không phải lời giải thích nhân quả hay cận trên chính xác của F1.

### 8.5 Phân tích theo nguồn (R2)
- Chỉ báo cáo VSMEC, Facebook, TikTok, YouTube. Nêu rõ phép suy luận về nguồn (mục 5.2) trong báo cáo.
- Với R3 (nếu được duyệt): pool toàn bộ train+val+test của nguồn bị giữ lại để đủ mẫu; anger ở TikTok gần như không có nên không đánh giá anger ở đó.

---

## 9. Việc cần làm (thứ tự) và mốc [ĐỀ XUẤT, Phúc chưa chốt ngày]

1. Clone repo; **xác nhận 4 điểm còn mở ở mục 4.3b trực tiếp trong notebook** (`max_len` thực dùng, đầu vào FC, cách đo test, vai trò của ba file từ điển). Copy ba file từ điển vào `configs/`. Đặt dữ liệu vào `data/` (không commit).
2. Chạy `data_audit.py` trong repo của nhóm để xác nhận lại các số ở mục 5.
3. Tái lập A (một seed trước), so với 61,50 test / 62,33 dev; lưu xác suất.
4. Chạy A0 trước để xem hướng recall/precision (A − A0, và C − A khi có threshold), rồi chốt RQ.
5. Chạy B và hoàn tất 3 seeds cho A0, A, B; lưu xác suất; sinh C, D offline.
6. Phân tích theo nhãn và theo nguồn trên xác suất đã lưu.
7. Error analysis 100–200 câu (mỉa mai, teencode, emoji, câu ngắn, nhiều nhãn; grief so với sadness; neutral kèm nhãn khác).
8. Dashboard Streamlit (không đóng gói dữ liệu), báo cáo.

| Mốc | Việc |
|---|---|
| 11/10 | Tái lập A, lưu xác suất, khóa pre-registration (nhóm vận hành, định nghĩa nhóm low-frequency hoặc biến liên tục, biến giải thích R1, danh sách seed) |
| 18/10 | Xong A0 và B; sinh C và D |
| **25/10** | **Mốc quyết định:** pattern có lặp lại trên ≥2/3 seeds không; chênh lệch có vượt SD giữa seed không; có đủ để theo hướng hội nghị không; có đưa R3 vào không |
| 02/11 | Error analysis, dashboard |
| 09/11 | Nộp báo cáo |

Nguyên tắc ở mốc 25/10: **chọn câu chuyện mà dữ liệu thật hỗ trợ**, không quyết định "sẽ có bài hội nghị" trước. Nếu hiệu ứng nhỏ hoặc đảo dấu giữa seed, câu chuyện là "lợi ích của can thiệp không ổn định / không đồng đều" và vẫn đủ cho môn học.

---

## 10. Rủi ro và điều không được claim

- Không claim "chưa ai làm ASL/threshold trên ViGoEmotions" khi chưa tra cứu tài liệu. Không đưa claim đó vào báo cáo nếu chưa xác minh.
- Không gọi ngưỡng/nhóm nào là "rare" khi nhãn ít nhất còn 635 mẫu; không gọi fear hoặc grief là rare hay khó (fear F1 71,36 và Kappa 0,7148; grief F1 65,66); disappointment thì khó và mơ hồ.
- Không diễn giải 40,82% là trần hiệu năng của mô hình.
- Không nói "phản hồi khách hàng" như một sự thật về dữ liệu; không đặt tỉ lệ chi phí r như chân lý nghiệp vụ (chỉ là lưới độ nhạy minh họa).
- Không chọn lại nhóm vận hành, ngưỡng hay công thức sau khi thấy kết quả test.
- Không dùng thông tin test để chọn config hay threshold, kể cả cho dashboard.
- Không nhồi nhiều RQ vào một bài: chọn **một** RQ trung tâm, các RQ khác hỗ trợ.
- Mọi trích dẫn phải tự mở nguồn kiểm tra; không dùng nguồn chưa mở.
- Các nguồn ngoài đã dẫn trong cuộc trao đổi về chi phí (NeurIPS 2014 về F-measure, ví dụ cost-sensitive của scikit-learn, Elkan 2001 cho công thức threshold 1/(1 + r)) chưa được Claude mở và đọc [CHƯA KIỂM CHỨNG].

---

## 11. Cách làm việc và ràng buộc

- **Cách trả lời:** kết luận trước, rồi lý do, rồi bước cụ thể. Thẳng, trung thực, không rào đón; nêu rõ chỗ chưa kiểm chứng. Đề xuất khách quan, **không biện minh bằng việc viện dẫn project hay kinh nghiệm cũ** của Phúc. Không diễn giải lại động cơ của Phúc.
- Không push back quá mức khi Phúc bàn về hướng research (yêu cầu đã nêu). Nhưng vẫn phải nói rõ rủi ro và chỗ yếu của RQ khi có.
- **Dữ liệu:** giấy phép cấm phân phối lại. Không commit dữ liệu vào git; không in nội dung câu hàng loạt vào log, chat hay báo cáo; chỉ dùng vài ví dụ ngắn trong error analysis. Dashboard công khai chỉ chứa model và threshold, không kèm câu từ train/test. Notebook phải xóa output trước khi chia sẻ. Không dán token Hugging Face ở bất cứ đâu.
- **Phạm vi:** không đề xuất kiến trúc model mới, không đổi backbone chỉ để tăng số; ý hay nằm ngoài phạm vi thì ghi một dòng vào "Future work".
- Paper ICTAI 2026 về UAV của Phúc là việc riêng, **không liên quan** đến project này.
- Thông tin chưa biết: số thành viên và phân công trong nhóm; compute sẵn có (GPU, thời gian); yêu cầu chính xác của giảng viên về framing và format báo cáo.

---

## 12. File liên quan

| File | Nội dung |
|---|---|
| `data_audit.py` | Tái lập mọi số ở mục 5, chỉ in số tổng hợp (cần `data/train.csv`, `val.csv`, `test.csv`, `label_dict.json`) |
| `ViGoEmotions_EACL2026.pdf` | Paper gốc |
| repo `ricardo-tran/ViGoEmotions`: `annotation/`, `corpus/`, `model/` | Prompt và notebook gán nhãn; dữ liệu; notebook huấn luyện (chưa đọc) |