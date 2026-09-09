# Project Planning

## Phát hiện cảm xúc của văn bản tiếng Việt trên tập dữ liệu UIT-VSFC

> **Môn học:** DAP391m: AI and Data Science Project  
> **Giảng viên:** Lê Võ Minh Thư  
> **Học kỳ:** Fall 2026, FPT University TP.HCM  
> **Ngày nộp:** 10/09/2026 (Tuần 1)

---

## 1. Thông tin nhóm

| Họ và tên | MSSV |
|-----------|------|
| Phạm Hoàng Gia Phúc | [MSSV] |
| Nguyễn Hà Anh Khoa | [MSSV] |
| Phạm Phú Quý | [MSSV] |

---

## 2. Đề tài

**Tên:** Phát hiện cảm xúc của văn bản tiếng Việt trên tập dữ liệu UIT-VSFC

**Mô tả:** Xây dựng pipeline phát hiện cảm xúc (positive / neutral / negative) từ văn bản phản hồi của sinh viên Việt Nam, sử dụng bộ dữ liệu UIT-VSFC. Pipeline bao gồm EDA, tiền xử lý văn bản tiếng Việt, huấn luyện và so sánh tối thiểu 5 mô hình (TF-IDF + classical ML và PhoBERT fine-tuning), đánh giá bằng F1-macro và F1-weighted để xử lý class imbalance nghiêm trọng (neutral chỉ chiếm 4%).

**Ứng dụng thực tế:** Hỗ trợ tự động phân tích phản hồi sinh viên ở quy mô lớn, giúp nhà trường nhanh chóng nhận diện vấn đề nổi bật trong giảng dạy và chương trình học.

---

## 3. Dataset

**Tên:** UIT-VSFC (Vietnamese Students' Feedback Corpus)  
**Nguồn:** HuggingFace - `tridm/UIT-VSFC`  
**Paper gốc:** https://ieeexplore.ieee.org/document/8573337

### Thống kê

| Split | Số câu |
|-------|-------:|
| Train | 11,426 |
| Dev | 1,583 |
| Test | 3,166 |
| **Tổng** | **16,175** |

### Phân bố nhãn (train set)

| Nhãn | Số câu | Tỉ lệ |
|------|-------:|------:|
| positive | 5,643 | 49.4% |
| negative | 5,325 | 46.6% |
| neutral | 458 | 4.0% |

> **Lưu ý:** Class imbalance nghiêm trọng - neutral chỉ 4%. Accuracy không phải metric phù hợp; cần dùng F1-macro để đánh giá công bằng cho cả 3 lớp.

### Data Dictionary

| Cột | Kiểu dữ liệu | Mô tả | Ví dụ |
|-----|------|-------|-------|
| Sentence | string | Câu phản hồi gốc của sinh viên | "Thầy giảng bài hay, nhiều ví dụ thực tế." |
| Topic | string | Chủ đề phản hồi | lecturer / program / others |
| Sentiment | string | Nhãn cảm xúc | positive / neutral / negative |
| Encoded_topic | int | Mã hóa số của Topic | 0 (lecturer), 1 (program), 3 (others) |
| Encoded_sentiment | int | Mã hóa số của Sentiment | 0 (negative), 1 (neutral), 2 (positive) |

---

## 4. Research Questions

### RQ1: Evaluation Metric

**Câu hỏi:** Khi dữ liệu bị mất cân bằng lớp nghiêm trọng (neutral chỉ 4%), metric nào phản ánh chính xác nhất hiệu quả của model trên UIT-VSFC?

| | |
|---|---|
| Biến đầu vào | Văn bản phản hồi sinh viên (text_clean) |
| Biến mục tiêu | Nhãn cảm xúc (Sentiment: positive / neutral / negative) |
| Metric | Accuracy, F1-weighted, F1-macro |
| Bước trả lời | Bước 5: Build model |

**Bằng chứng:** So sánh Accuracy, F1-weighted, F1-macro trên cùng kết quả model - chứng minh bằng số liệu thực tế tại sao cần dùng nhiều metric thay vì chỉ 1.

### RQ2: Model Complexity vs Performance

**Câu hỏi:** Trên dataset có câu ngắn (median ~10 từ) như UIT-VSFC, PhoBERT có đạt weighted-F1 cao hơn đáng kể so với TF-IDF + classical ML không?

| | |
|---|---|
| Biến đầu vào | Văn bản phản hồi sinh viên (text_clean) |
| Biến mục tiêu | Nhãn cảm xúc (Sentiment: positive / neutral / negative) |
| Metric | F1-weighted, F1-macro, training time |
| Bước trả lời | Bước 5: Build model |

**Bằng chứng:** Bảng metric 5 model (TF-IDF + LR, SVM, XGBoost, RF, PhoBERT), so với baseline paper. Ghi nhận training time để đánh giá đánh đổi tài nguyên-hiệu suất.

### RQ3: Vietnamese Tokenization

**Câu hỏi:** Vietnamese word tokenization (underthesea) ảnh hưởng như thế nào đến hiệu quả của TF-IDF + classical ML so với whitespace tokenization đơn giản?

| | |
|---|---|
| Biến đầu vào | Sentence gốc (whitespace split) và text_clean (underthesea tokenize) |
| Biến mục tiêu | Nhãn cảm xúc (Sentiment: positive / neutral / negative) |
| Metric | F1-weighted, F1-macro |
| Bước trả lời | Bước 5: Build model |

**Bằng chứng:** Chạy TF-IDF + Logistic Regression 2 lần - 1 lần dùng `.split()`, 1 lần dùng underthesea - so sánh F1-weighted và F1-macro.

---

## 5. Kế hoạch 10 tuần

| Tuần | Nội dung | Phụ trách | Deliverable |
|------|----------|-----------|-------------|
| T1 (Sep 8-11) | B1+B2: EDA, preprocessing, viz cơ bản, 3 bài báo baseline | Cả nhóm | Project Planning |
| T2 (Sep 14-18) | B3: Advanced viz, interactive dashboard, đọc kỹ 3 paper | Quý + Gia Phúc | Dashboard HTML |
| T3 (Sep 21-25) | B4+B5: API endpoint, TF-IDF + 4 classical ML models | Khoa + Quý | Research Proposal |
| T4 (Sep 28-Oct 2) | B5: PhoBERT fine-tune (Colab), B6: app demo | Khoa + Quý | PhoBERT results, app prototype |
| T5 (Oct 5-9) | Review 1, so sánh 5 model, viết Methodology | Cả nhóm | Audit Log đợt 1 |
| T6 (Oct 12-16) | RQ mới, lặp lại 6 bước, hyperparameter tuning | Gia Phúc + Khoa | RQ mới, kết quả tuning |
| T7 (Oct 19-23) | Review 2, viết Discussion + Conclusion | Cả nhóm | Audit Log đợt 2 |
| T8 (Oct 26-30) | Kiểm tra code, viết Abstract + Introduction + Reflection | Gia Phúc | Draft Final Report |
| T9 (Nov 2-6) | Review 3, finalize Final Report + slide 22 trang | Cả nhóm | Final Report, Audit Log đợt 3 |
| T10 (Nov 9-13) | Luyện trình bày, kiểm tra demo | Cả nhóm | Demo hoàn chỉnh |
| T11 (Nov 16+) | Chấm hội đồng | Cả nhóm | Vấn đáp + demo |

---

## 6. Bài báo tham khảo

### Paper 1: Baseline (bắt buộc vượt)

- **Tên:** UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis
- **Tác giả:** Kiet Van Nguyen et al.
- **Năm:** 2018, KSE Conference
- **Phương pháp:** Maximum Entropy classifier
- **Kết quả:** F1 ~88% trên sentiment task
- **Trạng thái đọc:** Đã đọc tóm tắt
- **Link:** https://ieeexplore.ieee.org/document/8573337

### Paper 2: Model deep learning sẽ dùng

- **Tên:** PhoBERT: Pre-trained language models for Vietnamese
- **Tác giả:** Dat Quoc Nguyen & Anh Tuan Nguyen
- **Năm:** 2020, Findings of EMNLP
- **Phương pháp:** BERT-based, pre-trained trên 20GB văn bản tiếng Việt
- **Vai trò trong project:** Model chính cho RQ2 - fine-tune trên UIT-VSFC
- **Trạng thái đọc:** Đã đọc tóm tắt
- **Link:** https://arxiv.org/abs/2003.00744

### Paper 3: SOTA hiện tại

- **Tên:** SVSD: A Comprehensive Framework for Vietnamese Sentiment Analysis
- **Năm:** 2024
- **Phương pháp:** PhoBERT + FastText + SVM ensemble
- **Kết quả:** ~94% accuracy trên UIT-VSFC
- **Trạng thái đọc:** Đã đọc tóm tắt
- **Link:** https://link.springer.com/chapter/10.1007/978-981-96-0434-0_26

### Mục tiêu F1

| Mốc | F1 mục tiêu | Ghi chú |
|-----|:-----------:|---------|
| Baseline (paper gốc) | ~88% | Phải vượt |
| Mục tiêu thực tế | ~91-92% | PhoBERT fine-tune cơ bản |
| Tham vọng (SOTA) | ~94% | Cần tuning kỹ |

---

## 6. Phân công và sản phẩm bàn giao

### Phân công theo bước dự án

| Bước | Nội dung | Người phụ trách |
|------|----------|-----------------|
| B1+B2: Business Understanding + Data Preparation | Problem Statement, RQ, Data Dictionary, EDA, preprocessing, SQL | Phạm Hoàng Gia Phúc |
| B3: Data Visualization advanced | Advanced charts, interactive dashboard Plotly | Phạm Phú Quý |
| B4: Integrating Services API | API endpoint, chatbot tích hợp | Phạm Phú Quý |
| B5: Build model | 5 mô hình, tuning, cross-validation, bảng metric | Nguyễn Hà Anh Khoa |
| B6: Develop applications | App demo, kết luận có số liệu | Phạm Phú Quý + Nguyễn Hà Anh Khoa |
| Research Proposal + Final Report | LaTeX Springer, slide 22 trang | Phạm Hoàng Gia Phúc |

### Sản phẩm bàn giao cuối kỳ

- Notebook và mã nguồn trên GitHub
- Ứng dụng có tích hợp AI services
- Final Report 10-12 trang (LaTeX Springer 1 cột)
- Slide theo template DAP391m
- File AI Audit Log riêng của từng thành viên (3 đợt nộp: tuần 5, 7, 9)

---

## 8. Rủi ro và phương án dự phòng

| Rủi ro | Khả năng xảy ra | Phương án dự phòng |
|---|---|---|
| PhoBERT không vượt baseline paper | Trung bình | Tuning hyperparameter, thử thêm class_weight để xử lý imbalance |
| Colab hết GPU quota khi train PhoBERT | Cao | Dùng Colab Pro hoặc chia nhỏ training, lưu checkpoint thường xuyên |
| Thành viên không hoàn thành đúng deadline | Thấp | Người phụ trách chính hỗ trợ, deadline nội bộ sớm hơn 2 ngày |
| Dataset quá nhỏ, model overfit | Thấp | Dùng cross-validation, regularization, dropout trong PhoBERT |
| API endpoint bị lỗi khi demo | Trung bình | Test kỹ trước ngày chấm, chuẩn bị video demo dự phòng |

---

## 7. Kế hoạch sử dụng AI và Audit Log

### Việc dự kiến dùng AI và cách kiểm chứng

| Việc dùng AI | Công cụ | Cách kiểm chứng |
|---|---|---|
| Giải thích khái niệm F1-macro, class imbalance | Claude | Đối chiếu với định nghĩa trong scikit-learn docs |
| Gợi ý cấu trúc pipeline EDA | Claude | Tự chạy từng bước và kiểm tra output thực tế |
| Gợi ý ngram_range cho TF-IDF | Claude | Thực nghiệm so sánh (1,1) vs (1,2), lấy kết quả thực tế |
| Viết SQL query DuckDB | Claude | Đối chiếu kết quả với value_counts() của pandas |
| Hỗ trợ viết code fine-tune PhoBERT | Claude | Chạy thử trên Colab, kiểm tra loss và F1 sau mỗi epoch |
| Debug lỗi trong quá trình training | Claude | Xác nhận fix bằng cách chạy lại và xem output |
| Hỗ trợ viết LaTeX Final Report | Claude | Biên dịch PDF, kiểm tra format theo chuẩn Springer |

### Audit Log tuần 1

*3 quyết định kỹ thuật của nhóm trong tuần 1. Sẽ chuyển vào file .xlsx sau khi có template chính thức.*

| Thành viên | Quyết định | Lý do | Kết quả |
|-------|-----------|-------|---------|
| Phạm Hoàng Gia Phúc | Chọn F1-macro làm primary metric thay vì Accuracy | Neutral chỉ 4%, model có thể đạt accuracy cao bằng cách bỏ qua hoàn toàn lớp này | Ghi nhận trong RQ1, báo cáo F1-macro, F1-weighted và Accuracy song song |
| Nguyễn Hà Anh Khoa | Thử nghiệm ngram_range=(1,2) thay vì (1,1) theo đề xuất ban đầu | Sentiment tiếng Việt thường phụ thuộc vào cụm từ như "không tốt", "rất thích", "quá tệ" | ngram_range=(1,2) cho F1-weighted cao hơn, quyết định giữ cấu hình này |
| Phạm Phú Quý | Dùng 3 loại chart khác nhau (pie, bar, histogram) thay vì bar chart chung | Mỗi biến có bản chất khác nhau: Sentiment dùng pie, Topic dùng bar, word_count dùng histogram | Cả 3 chart trực quan, thể hiện rõ đặc trưng từng biến |
