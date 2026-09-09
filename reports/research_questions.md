# Research Questions — Vietnamese Sentiment Detection on UIT-VSFC

## Context

Dataset: UIT-VSFC (Vietnamese Students' Feedback Corpus)
- Train: 11,426 câu | Dev: 1,583 | Test: 3,166
- Labels: positive (49.4%) / negative (46.6%) / neutral (4.0%)
- Domain: feedback sinh viên về giảng viên và chương trình học

Các RQ bên dưới xuất phát từ quan sát thực tế trong EDA.

---

## RQ1 — Evaluation Metric

**Câu hỏi:** Khi dữ liệu bị mất cân bằng lớp nghiêm trọng (neutral chỉ 4%), metric nào phản ánh chính xác nhất hiệu quả của model trên UIT-VSFC?

**Lý do đặt RQ:** Accuracy có thể cao dù model bỏ qua hoàn toàn neutral. F1-weighted cũng bị ảnh hưởng tương tự vì neutral chỉ đóng góp 4% vào tổng.

**Bằng chứng sẽ dùng:** So sánh Accuracy, F1-weighted, F1-macro trên cùng kết quả model — chứng minh bằng số liệu thực tế tại sao cần dùng nhiều metric thay vì chỉ 1.

---

## RQ2 — Model Complexity vs Performance

**Câu hỏi:** Trên dataset có câu ngắn (median ~10 từ) như UIT-VSFC, PhoBERT có đạt weighted-F1 cao hơn đáng kể so với TF-IDF + classical ML không?

**Lý do đặt RQ:** PhoBERT mạnh ở ngữ cảnh dài — nhưng câu trong UIT-VSFC rất ngắn. Classical ML có thể đủ tốt với ít tài nguyên hơn.

**Bằng chứng sẽ dùng:** Bảng metric 5 model (TF-IDF + LR, SVM, XGBoost, RF, PhoBERT), so với baseline paper UIT-VSFC. Ghi nhận training time để đánh giá đánh đổi.

---

## RQ3 — Vietnamese Tokenization

**Câu hỏi:** Vietnamese word tokenization (underthesea) ảnh hưởng như thế nào đến hiệu quả của TF-IDF + classical ML so với whitespace tokenization đơn giản?

**Lý do đặt RQ:** Tiếng Việt có nhiều từ ghép quan trọng (nhiệt_tình, dễ_hiểu, khó_hiểu). Tách sai làm mất nghĩa và ảnh hưởng đến feature quality của TF-IDF.

**Bằng chứng sẽ dùng:** Chạy TF-IDF + Logistic Regression 2 lần — 1 lần dùng `.split()`, 1 lần dùng underthesea. So sánh F1-weighted và F1-macro.

---

## Papers tham khảo

### Baseline — Paper gốc 
- **Tên:** UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis
- **Tác giả:** Kiet Van Nguyen et al.
- **Năm:** 2018, KSE Conference
- **Kết quả:** Maximum Entropy classifier, F1 ~88% sentiment
- **Link:** https://ieeexplore.ieee.org/document/8573337

### PhoBERT — Model sẽ dùng cho deep learning
- **Tên:** PhoBERT: Pre-trained language models for Vietnamese
- **Tác giả:** Dat Quoc Nguyen & Anh Tuan Nguyen
- **Năm:** 2020, Findings of EMNLP 2020
- **Mô tả:** BERT-based model pre-trained trên 20GB văn bản tiếng Việt. Đây là model sẽ fine-tune trong RQ2.
- **Link:** https://arxiv.org/abs/2003.00744
- **HuggingFace:** `vinai/phobert-base`

### SOTA(2024)
- **Tên:** SVSD: A Comprehensive Framework for Vietnamese Sentiment Analysis
- **Phương pháp:** PhoBERT + FastText + SVM
- **Kết quả:** Accuracy ~94% trên UIT-VSFC
- **Link:** https://link.springer.com/chapter/10.1007/978-981-96-0434-0_26

### Model PhoBERT fine-tuned sẵn trên UIT-VSFC (tham khảo)
- **HuggingFace:** https://huggingface.co/tmt3103/VSFC-sentiment-classify-phoBERT

## Target của project

| Mốc | F1 mục tiêu | Ghi chú |
|-----|------------|---------|
| Baseline (paper gốc) | ~88% | Phải vượt |
| Mục tiêu thực tế | ~91-92% | PhoBERT fine-tune cơ bản |
| Tham vọng (gần SOTA) | ~94% | Cần tuning kỹ |
