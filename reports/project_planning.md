# Project Planning

## Vietnamese Fine-grained Emotion Detection on ViGoEmotions

**Mon hoc:** DAP391m: AI and Data Science Project
**Giang vien:** Le Vo Minh Thu
**Hoc ky:** Fall 2026, FPT University TP.HCM
**Ngay nop:** 15/09/2026 (Tuan 2)

---

## 1. Thong tin nhom

| Ho va ten | MSSV |
|---|---|
| Pham Hoang Gia Phuc | SE190403 |
| Nguyen Ha Anh Khoa | SE193979 |
| Pham Phu Quy | SE193670 |

---

## 2. De tai

**Ten:** Vietnamese Fine-grained Emotion Detection on ViGoEmotions

**Mo ta:** Xay dung pipeline phat hien 27 cam xuc chi tiet tu binh luan mang xa hoi tieng Viet, su dung bo du lieu ViGoEmotions (EACL 2026). Pipeline bao gom EDA, tien xu ly van ban tieng Viet (teencode normalization, xu ly emoji), TF-IDF baseline, va ViSoBERT fine-tuning. Danh gia bang F1-macro va F1-weighted, so sanh voi baseline paper.

**Ung dung thuc te:** Ho tro phan tich phan ung cam xuc cua nguoi dung mang xa hoi o do phan giai cao (27 cam xuc thay vi 3 lop), phuc vu brand monitoring, content moderation, va nghien cuu tam ly xa hoi.

---

## 3. Dataset

**Ten:** ViGoEmotions (Vietnamese GoEmotions)
**Nguon:** EACL 2026 -- Hung et al.
**Paper goc:** ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts

### Thong ke split

| Split | So cau |
|---|---|
| Train | 16,531 |
| Val | 2,066 |
| Test | 2,067 |
| Tong | 20,664 |

### Phan bo nhan (train set)

| Nhan | So cau |
|---|---|
| amusement (0) | 2,868 |
| approval (21) | 2,785 |
| admiration (25) | 2,662 |
| ... | ... |
| disapproval (13) | 651 |
| relief (10) | 635 |

Chenh lech nhan nhieu nhat va it nhat: 4.5 lan.

**EDA facts:**
- 24.9% cau chua emoji
- 15.3% cau chua teencode
- 75% cau duoi 81 ky tu
- Median 2 nhan/cau (multi-label thuc su)

### Data Dictionary

| Cot | Kieu | Mo ta | Vi du |
|---|---|---|---|
| id | string | ID binh luan goc | "tik000008", "5743" |
| text | string | Noi dung binh luan | "buc anh xuat sac" |
| labels | list[int] | Chi so cam xuc (0-26), multi-label | [2, 8, 3] |

---

## 4. Research Questions

### RQ1: Teencode Normalization

**Cau hoi:** Teencode normalization (rule-based dictionary) co cai thien F1-macro cua model so voi khong normalize khong?

**Bang chung:** 15.3% cau trong tap train chua teencode (ko, dc, vs, mn...). BERT tokenize cac tu nay thanh subword la, mat ngu nghia. Paper goc khong thuc hien buoc nay.

**Phuong phap so sanh:**
- S0: Khong normalize (baseline paper)
- S1: Rule-based dictionary (teencode -> chuan)
- S2 (tuy thoi gian): ViSoLex model-based normalization

### RQ2: Emoji Strategy

**Cau hoi:** Chien luoc xu ly emoji nao cho ket qua tot nhat khi ket hop voi teencode normalization tu RQ1?

**Bang chung:** 24.9% cau chua emoji (gan 1/4 tap train), trung binh 1.91 emoji/cau (trong so cau co emoji). Paper goc thu 3 scenario rieng le nhung chua ket hop voi teencode normalization.

**Phuong phap so sanh:**
- E1: Giu nguyen emoji (paper S1)
- E2: Convert emoji sang text mo ta (paper S2)
- E3: Xoa emoji (paper S3)

Ket hop voi RQ1 tao thanh ma tran thu nghiem.

### RQ3: Per-label Threshold Tuning

**Cau hoi:** Per-label threshold tuning co cai thien F1-macro cho nhan thieu so so voi threshold 0.5 co dinh khong?

**Bang chung:** Paper dung threshold 0.5 cho tat ca 27 nhan. Nhan thieu so (relief 635 cau) kho dat xac suat > 0.5 -- model thien ve khong du doan nhan do. Dieu chinh threshold rieng cho tung nhan dua tren val set co the cai thien recall cho nhan thieu so.

**Phuong phap:** Grid search threshold [0.3, 0.35, 0.4, 0.45, 0.5] tren val set cho tung nhan, danh gia tren test set.

---

## 5. Ke hoach 10 tuan

| Tuan | Noi dung | Phu trach | Deliverable |
|---|---|---|---|
| T1 (Sep 8-14) | EDA, Project Planning | Ca nhom | Project Planning |
| T2 (Sep 14-21) | Preprocessing pipeline (RQ1+RQ2) | Gia Phuc | preprocessing.py |
| T3 (Sep 21-28) | TF-IDF baseline + classical ML | Khoa | Baseline results |
| T4 (Sep 28-Oct 5) | ViSoBERT/PhoBERT fine-tuning | Khoa | BERT results |
| T5 (Oct 5-12) | Ablation RQ1+RQ2, so sanh cac scenario | Gia Phuc | Ablation table |
| T6 (Oct 12-19) | Per-label threshold tuning (RQ3) | Khoa + Gia Phuc | RQ3 results |
| T7 (Oct 19-26) | Visualization, interactive dashboard | Quy | Dashboard HTML |
| T8 (Oct 26-Nov 2) | Viet bao cao, demo app | Gia Phuc + Quy | Draft report |
| T9 (Nov 2-9) | Finalize report + slide 22 trang | Ca nhom | Final report |
| T10 (Nov 9+) | Luyen trinh bay, kiem tra demo | Ca nhom | Demo hoan chinh |

---

## 6. Bai bao tham khao

### Paper 1 -- Baseline bat buoc vuot

- **Ten:** ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts
- **Tac gia:** Hung et al.
- **Nam:** 2026 | **Hoi nghi:** EACL
- **Phuong phap:** ViSoBERT fine-tuning voi 3 preprocessing scenarios (giu emoji, convert text, model-based)
- **Ket qua:** F1-macro 61.50%, F1-weighted 63.26%
- **Vai tro trong project:** Dataset source + baseline bat buoc vuot

### Paper 2 -- Model chinh

- **Ten:** PhoBERT: Pre-trained language models for Vietnamese
- **Tac gia:** Dat Quoc Nguyen & Anh Tuan Nguyen
- **Nam:** 2020 | **Venue:** EMNLP Findings
- **Phuong phap:** BERT pre-trained tren 20GB van ban tieng Viet (RoBERTa architecture)
- **Ket qua:** SOTA nhieu NLP task tieng Viet tai thoi diem cong bo
- **Vai tro trong project:** Model backbone de fine-tune, so sanh voi ViSoBERT

### Muc tieu F1

| Moc | F1-macro | Ghi chu |
|---|---|---|
| Baseline (paper goc) | 61.50% | Phai vuot |
| Muc tieu thuc te | 64-66% | Cai thien preprocessing |
| Tham vong | 68%+ | Neu RQ3 per-label threshold hieu qua |

---

## 7. AI Audit Log

| Thanh vien | Quyet dinh | Ly do | Ket qua |
|---|---|---|---|
| Pham Hoang Gia Phuc | Chuyen dataset tu UIT-VSFC sang ViGoEmotions | Giang vien dinh huong ViGoEmotions; 27 nhan chi tiet hon 3 nhan, phu hop huong fine-grained emotion detection | Toan bo codebase tai cau truc cho ViGoEmotions; README va CLAUDE.md cap nhat |
| Pham Hoang Gia Phuc | Dung encoding utf-8 voi encoding_errors replace thay vi latin-1 | latin-1 load duoc nhung hien thi tieng Viet sai; replace giu dung ky tu UTF-8 | CSV load thanh cong, tieng Viet hien thi dung trong tat ca 3 split |
| Pham Hoang Gia Phuc | Dat 3 RQ tap trung vao preprocessing thay vi model architecture | Paper da co ViSoBERT baseline tot; gap thuc su nam o preprocessing (teencode, emoji chua toi uu) va threshold co dinh 0.5 | 3 RQ co bang chung tu EDA: 15.3% teencode, 24.9% emoji, 4.5x label imbalance |
