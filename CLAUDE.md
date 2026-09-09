# DAP391m — Vietnamese Sentiment Detection

## Project
Phat hien cam xuc van ban tieng Viet — DAP391m, FPT University, T9-T11/2026.
Neu project du tot: nop len hoi nghi co Scopus index (RIVF, KSE, ACIIDS).

## Dataset
UIT-VSFC (Vietnamese Students Feedback Corpus) — dat tai `data/raw/`.
Paper goc cua UIT-VSFC co baseline results — muc tieu la beat baseline do.

## Pipeline muc tieu
1. EDA: class distribution, text length, word frequency, class imbalance check
2. Preprocessing: lowercase, remove noise, tokenize tieng Viet (underthesea hoac pyvi)
3. Baseline: TF-IDF + Logistic Regression / XGBoost
4. Nang cao: PhoBERT fine-tuning (vinai/phobert-base tu HuggingFace)
5. Evaluation: Accuracy, Precision, Recall, F1 (weighted) — compare voi UIT-VSFC paper

## Stack
Python, pandas, scikit-learn, xgboost, transformers, torch, underthesea, matplotlib, seaborn

## Role cua Claude o day
Collaborator / co-developer — khong phai mentor day hoc.
Giup build code, debug, optimize, review logic. Giai thich bang tieng Viet, giu nguyen thuat ngu ky thuat tieng Anh. Khong dung emoji.

## Structure
- `data/raw/` — dataset goc, KHONG commit len GitHub
- `data/processed/` — sau khi clean/tokenize
- `notebooks/` — EDA va experiments (Jupyter)
- `src/` — source code modular (preprocessing.py, features.py, train.py, evaluate.py)
- `models/` — saved model files, KHONG commit file lon
- `reports/` — bao cao, figures, ket qua

## Deadline
Tuan cuoi hoc ki: ~09/11/2026 (chon hoi dong DAP391m).
