# DAP391m — Vietnamese Fine-grained Emotion Detection

## Project
Phat hien cam xuc chi tiet van ban tieng Viet — DAP391m, FPT University, T9-T11/2026.
Neu project du tot: nop len hoi nghi co Scopus index (RIVF, KSE, ACIIDS).

## Dataset
ViGoEmotions (Vietnamese GoEmotions) — dat tai `dataset/`.
Paper goc: EACL 2026 — baseline ViSoBERT dat F1-macro 61.5%, F1-weighted 63.26%.
Muc tieu la beat baseline do.

- 20,664 binh luan mang xa hoi tieng Viet
- 27 nhan cam xuc chi tiet (multi-label: moi cau co the co nhieu nhan)
- Split: train 16,531 / val 2,066 / test 2,067

## Pipeline muc tieu
1. EDA: label distribution, text length, co-occurrence matrix, class imbalance check
2. Preprocessing: lowercase, remove noise, xu ly emoji, tokenize tieng Viet (underthesea)
3. Baseline: TF-IDF + classical ML (multi-label: OneVsRest)
4. Nang cao: PhoBERT / ViSoBERT / CafeBERT fine-tuning
5. Evaluation: F1-macro, F1-weighted — compare voi EACL 2026 paper

## Stack
Python, pandas, scikit-learn, xgboost, transformers, torch, underthesea, matplotlib, seaborn

## Role cua Claude o day
Collaborator / co-developer — khong phai mentor day hoc.
Giup build code, debug, optimize, review logic. Giai thich bang tieng Viet, giu nguyen thuat ngu ky thuat tieng Anh. Khong dung emoji.

## Structure
- `dataset/` — ViGoEmotions (train/val/test + paper), KHONG commit len GitHub
- `data/processed/` — sau khi clean/tokenize
- `notebooks/` — EDA va experiments (Jupyter)
- `src/` — source code modular (preprocessing.py, features.py, train.py, evaluate.py)
- `models/` — saved model files, KHONG commit file lon
- `reports/` — bao cao, figures, ket qua
- `papers/` — paper tham khao
- `School_docs/` — tai lieu tu giang vien

## Deadline
Tuan cuoi hoc ki: ~09/11/2026 (chon hoi dong DAP391m).
