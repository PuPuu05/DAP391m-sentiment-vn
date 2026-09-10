# Vietnamese Fine-grained Emotion Detection

Phat hien cam xuc chi tiet van ban tieng Viet — DAP391m project, FPT University, T9-T11/2026.

## Dataset

ViGoEmotions (EACL 2026)
- 20,664 binh luan mang xa hoi tieng Viet
- 27 nhan cam xuc chi tiet (multi-label)
- Split: train 16,531 / val 2,066 / test 2,067
- Baseline (ViSoBERT): F1-macro 61.50%, F1-weighted 63.26%

## Project Structure

```
data/
  raw/          # Du lieu goc — KHONG commit len GitHub
  processed/    # Du lieu da clean, tokenize
notebooks/      # EDA, experiments, visualization
src/            # Source code (preprocessing, features, model, evaluation)
models/         # Saved model files — KHONG commit file lon
reports/        # Bao cao, figures, ket qua
papers/         # Paper tham khao
School_docs/    # Tai lieu tu giang vien — KHONG commit len GitHub
```

## Pipeline

1. EDA: label distribution, text length, co-occurrence, class imbalance
2. Preprocessing: lowercase, remove noise, xu ly emoji, tokenize (underthesea)
3. Baseline: TF-IDF + classical ML (OneVsRest)
4. Nang cao: PhoBERT / ViSoBERT / CafeBERT fine-tuning
5. Evaluation: F1-macro, F1-weighted — compare voi EACL 2026 paper

## Results

TODO: Dien vao sau khi co ket qua thi nghiem.

| Model | F1-macro | F1-weighted |
|---|---|---|
| TF-IDF + LR (baseline) | - | - |
| PhoBERT fine-tuned | - | - |
| ViSoBERT fine-tuned | - | - |

## Team

- Pham Hoang Gia Phuc — Research & Report
- Nguyen Ha Anh Khoa — Modelling
- Pham Phu Quy — Visualization & App
