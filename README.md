# Vietnamese Sentiment Detection

Phat hien cam xuc cua van ban tieng Viet — DAP391m project, FPT University, T9-T11/2026.

## Problem Statement

TODO: Phat bieu bai toan cu the sau khi nhom thong nhat.

## Dataset

UIT-VSFC (Vietnamese Students Feedback Corpus)
- Source: University of Information Technology, VNUHCM
- Labels: TODO (positive/negative/neutral hoac multi-class emotion)
- Size: TODO

## Project Structure

```
data/
  raw/          # Du lieu goc chua xu ly — KHÔNG commit len GitHub
  processed/    # Du lieu da clean, tokenize, encode
notebooks/      # EDA, experiments, visualization
src/            # Source code (preprocessing, features, model, evaluation)
models/         # Saved model files — KHÔNG commit file lon len GitHub
reports/        # Bao cao, hinh anh, ket qua thi nghiem
```

## Pipeline

1. Data collection & understanding
2. Text preprocessing (lowercase, remove noise, tokenize tieng Viet)
3. EDA (class distribution, text length, word frequency)
4. Feature extraction (TF-IDF baseline)
5. Model training & evaluation (Logistic Regression → XGBoost → PhoBERT)
6. Comparison with published baselines (UIT-VSFC paper)

## Results

TODO: Dien vao sau khi co ket qua thi nghiem.

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + LR (baseline) | - | - |
| TF-IDF + XGBoost | - | - |
| PhoBERT fine-tuned | - | - |

## Team

TODO

## References

- UIT-VSFC paper: TODO them citation
