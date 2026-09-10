# Vietnamese Fine-grained Emotion Detection

Fine-grained emotion detection on Vietnamese social media text using the ViGoEmotions dataset.

DAP391m — AI and Data Science Project, FPT University, Fall 2026.

## Dataset

**ViGoEmotions** (EACL 2026)
- 20,664 Vietnamese social media comments
- 27 fine-grained emotion labels (multi-label: each comment may have multiple emotions)
- Split: train 16,531 / val 2,066 / test 2,067
- Published baseline (ViSoBERT): F1-macro 61.50%, F1-weighted 63.26%

## Project Structure

```
data/
  raw/          # Raw dataset — not committed to GitHub
  processed/    # Cleaned and tokenized data
notebooks/      # EDA, experiments, visualization
src/            # Source code (preprocessing, features, model, evaluation)
models/         # Saved model files — large files not committed
reports/        # Reports, figures, experiment results
papers/         # Reference papers
```

## Pipeline

1. EDA: label distribution, text length, co-occurrence matrix, class imbalance analysis
2. Preprocessing: lowercasing, noise removal, emoji handling, Vietnamese tokenization (underthesea)
3. Baseline: TF-IDF + classical ML with OneVsRest for multi-label classification
4. Advanced: PhoBERT / ViSoBERT / CafeBERT fine-tuning
5. Evaluation: F1-macro, F1-weighted — compared against EACL 2026 paper baselines

## Results

*To be updated after experiments.*

| Model | F1-macro | F1-weighted |
|---|---|---|
| TF-IDF + LR | - | - |
| PhoBERT fine-tuned | - | - |
| ViSoBERT fine-tuned | - | - |

## Team

| Name | Role |
|---|---|
| Pham Hoang Gia Phuc | Research & Report |
| Nguyen Ha Anh Khoa | Modelling |
| Pham Phu Quy | Visualization & App |

## References

- Hung et al. (2026). ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts. EACL 2026.
- Nguyen & Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP.
