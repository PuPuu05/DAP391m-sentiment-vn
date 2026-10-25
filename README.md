# Vietnamese Emotion Detection (Ekman-7)

Emotion detection on Vietnamese social media text using the ViGoEmotions dataset remapped to 7 Ekman emotion categories.

DAP391m — AI and Data Science Project, FPT University, Fall 2026.

## Dataset

**ViGoEmotions** (EACL 2026) -- remapped to Ekman-7
- 20,664 Vietnamese social media comments
- 7 labels: joy, sadness, anger, fear, surprise, disgust, neutral (multi-label)
- Split: train 16,531 / val 2,066 / test 2,067
- Original 28-label baseline (ViSoBERT): F1-macro 61.50%, F1-weighted 63.26%

## Project Structure

```
notebooks/      # Training experiments per model
runs/           # Saved checkpoints and metrics — large files not committed
reports/        # Reports, figures, experiment results
papers/         # Reference papers
```

## Experiments

| Model | F1-macro | F1-weighted |
|---|---|---|
| CafeBERT | - | - |
| XLM-R base | - | - |

Config: seed 42, epochs 12, batch 32, lr 5e-5, max_len 200, threshold 0.5, BCE + pos_weight

## Team

| Name | Role |
|---|---|
| Pham Hoang Gia Phuc | Research & Report |
| Nguyen Ha Anh Khoa | Modelling |
| Pham Phu Quy | Visualization & App |

## References

- Hung et al. (2026). ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts. EACL 2026.
- Nguyen & Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese. Findings of EMNLP.
