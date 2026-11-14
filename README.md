# Vietnamese Fine-grained Emotion Detection (Ekman-7)

Multi-label emotion classification on Vietnamese social media text.
ViGoEmotions dataset (28 labels) remapped to 7 Ekman categories.

**DAP391m -- AI and Data Science Project, FPT University, Fall 2026**
**Nhom 8**: Pham Hoang Gia Phuc (SE190403) · Nguyen Ha Anh Khoa (SE193979) · Pham Phu Quy (SE193670)

---

## Results

| Model | Type | F1-macro | F1-weighted | vs Baseline |
|---|---|---|---|---|
| **CafeBERT** | Vietnamese (XLM-R Large) | **67.20%** | **75.97%** | **+5.70 pp** |
| PhoBERT | Vietnamese (RoBERTa) | 64.85% | 73.77% | +3.35 pp |
| ViSoBERT | Vietnamese (BERT) | 64.66% | 74.19% | +3.16 pp |
| XLM-R base | Multilingual | 62.81% | 72.94% | +1.31 pp |
| TF-IDF + LR | Classical baseline | 55.51% | 65.22% | -5.99 pp |
| **Paper baseline** (EACL 2026) | ViSoBERT 28 labels | 61.50% | 63.26% | -- |

Training config: seed 42, epochs 12, batch 32, lr 5e-5, max_len 200, threshold 0.5, BCE + pos_weight

---

## Dataset

**ViGoEmotions** -- Vietnamese GoEmotions (EACL 2026)

- 20,664 Vietnamese social media comments
- 28 original GoEmotions labels remapped to 7 Ekman categories
- Conflict resolution: **neutral wins** (if a sample has neutral + other labels, keep neutral)
- Split: train 16,531 / val 2,066 / test 2,067
- License: research and education only -- raw data not committed to this repo

| Ekman label | GoEmotions labels mapped | % of total labels |
|---|---|---|
| joy | joy, amusement, approval, excitement, gratitude, love, optimism, relief, pride, admiration, desire, caring | 36.9% |
| sadness | sadness, disappointment, embarrassment, grief, remorse | 21.2% |
| anger | anger, annoyance, disapproval | 16.5% |
| surprise | surprise, realization, confusion, curiosity | 12.0% |
| fear | fear, nervousness | 5.5% |
| disgust | disgust | 4.1% |
| neutral | neutral | 3.8% |

---

## Research Questions

**RQ1** -- How well do models perform after remapping 28 labels to 7 Ekman categories?
CafeBERT leads at F1-macro 67.20%, surpassing the EACL 2026 paper baseline by +5.70 pp.

**RQ2** -- Which emotions are still confused after label merging -- data bias or model limitation?
Per-label F1 and co-occurrence analysis show confusion is mainly due to data distribution (e.g. slang misclassification) rather than model architecture.

**RQ3** -- Can post-processing techniques (per-label threshold tuning, model ensembling) reduce confusion?
Label-wise threshold tuning on val set + 4-PLM ensemble improves Macro-F1 on test.

---

## Project Structure

```
notebooks/
  01_eda_ekman7.ipynb          # EDA, SQL (DuckDB), data dictionary
  CafeBERT_ekman7.ipynb        # Fine-tuning CafeBERT
  PhoBERT_ekman7.ipynb         # Fine-tuning PhoBERT (Colab)
  ViSoBERT_ekman7.ipynb        # Fine-tuning ViSoBERT
  XLMR_ekman7.ipynb            # Fine-tuning XLM-R base
  TFIDF_LR_ekman7.ipynb        # Classical TF-IDF + Logistic Regression baseline

api/
  app.py                       # FastAPI: GET /health, POST /predict
  telegram_alert.py            # Telegram Bot alert (anger/disgust/fear >= 0.70)
  static/demo.html             # Web demo UI (Plotly bar chart, Ctrl+Enter)
  .env.example                 # Credentials template (token not committed)

reports/
  FINAl (1).pdf                # 10-page paper manuscript
  dashboard.html               # Interactive Plotly dashboard
  figures/
    sankey_28to7.png           # Sankey: 28 GoEmotions -> 7 Ekman
    architecture_diagram.png   # System architecture
    pipeline_ml.png            # ML pipeline
    cooccurrence_heatmap.png   # Label co-occurrence heatmap
    wordcloud_ekman7.png       # Word cloud per emotion (7 panels)
  submission_links.txt         # GitHub + dataset Drive link
  demo_link.txt                # Demo app + video links
  AI_Audit_Log_MERGED.xlsx     # AI Audit Log (21 entries, all 3 Dots)

runs/                          # Checkpoints and metrics -- large files not committed
```

---

## Demo App

```bash
# Install dependencies
pip install -r api/requirements.txt

# Set up credentials
cp api/.env.example api/.env
# Edit .env: add BOT_TOKEN and CHAT_ID

# Start server (requires checkpoint in runs/CafeBERT/best/)
uvicorn api.app:app --reload

# Open browser
http://localhost:8000
```

**API endpoints:**

```
GET  /health    -- model status and label list
POST /predict   -- {"text": "..."} -> {"emotions": [...], "scores": {...}, "alert_sent": bool}
```

Telegram alert fires when `anger`, `disgust`, or `fear` score >= 0.70.

---

## References

- Tran et al. (2026). ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts. *EACL 2026*.
- Nguyen et al. (2023). CafeBERT: A Pre-trained Language Model for Vietnamese.
- Conneau et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). *ACL 2020*.
- Nguyen & Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese. *Findings of EMNLP 2020*.
