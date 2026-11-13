import torch
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

_STATIC = Path(__file__).parent / "static"

from .model import load_model, get_tokenizer, LABEL_NAMES, MAX_LEN, THRESHOLD
from .telegram_alert import should_alert, send_telegram_alert

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model(DEVICE)
    yield


app = FastAPI(
    title="Ekman-7 Emotion Detection API",
    description="Fine-grained Vietnamese emotion detection — DAP391m Nhóm 8",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=_STATIC), name="static")


# ---------- schemas ----------

class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    emotions: list[str]
    scores: dict[str, float]
    alert_sent: bool


# ---------- routes ----------

@app.get("/")
def root():
    return RedirectResponse(url="/static/demo.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "CafeBERT-Ekman7",
        "device": DEVICE,
        "labels": LABEL_NAMES,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    tokenizer = get_tokenizer()
    enc = tokenizer(
        req.text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = _model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    scores   = {label: round(float(p), 4) for label, p in zip(LABEL_NAMES, probs)}
    emotions = [label for label, p in zip(LABEL_NAMES, probs) if p >= THRESHOLD]

    alert_sent = False
    if should_alert(scores):
        alert_sent = await send_telegram_alert(req.text, scores)

    return PredictResponse(
        text=req.text,
        emotions=emotions,
        scores=scores,
        alert_sent=alert_sent,
    )
