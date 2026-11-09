import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from pathlib import Path

LABEL_NAMES = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
MODEL_NAME  = "uitnlp/CafeBERT"
CKPT_PATH   = Path(__file__).parent.parent / "runs" / "CafeBERT" / "best" / "best.pt"
MAX_LEN     = 200
THRESHOLD   = 0.5


class ModelSentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, model_name: str, pooling: str = "pooler"):
        super().__init__()
        self.pooling = pooling
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.drop = nn.Dropout(p=0.2)
        self.fc   = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = out.pooler_output if self.pooling == "pooler" else out.last_hidden_state[:, 0]
        return self.fc(self.drop(x))


def load_model(device: str = "cpu") -> ModelSentimentClassifier:
    model = ModelSentimentClassifier(
        n_classes=len(LABEL_NAMES),
        model_name=MODEL_NAME,
        pooling="pooler",
    )
    if CKPT_PATH.exists():
        ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[model] Loaded checkpoint epoch={ck.get('epoch')} val_f1={ck.get('val_f1'):.4f}")
    else:
        print(f"[model] WARNING: {CKPT_PATH} not found — using base HuggingFace weights only.")
    model.to(device)
    model.eval()
    return model


_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    return _tokenizer
