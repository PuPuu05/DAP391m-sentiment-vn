import os
import httpx
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

ALERT_THRESHOLD = 0.7
ALERT_LABELS    = {"anger", "disgust", "fear"}


def should_alert(scores: dict[str, float]) -> bool:
    return any(scores.get(label, 0.0) >= ALERT_THRESHOLD for label in ALERT_LABELS)


async def send_telegram_alert(text: str, scores: dict[str, float]) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        print("[alert] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping.")
        return False

    triggered = [l for l in ALERT_LABELS if scores.get(l, 0.0) >= ALERT_THRESHOLD]
    score_str = ", ".join(f"{l}={scores[l]:.2f}" for l in triggered)
    preview   = text[:200].replace("_", "\\_").replace("*", "\\*")

    message = (
        f"*Emotion Alert*\n"
        f"Triggered: `{', '.join(triggered)}`\n"
        f"Scores: `{score_str}`\n"
        f"Text: _{preview}_"
    )

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={
                "chat_id": CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            })
        success = resp.status_code == 200
        if success:
            print(f"[alert] Telegram sent — triggered: {triggered}")
        else:
            print(f"[alert] Telegram failed: {resp.text}")
        return success
    except Exception as e:
        print(f"[alert] Telegram error: {e}")
        return False
