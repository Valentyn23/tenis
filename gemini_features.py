import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# =========================================================
# INIT
# =========================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

MODEL = "gemini-1.5-flash"

TIMEOUT = 25
CACHE_TTL = 60 * 60 * 6   # 6 часов
SLEEP = 0.6

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# =========================================================
# DEFAULT SAFE VALUES
# =========================================================
def neutral():
    return {
        "injury_risk":0.5,
        "mental_form":0.5,
        "motivation":0.5,
        "fatigue_risk":0.5,
        "confidence":0.5
    }


# =========================================================
# CACHE
# =========================================================
def cache_path(player):
    name = player.replace(" ","_").lower()
    return CACHE_DIR / f"{name}.json"


def load_cache(player):

    path = cache_path(player)

    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        if time.time() - data["time"] > CACHE_TTL:
            return None

        return data["value"]

    except:
        return None


def save_cache(player,value):

    try:
        path = cache_path(player)

        with open(path,"w") as f:
            json.dump({
                "time":time.time(),
                "value":value
            },f)

    except:
        pass


# =========================================================
# NEWS FETCHER
# =========================================================
def get_player_news(player_name, max_articles=5):

    if not NEWS_API_KEY:
        return []

    try:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": player_name,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": NEWS_API_KEY
        }

        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()

        data = r.json()

        texts = []

        for a in data.get("articles",[]):

            title = a.get("title","")
            desc = a.get("description","")

            text = (title + ". " + desc).strip()

            if len(text) > 40:
                texts.append(text)

        return texts

    except:
        return []


# =========================================================
# GEMINI CALL
# =========================================================
def ask_gemini(prompt):

    if not GEMINI_API_KEY:
        return None

    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"

        body = {
            "contents":[{"parts":[{"text":prompt}]}]
        }

        r = requests.post(url, json=body, timeout=TIMEOUT)
        r.raise_for_status()

        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    except:
        return None


# =========================================================
# PROMPT
# =========================================================
def build_prompt(player, news_list):

    news_text = "\n".join(news_list)

    return f"""
You are a sports analytics model.

Analyze recent news about tennis player: {player}

NEWS:
{news_text}

Return ONLY JSON:

{{
 "injury_risk": number 0..1,
 "mental_form": number 0..1,
 "motivation": number 0..1,
 "fatigue_risk": number 0..1,
 "confidence": number 0..1
}}

Rules:
- only JSON
- no explanation
- if news irrelevant → return 0.5 for all
"""


# =========================================================
# JSON PARSER
# =========================================================
def parse_json(text):

    if text is None:
        return None

    try:
        return json.loads(text)

    except:
        try:
            start = text.find("{")
            end = text.rfind("}")+1
            return json.loads(text[start:end])
        except:
            return None


# =========================================================
# MAIN FUNCTION
# =========================================================
def get_gemini_features(player_name):

    # ---------- cache ----------
    cached = load_cache(player_name)
    if cached:
        return cached

    # ---------- news ----------
    news = get_player_news(player_name)

    if not news:
        return neutral()

    # ---------- prompt ----------
    prompt = build_prompt(player_name, news)

    # ---------- request ----------
    text = ask_gemini(prompt)

    data = parse_json(text)

    if not data:
        return neutral()

    # ---------- clamp ----------
    for k in data:
        try:
            data[k] = max(0,min(1,float(data[k])))
        except:
            data[k] = 0.5

    # ---------- save cache ----------
    save_cache(player_name,data)

    time.sleep(SLEEP)

    return data


def get_pick_opinion(payload: dict) -> dict:
    """Secondary Gemini opinion for an already-produced model pick.

    Returns dict with:
      {"stance": "agree|neutral|disagree", "confidence": 0..1, "short_reason": str}
    """
    neutral_resp = {"stance": "neutral", "confidence": 0.5, "short_reason": "Gemini unavailable"}
    if not GEMINI_API_KEY:
        return neutral_resp

    prompt = f"""
You are a tennis betting assistant. Evaluate the model pick below as a SECONDARY opinion.

Return ONLY JSON:
{{
  "stance": "agree" | "neutral" | "disagree",
  "confidence": number 0..1,
  "short_reason": "max 180 chars"
}}

Model pick context:
{json.dumps(payload, ensure_ascii=False)}

Rules:
- Keep short_reason concise.
- If context is weak/unclear, return neutral with confidence around 0.5.
- No extra text, only JSON.
"""

    text = ask_gemini(prompt)
    data = parse_json(text)
    if not isinstance(data, dict):
        return neutral_resp

    stance = str(data.get("stance", "neutral")).strip().lower()
    if stance not in {"agree", "neutral", "disagree"}:
        stance = "neutral"

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except Exception:
        confidence = 0.5

    reason = str(data.get("short_reason", "")).strip() or "No extra context"
    if len(reason) > 180:
        reason = reason[:177] + "..."

    return {"stance": stance, "confidence": confidence, "short_reason": reason}


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":
    print(get_gemini_features("Carlos Alcaraz"))
