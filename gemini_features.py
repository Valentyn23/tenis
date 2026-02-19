import os
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv, find_dotenv

# =========================================================
# INIT
# =========================================================
load_dotenv(find_dotenv(usecwd=True))
load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


def _env(name: str) -> str:
    return str(os.getenv(name, "")).strip()


MODEL = "gemini-2.5-flash"

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

    if not _env("NEWS_API_KEY"):
        return []

    try:
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": player_name,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": _env("NEWS_API_KEY")
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

    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        return None, "Missing GEMINI_API_KEY"

    url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    try:
        r = requests.post(url, json=body, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Gemini request failed: {exc.__class__.__name__}"

    if r.status_code != 200:
        try:
            payload = r.json()
            msg = payload.get("error", {}).get("message") or payload.get("error", {}).get("status")
        except Exception:
            msg = r.text[:160]
        return None, f"Gemini HTTP {r.status_code}: {msg or 'unknown error'}"

    try:
        payload = r.json()
    except ValueError:
        return None, "Gemini returned non-JSON response"

    candidates = payload.get("candidates") or []
    if not candidates:
        return None, "Gemini response has no candidates"

    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        return None, "Gemini response has no text parts"

    text = parts[0].get("text")
    if not text:
        return None, "Gemini response text is empty"

    return text, "ok"



def gemini_is_configured() -> tuple[bool, str]:
    if not _env("GEMINI_API_KEY"):
        return False, "Missing GEMINI_API_KEY"
    return True, "ok"



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
    text, _ = ask_gemini(prompt)

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
    configured, status_reason = gemini_is_configured()
    neutral_resp = {"stance": "neutral", "confidence": 0.5, "short_reason": status_reason}
    if not configured:
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

    text, call_reason = ask_gemini(prompt)
    if text is None:
        return {"stance": "neutral", "confidence": 0.5, "short_reason": call_reason}

    data = parse_json(text)
    if not isinstance(data, dict):
        return {"stance": "neutral", "confidence": 0.5, "short_reason": "Gemini returned non-JSON output"}

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
