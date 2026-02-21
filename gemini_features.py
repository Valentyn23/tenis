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


DEFAULT_MODEL = "gemini-2.5-flash"
API_VERSIONS = ("v1", "v1beta")

TIMEOUT = 25
CACHE_TTL = 60 * 60 * 6   # 6 часов
SLEEP = 0.6

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def _gemini_models() -> list[str]:
    # 1. Проверяем, задана ли модель вручную в .env
    configured = _env("GEMINI_MODEL")
    if configured:
        return [configured]

    # 2. Если нет — используем самую актуальную бесплатную модель
    return ["gemini-2.5-flash"]


def _extract_http_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return payload.get("error", {}).get("message") or payload.get("error", {}).get("status") or response.text[:160]
    except Exception:
        return response.text[:160]


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

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }

    last_reason = "Gemini call not attempted"
    for model in _gemini_models():
        for version in API_VERSIONS:
            url = f"https://generativelanguage.googleapis.com/{version}/models/{model}:generateContent?key={api_key}"

            try:
                r = requests.post(url, json=body, timeout=TIMEOUT)
            except requests.RequestException as exc:
                last_reason = f"Gemini request failed: {exc.__class__.__name__}"
                continue

            if r.status_code != 200:
                msg = _extract_http_error(r) or "unknown error"
                last_reason = f"Gemini HTTP {r.status_code} [{version}/{model}]: {msg}"
                # keep trying known fallbacks for 404/400 capability mismatches
                if r.status_code in {400, 404}:
                    continue
                # for auth/quota etc. retrying other models usually won't help
                return None, last_reason

            try:
                payload = r.json()
            except ValueError:
                last_reason = f"Gemini returned non-JSON response [{version}/{model}]"
                continue

            candidates = payload.get("candidates") or []
            if not candidates:
                last_reason = f"Gemini response has no candidates [{version}/{model}]"
                continue

            parts = candidates[0].get("content", {}).get("parts") or []
            if not parts:
                last_reason = f"Gemini response has no text parts [{version}/{model}]"
                continue

            text = parts[0].get("text")
            if not text:
                last_reason = f"Gemini response text is empty [{version}/{model}]"
                continue

            return text, f"ok [{version}/{model}]"

    return None, last_reason



def gemini_is_configured() -> tuple[bool, str]:
    if not _env("GEMINI_API_KEY"):
        return False, "Missing GEMINI_API_KEY"
    model_hint = _env("GEMINI_MODEL") or DEFAULT_MODEL
    return True, f"ok (model={model_hint})"



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
