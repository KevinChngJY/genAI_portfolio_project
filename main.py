import os, json, datetime as dt
from typing import Optional, List
import requests
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
from pathlib import Path
from upstash_redis import Redis

# Optional LLM fallback
from openai import OpenAI
from memory import get_session, append_turn, maybe_refresh_summary
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ---------- ENV ----------
DO_AGENT_BASE = os.getenv("AGENT_BASE", "").rstrip("/")    # e.g. https://.../api/v1
DO_AGENT_KEY  = os.getenv("AGENT_KEY", "")
VERIFY_TLS    = os.getenv("VERIFY_TLS", "false").lower() == "true"
CORS_ORIGINS  = os.getenv("CORS_ORIGINS", "*").split(",")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_LLM_NER = bool(OPENAI_API_KEY)

# Telegram alerts
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")  # can be negative for groups

# Redis for session memory
REDIS_URL = os.getenv("REDIS_URL", "")
USE_REDIS = bool(REDIS_URL)
# Upstash Redis REST API
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# ---------- Clients ----------
http_client = httpx.Client(verify=VERIFY_TLS, timeout=60)
llm = OpenAI(api_key=OPENAI_API_KEY) if USE_LLM_NER else None

# ---------- Company Detection ----------
COMPANY_LIST = [
    "Traveloka","Great Eastern","Great Eastern Insurance","Great Eastern Life",
    "NTUC Income","GlobalFoundries","Global Foundries","UOB","United Overseas Bank",
    "OCBC","Micron","Hyundai","Yokogawa","DXC","Singapore Pools","SGPools","SingTel",
    "Singapore Airlines","SIA","Grab","Shopee","Sea Group","Sea Ltd","Razer",
    "DBS Bank","DBS","Standard Chartered Bank","SCB","HSBC","Bank of China",
    "OMNIA","EVONIK"
]
FUZZY_CUTOFF = 80  # 0..100

MEMORY_HEADER = (
    "You are continuing an ongoing conversation. Use SUMMARY and RECENT to stay consistent and avoid repeating.\n"
    "SUMMARY:\n{summary}\n\nRECENT:\n{recent}\n\n"
)

def extract_company_fuzzy_multi(text: str, companies=COMPANY_LIST, cutoff=FUZZY_CUTOFF, limit=5) -> List[dict]:
    matches = process.extract(text, companies, scorer=fuzz.WRatio, limit=limit)
    out, seen = [], set()
    for name, score, _ in matches:
        if score >= cutoff and name not in seen:
            out.append({"name": name, "confidence": score/100.0, "method": "fuzzy"})
            seen.add(name)
    return out

def extract_company_llm(text: str) -> List[dict]:
    if not (USE_LLM_NER and llm):
        return []
    r = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":
             "Extract organisation/company names mentioned by the user. "
             "Return strict JSON: {\"companies\":[{\"name\":string,\"confidence\":number}]}. "
             "Only organisations; no people or job titles. Confidence in [0,1]. "
             "If none, return {\"companies\":[]}"
            },
            {"role":"user","content": text}
        ],
        temperature=0,
        response_format={"type":"json_object"},
        max_tokens=150,
        timeout=20
    )
    raw = (r.choices[0].message.content or "").strip()
    data = json.loads(raw)
    out = data.get("companies", [])
    for c in out: c["method"] = "llm"
    return out

def detect_companies_hybrid(text: str) -> List[dict]:
    fast = extract_company_fuzzy_multi(text)
    if fast:
        return fast
    try:
        return extract_company_llm(text)
    except Exception:
        return fast

# ---------- Telegram helpers ----------
def _mdv2_escape(s: str) -> str:
    # Telegram MarkdownV2 requires escaping these characters
    for ch in r'_\*\[\]\(\)~`>#+-=|{}.!':
        s = s.replace(ch, "\\" + ch)
    return s

def _format_telegram_message(ts_iso: str, session_id: Optional[str], companies: List[dict], msg: str, ip: Optional[str], ua: str, utm: Optional[str]) -> dict:
    # Build a concise alert using MarkdownV2
    company_lines = "\n".join([f"- {(c['name']) } ({c['confidence']:.2f}) \\- {c['method']}" for c in companies])
    text = (
        f"*Company detected* \\(Virtual Kevin\\)\n"
        f"*Time:* `{ts_iso}`\n"
        f"*Session:* `{session_id or '-'}`\n"
        #f"*UTM:* `{utm or '-'}`\n"
        #f"*IP:* `{ip or '-'}`\n"
        #f"*UA:* `{ua[:120]}`\n"
        f"*Message:*\n`{msg[:400]}`\n"
        f"*Companies:*\n{company_lines}"
    )
    return {
        "text": text
        ##"parse_mode": "MarkdownV2",
        ##"disable_web_page_preview": True,
    }

def send_telegram_alert(payload: dict):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        **payload
    }

    log_path = Path("telegram_alert.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = httpx.post(url, json=data, timeout=10)
        try:
            resp_data = resp.json()
        except Exception:
            resp_data = {"raw_text": resp.text}

        #log_entry = {
        #    "timestamp": dt.datetime.now().isoformat(),
        #    "request": data,
        #    "status_code": resp.status_code,
        #    "response": resp_data
        #}
        #with log_path.open("a", encoding="utf-8") as f:
        #    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    except Exception as e:
        # catch network / timeout errors
        log_entry = {
            "timestamp": dt.datetime.now().isoformat(),
            "request": data,
            "error": str(e),
            **payload
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            print("Failed to send Telegram alert:", log_entry)
            
def _format_recent(history: list, n:int=8) -> str:
    lines=[]
    for m in history[-n:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else "(none)"


def _append_prompt_for_companies(orig: str, companies: List[dict]) -> str:
    return (
        orig
        + "\n\n(1) Appreciate the user first."
        + "\n(2) Avoid negative phrases like 'I am not sure' or 'I cannot help you'."
        + "\n(3) Briefly tailor the response for these companies: "
        + ", ".join([f"{c['name']} ({c['confidence']:.2f})" for c in companies])
        + "\n(4) Show genuine interest in collaborating with these companies."
    )
    
# ---------- FastAPI ----------
app = FastAPI(title="Virtual Kevin – DO Agent Proxy (Hybrid detection + Telegram alerts)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    session_id: Optional[str] = None
    message: str
    utm_source: Optional[str] = None

@app.get("/health")
def health():
    return {
        "ok": True,
        "agent_base_set": bool(DO_AGENT_BASE),
        "verify_tls": VERIFY_TLS,
        "company_list_size": len(COMPANY_LIST),
        "llm_ner_enabled": USE_LLM_NER,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }
    
    
@app.get("/redis-ping-rest")
def redis_ping_rest():
    try:
        if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
            return {"ok": False, "error": "REST URL/TOKEN not set"}

        resp = requests.get(
            f"{UPSTASH_REDIS_REST_URL}/ping",
            headers={"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"},
            verify=False   # ⚠️ disables SSL check (only for testing!)
        )

        # Upstash returns JSON like {"result":"PONG"}
        data = resp.json()
        pong = (str(data.get("result", "")).upper() == "PONG")

        return {"ok": pong, "raw": data, "status": resp.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/chat")
def chat(body: ChatIn, request: Request, background_tasks: BackgroundTasks):
    if not (DO_AGENT_BASE and DO_AGENT_KEY):
        return {"error": "Server not configured with DO agent base/key."}

    # 1) detect companies first
    companies = detect_companies_hybrid(body.message)

    # 2) Telegram alert (non-blocking)
    if companies:
        ts = dt.datetime.utcnow().isoformat()
        alert_payload = _format_telegram_message(
            ts_iso=ts,
            session_id=body.session_id,
            companies=companies,
            msg=body.message,
            ip=(request.client.host if request.client else None),
            ua=request.headers.get("user-agent", ""),
            utm=body.utm_source,
        )
        log_path = Path("company_detection.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert_payload, ensure_ascii=False) + "\n")
            print("Detected companies:", companies)
        background_tasks.add_task(send_telegram_alert, alert_payload)
        # Optional: nudge downstream answer
        body.message = _append_prompt_for_companies(body.message, companies)

    
    # ---- Memory load & augment message ----
    session_id = body.session_id or "anon"
    sess = get_session(session_id)
    summary = maybe_refresh_summary(session_id) or "(none)"
    recent = _format_recent(sess["history"])
    memory_block = MEMORY_HEADER.format(summary=summary, recent=recent)

    augmented_user_msg = memory_block + "\n" + body.message
    
    
    # 3) forward to DO agent
    url = f"{DO_AGENT_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DO_AGENT_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role":"user","content": augmented_user_msg}],
        "stream": False,
        "include_retrieval_info": True
    }
    resp = http_client.post(url, headers=headers, json=payload)

    try:
        data = resp.json()
    except Exception:
        data = {"status": resp.status_code, "text": resp.text}

    answer = None
    retrieval_present = False
    if isinstance(data, dict):
        try:
            answer = data.get("choices",[{}])[0].get("message",{}).get("content")
            retrieval_present = bool((data.get("retrieval") or {}).get("contexts"))
        except Exception:
            pass
        
    # ---- Save turn to memory ----
    append_turn(session_id, body.message, answer if answer else None)

    return {
        "session_id": body.session_id,
        "companies": companies,
        "answer": answer,
        "retrieval_present": retrieval_present,
        "raw_status": resp.status_code
    }
