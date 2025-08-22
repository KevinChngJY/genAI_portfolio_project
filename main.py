import os, json, datetime as dt
from typing import Optional, List, Dict, Tuple, Any
import requests
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
from pathlib import Path
from upstash_redis import Redis
from flow import build_graph

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
             "Extract organisation/company names mentioned by the user. \n"
             "Make sure that it is not IT technical skills. \n"
             "Please verify the organization/company names. recognize and briefly describe the organization as a real company/institution. \n"
             "Do not include any people or job titles. \n"
             "If you are unsure or cannot describe the company, mark company_known='Not Sure' else 'Yes'. "
             "if no companies found, mark company_known='No Company'. "
             "if no companies found, mark company_known empty. "
             "Return strict JSON: {\"companies\":[{\"name\":string,\"confidence\":number}], \"company_known\":string}. "
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
    need_more_info = data.get("company_known", True)
    for c in out: c["method"] = "llm"
    return out, need_more_info

def detect_companies_hybrid(text: str) -> List[dict]:
    # fast = extract_company_fuzzy_multi(text)
    # if fast:
    #    return fast
    #try:
        #return extract_company_llm(text)
    #except Exception:
        #return fast
        
    return extract_company_llm(text)
    
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
    
# LLM valiate company:
def llm_valdate_companies(candidates: List[dict]) -> dict:
    """
    Ask the LLM if it can confidently recognize/describe each candidate company.
    Returns a dict {name: bool_known}. If LLM is off/unknown -> False.
    """
    names = [ (c.get("name") or "").strip() for c in candidates if c.get("name") ]
    if not names:
        return {}

    if not (USE_LLM_NER and llm):
        # LLM disabled -> treat as unknown
        return {n: False for n in names}

    system_msg = (
        "You verify organization names. For each name, answer if you can confidently "
        "recognize and briefly describe the organization as a real company/institution. "
        "If you are unsure or cannot describe it, mark known=false. "
        'Return strict JSON: {"results":[{"name":string,"known":boolean}]}.'
    )
    user_payload = json.dumps({"candidates": names}, ensure_ascii=False)

    r = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_payload}
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=300,
        timeout=20
    )

    raw = (r.choices[0].message.content or "").strip()
    try:
        data = json.loads(raw)
        results = data.get("results", [])
        return {
            (str(item.get("name") or "").strip()): bool(item.get("known", False))
            for item in results
        }
    except Exception:
        # Any parsing issue -> treat as unknown
        return {n: False for n in names}

# role classification ---------
ROLE_RULES = {
  "recruiter_interviewer": [
    "recruiter","headhunter","hiring","interview","technical round","system design",
    "take-home","coding test","notice period","salary","compensation","JD","resume","CV"
  ]
}

def _rule_score(text: str) -> Tuple[str, float, List[str]]:
    t = text.lower()
    hits = [kw for kw in ROLE_RULES["recruiter_interviewer"] if kw in t]
    if hits:
        # simple capped confidence
        conf = min(0.9, 0.25 + 0.15*len(hits))
        return ("recruiter_interviewer", conf, hits)
    return ("general", 0.4, [])

def llm_role_fallback(text: str, summary: str, llm) -> Tuple[str, float, List[str]]:
    # Only call if confidence still low/ambiguous
    if not llm:
        return ("general", 0.5, [])
    sys = 'Classify the user as "recruiter_interviewer" or "general".[Very strict] Classify as general unless you are confident he is recruiter_interviewe."Example Return strict JSON: {"role": "...", "confidence": 0..1, "signals": [string]}'
    usr = f'Message: "{text}"\nSummary: "{summary[-800:]}"'
    r = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
        temperature=0, response_format={"type":"json_object"}, max_tokens=120, timeout=15
    )
    try:
        data = json.loads((r.choices[0].message.content or "").strip())
        return (data.get("role","general"), float(data.get("confidence",0.5)), data.get("signals",[]))
    except Exception:
        return ("general", 0.5, [])

def classify_role(text: str, summary: str, llm=None) -> Tuple[str,float,List[str]]:
    role, conf, signals = _rule_score(text)
    if conf < 0.65:
        r2, c2, s2 = llm_role_fallback(text, summary, llm)
        if c2 > conf:
            role, conf, signals = r2, c2, s2
    return role, conf, signals

def update_role_state(sess, new_role: str, new_conf: float, alpha: float = 0.3):
    cur_role = sess["state"].get("role","general")
    cur_conf = float(sess["state"].get("role_conf",0.0))
    if new_role == cur_role:
        cur_conf = (1-alpha)*cur_conf + alpha*new_conf
    else:
        # switch only if new confidence is clearly higher
        if new_conf >= max(0.75, cur_conf + 0.15):
            cur_role, cur_conf = new_role, new_conf
        else:
            # keep old
            pass
    sess["state"]["role"] = cur_role
    sess["state"]["role_conf"] = round(cur_conf, 3)
    
def update_company_state(sess: dict, companies: list[dict]) -> None:
    """
    Remember detected companies in the session state.
    Each entry is {name: str, confidence: float}.
    If nothing valid, store [].
    """
    valid = []
    for c in companies or []:
        name = (c.get("name") or "").strip()
        if name:
            conf = float(c.get("confidence") or 0.0)
            valid.append({"name": name, "confidence": conf})

    # always ensure sess["state"] exists
    if "state" not in sess:
        sess["state"] = {}
    sess["state"]["companies"] = valid
    
ROLE_POLICIES = {
  "recruiter_interviewer": (
    "Audience: Recruiter/Interviewer. Tone: concise, professional, metric-driven. "
    "Prioritize relevant experience, availability, notice period, and highlights. "
    "Offer 1 clarifying question if needed."
  ),
  "general": (
    "Audience: General viewer. Tone: friendly, informative. "
    "Offer helpful pointers to projects and skills."
  )
}
        
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
    companies, company_is_known = detect_companies_hybrid(body.message)
    
    # 1.5) guard against companies (make sure valid company names):
    #if companies:
    #    llm_valdate_companies(companies)
        
    # 1.5) guard against companies (make sure valid/known company names via LLM):
    #needs_company_more_info = False
    #if companies:
        # Only check top 3 to keep latency sane
    #    known_map = llm_valdate_companies(companies[:3])

        # Keep only companies the LLM says it can recognize/describe
    #    companies = [c for c in companies if known_map.get((c.get("name") or "").strip(), False)]

        # If nothing left after validation → ask user for more info (no names in the question)
    #    if not companies:
    #        needs_company_more_info = True    
    
    # 2) ---- Memory load & augment message ----
    session_id = body.session_id or "anon"
    sess = get_session(session_id)
    
    # ensure state exists to avoid KeyError
    if "state" not in sess or not isinstance(sess["state"], dict):
        sess["state"] = {"role": "general", "role_conf": 0.0}
    
    summary = maybe_refresh_summary(session_id) or "(none)"
    recent = _format_recent(sess["history"])
    memory_block = MEMORY_HEADER.format(summary=summary, recent=recent)

    # NEW: classify + persist role in session state
    if sess["state"].get("role") == "recruiter_interviewer":
    # already recruiter → skip further detection
        role = "recruiter_interviewer"
        role_conf = sess["state"].get("role_conf", 0.9)
        #policy_block = ROLE_POLICIES["recruiter_interviewer"]
    else:
        role, role_conf, _signals = classify_role(
            body.message,
            summary,
            llm if USE_LLM_NER else None
        )
        update_role_state(sess, role, role_conf)
        #policy_block = ROLE_POLICIES["recruiter_interviewer"]
    print('companies:', companies)
    print('needs_company_more_info:', company_is_known)
    print(companies, company_is_known, role, role_conf)
    # 3) If LLM couldn't recognize/describe any company, instruct it to ask user for details.
    if len(companies)==0 and company_is_known=='Not Sure':
        body.message += (
            "\n\n(Important to ask)Before proceeding, ask the user—in one concise question—to provide their company's "
            "full legal name, website or LinkedIn URL, and industry/location so you can tailor the answer. "
            "Do not suggest or guess any company names; just ask for details."
        )    
    elif companies and company_is_known=='Yes':
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
        #body.message = _append_prompt_for_companies(body.message, companies)
        #persist current companies into session state
        update_company_state(sess, companies)
    
    ## langgraph flow (ONE NODE PER CALL)
    use_flow = (sess["state"]["role"] == "recruiter_interviewer" and sess["state"]["role_conf"] >= 0.7)
    scripted = ""
    if use_flow:
        graph = build_graph()
        flow_state = {
            "messages": [],
            "user_message": body.message,
            "role": sess["state"]["role"],
            "companies": companies,
            "needs_company_more_info": company_is_known,
            "summary": summary,
            "stage": sess["state"].get("flow_stage"),   # resume from last node
            "want_all_projects": False,
            "one_shot": True,                           # <-- tells router to END after this node
        }
        flow_state = graph.invoke(flow_state)
        scripted = "\n\n".join(flow_state.get("messages", []))
        # persist stage for next turn
        sess["state"]["flow_stage"] = flow_state.get("stage")

    
    #augmented_user_msg = policy_block + "\n\n" + memory_block + "\n"
    #augmented_user_msg = policy_block + "\n\n" + memory_block + "\n"
    body.message = f"(the user’s message is below here.)\\n\n{body.message}"
    if scripted:
        guided_flow = f"(Follow this outline)\\n\n{scripted}\n\n"
        augmented_user_msg = memory_block + "\n\n" + guided_flow + "\n" + body.message
    else:
        augmented_user_msg = memory_block + "\n\n" + body.message\

    
    # 4) forward to DO agent
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
        "companies": sess["state"].get("companies", []),
        "role": sess["state"].get("role", "general"),
        "answer": answer,
        "retrieval_present": retrieval_present,
        "raw_status": resp.status_code
    }
