# memory.py
import os, json, time
from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI

# ---- Config: Upstash REST or in-memory fallback ----
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")   # e.g. https://xxx.upstash.io
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
USE_UPSTASH = bool(UPSTASH_URL and UPSTASH_TOKEN)

_mem: Dict[str, Dict] = {}  # fallback (per-process, non-shared)

def _now() -> float: 
    return time.time()

def _empty() -> Dict:
    return {"history": [], "summary": "", "updated_at": _now()}

def _serialize(history: List[BaseMessage], summary: str) -> str:
    # ensure_ascii=False preserves unicode; Upstash stores strings fine
    return json.dumps({
        "history": [{"t": "h", "c": m.content} if isinstance(m, HumanMessage)
                    else {"t": "a", "c": m.content} for m in history],
        "summary": summary,
        "updated_at": _now()
    }, ensure_ascii=False)

def _deserialize(raw: str) -> Dict:
    d = json.loads(raw)
    hist: List[BaseMessage] = []
    for m in d.get("history", []):
        hist.append(HumanMessage(m["c"]) if m["t"] == "h" else AIMessage(m["c"]))
    return {"history": hist, "summary": d.get("summary", ""), "updated_at": d.get("updated_at", _now())}

# ---- Upstash client factory (lazy) ----
_upstash_client = None
def _upstash():
    global _upstash_client
    if _upstash_client is None:
        from upstash_redis import Redis
        _upstash_client = Redis(url=UPSTASH_URL, token=UPSTASH_TOKEN)
    return _upstash_client

# ---- Public API ----
def get_session(session_id: str) -> Dict:
    key = f"chat:{session_id}"
    if USE_UPSTASH:
        r = _upstash()
        raw = r.get(key)  # returns str or None
        if not raw:
            r.set(key, _serialize([], ""),ex=86400)
            return _empty()
        return _deserialize(raw)
    # fallback (per-process)
    return _mem.setdefault(session_id, _empty())

def save_session(session_id: str, history: List[BaseMessage], summary: str):
    key = f"chat:{session_id}"
    payload = _serialize(history, summary)
    if USE_UPSTASH:
        _upstash().set(key, payload,ex=86400)
    else:
        _mem[session_id] = _deserialize(payload)

def append_turn(session_id: str, user_text: str, ai_text: Optional[str] = None):
    s = get_session(session_id)
    hist: List[BaseMessage] = s["history"]
    hist.append(HumanMessage(user_text))
    if ai_text is not None:
        hist.append(AIMessage(ai_text))
    save_session(session_id, hist, s.get("summary", ""))

def maybe_refresh_summary(session_id: str, max_turns: int = 12, keep_after: int = 6) -> str:
    """
    If history longer than max_turns, produce a compact rolling summary
    with a small model, then prune to last keep_after turns.
    """
    s = get_session(session_id)
    hist: List[BaseMessage] = s["history"]
    if len(hist) <= max_turns:
        return s.get("summary", "")

    # Build recent convo text
    convo_lines = []
    for m in hist[-(max_turns + keep_after):]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        convo_lines.append(f"{role}: {m.content}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Summarize the conversation into concise bullets capturing goals, facts, and decisions. "
        "Avoid repetition; keep important names/companies if relevant.\n\n"
        + "\n".join(convo_lines)
        + "\n\nSummary:"
    )
    new_summary = llm.invoke(prompt).content.strip()

    pruned = hist[-keep_after:]
    save_session(session_id, pruned, new_summary)
    return new_summary

def clear_session(session_id: str):
    key = f"chat:{session_id}"
    if USE_UPSTASH:
        _upstash().delete(key)
    else:
        _mem.pop(session_id, None)
