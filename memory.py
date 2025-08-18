# memory.py
import os, json, time
from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from upstash_redis import Redis as UpstashRedis  # if you use Upstash
import os

USE_UPSTASH = bool(os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN"))
_mem: Dict[str, Dict] = {}

def _upstash():
    return UpstashRedis(url=os.environ["UPSTASH_REDIS_REST_URL"], token=os.environ["UPSTASH_REDIS_REST_TOKEN"])

def _now(): return time.time()

def _empty():
    # default state includes role + companies
    return {"history": [], "summary": "", "state": {"role":"general","role_conf":0.0,"companies":[]}, "updated_at": _now()}

def _serialize(history: List[BaseMessage], summary: str, state: Dict):
    return json.dumps({
        "history": [{"t":"h","c":m.content} if isinstance(m, HumanMessage) else {"t":"a","c":m.content} for m in history],
        "summary": summary,
        "state": state or {"role":"general","role_conf":0.0,"companies":[]},
        "updated_at": _now()
    })

def _deserialize(raw: str):
    d = json.loads(raw)
    hist: List[BaseMessage] = []
    for m in d.get("history", []):
        hist.append(HumanMessage(m["c"]) if m.get("t") == "h" else AIMessage(m["c"]))
    state = d.get("state") or {"role":"general","role_conf":0.0,"companies":[]}
    return {"history": hist, "summary": d.get("summary",""), "state": state, "updated_at": d.get("updated_at", _now())}

def get_session(session_id: str) -> Dict:
    key = f"chat:{session_id}"
    if USE_UPSTASH:
        r = _upstash()
        raw = r.get(key)  # str or None
        if not raw:
            r.set(key, _serialize([], "", _empty()["state"]), ex=86400)
            return _empty()
        return _deserialize(raw)
    return _mem.setdefault(session_id, _empty())

def save_session(session_id: str, history: List[BaseMessage], summary: str, state: Dict):
    key = f"chat:{session_id}"
    payload = _serialize(history, summary, state)
    if USE_UPSTASH:
        _upstash().set(key, payload, ex=86400)
    else:
        _mem[session_id] = _deserialize(payload)

def append_turn(session_id: str, user_text: str, ai_text: Optional[str] = None, state: Optional[Dict] = None):
    s = get_session(session_id)
    hist: List[BaseMessage] = s["history"]
    hist.append(HumanMessage(user_text))
    if ai_text is not None:
        hist.append(AIMessage(ai_text))
    final_state = state if isinstance(state, dict) else s.get("state") or {"role":"general","role_conf":0.0,"companies":[]}
    save_session(session_id, hist, s.get("summary",""), final_state)

def maybe_refresh_summary(session_id: str, max_turns:int=12, keep_after:int=6) -> str:
    s = get_session(session_id)
    hist: List[BaseMessage] = s["history"]
    if len(hist) <= max_turns:
        return s["summary"]
    convo = []
    for m in hist[-(max_turns+keep_after):]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        convo.append(f"{role}: {m.content}")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "Summarize the conversation into concise bullets capturing goals, facts, and decisions. "
        "Avoid repetition, keep names/companies if relevant.\n\n" + "\n".join(convo) + "\n\nSummary:"
    )
    new_summary = llm.invoke(prompt).content.strip()
    pruned = hist[-keep_after:]
    save_session(session_id, pruned, new_summary, s.get("state") or {"role":"general","role_conf":0.0,"companies":[]})
    return new_summary
