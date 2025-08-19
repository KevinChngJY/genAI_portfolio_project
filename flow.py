# flow.py
from typing import Dict, Any, List, Literal, TypedDict
from langgraph.graph import StateGraph, END

# ---------- Types ----------
Stage = Literal["start", "ask_company", "showcase", "ask_more_projects", "cta", "done"]

class FlowState(TypedDict, total=False):
    # Inputs (set by caller)
    user_message: str
    role: str                                  # "recruiter_interviewer" | "general"
    companies: List[Dict[str, Any]]            # [{'name': 'UOB', 'confidence': 0.9, ...}, ...]
    needs_company_more_info: bool              # set True if your upstream company validator found none
    summary: str                               # optional: conversation summary
    want_all_projects: bool                    # optional: force full project list
    one_shot: bool                             # if True → run exactly one node this call

    # Working (mutated by nodes)
    stage: Stage
    messages: List[str]

# ---------- Nodes ----------
def start_node(state: FlowState) -> FlowState:
    """First touch: greet + clarify intent, tailored to role."""
    if state.get("role") == "recruiter_interviewer":
        state.setdefault("messages", []).append(
            "Hi! I’ll keep it crisp for recruiting. May I confirm the target role?"
        )
    else:
        state.setdefault("messages", []).append(
            "Hey! Happy to help. What role or goal are you exploring?"
        )
    state["stage"] = "start"
    return state

def ask_company_node(state: FlowState) -> FlowState:
    """Ask for company details to tailor responses."""
    state.setdefault("messages", []).append(
        "Before I tailor this, could you share your company’s full legal name and website/LinkedIn, plus industry & location?"
    )
    state["stage"] = "ask_company"
    return state

def showcase_node(state: FlowState) -> FlowState:
    """Show a compact project list; expand if asked/flagged."""
    brief = [
        "SGH AI Antibiotic Prescription Prediction — clinical decision support (PyTorch, MLflow, prod).",
        "Career Navigator — HR graph recommender + GenAI resume (Neo4j, Lucene).",
        "IT Incident Management — real-time dashboard + data pipeline (FastAPI, Airflow).",
    ]
    expanded_only = [
        "RAG Chatbot — eval & guardrails (OpenAI, evaluation harness).",
        "Predictive Maintenance — wafer tools KPI drift (XGBoost, pandas).",
        "Data Platform — ETL & MLOps (Airflow, Spark, MLflow CI/CD).",
    ]

    want_all = bool(state.get("want_all_projects"))
    # Also infer from user message (defensive)
    text = (state.get("user_message") or "").lower()
    if any(k in text for k in ["all projects", "list all projects", "full list"]):
        want_all = True

    if want_all:
        items = brief + expanded_only
        bullets = "\n- ".join(items)
        state.setdefault("messages", []).append("Here’s a fuller project list:\n- " + bullets)
    else:
        bullets = "\n- ".join(brief)
        state.setdefault("messages", []).append("A quick snapshot of relevant work:\n- " + bullets)
        state["messages"].append('Want the full list? Say "all projects" and I’ll expand.')

    state["stage"] = "showcase"
    return state

def ask_more_projects_node(state: FlowState) -> FlowState:
    """Explicit branch to full list."""
    state["want_all_projects"] = True
    return showcase_node(state)

def cta_node(state: FlowState) -> FlowState:
    """Close with a clear call-to-action for recruiters/general."""
    if state.get("role") == "recruiter_interviewer":
        state.setdefault("messages", []).append(
            "Would you like a short call or a tailored resume for the role?"
        )
    else:
        state.setdefault("messages", []).append(
            "Shall I tailor suggestions to your interests or share a resume-ready summary?"
        )
    state["stage"] = "cta"
    return state

# ---------- Router ----------
def router(state: FlowState) -> str:
    """
    Decide the next node. If one_shot=True, stop after current node (output only this step).
    """
    # EARLY EXIT: return only the current node's output this turn
    if state.get("one_shot") and state.get("stage") in (None, "start", "ask_company", "showcase", "ask_more_projects", "cta"):
        return "END"

    # Need company info first?
    if state.get("needs_company_more_info"):
        # If we haven't asked yet this turn, go ask
        if state.get("stage") not in ("ask_company",):
            return "ask_company"

    # Expand projects if requested
    text = (state.get("user_message") or "").lower()
    requested_all = any(k in text for k in ["all projects", "list all projects", "full list"])

    # Default linear path with a branch for 'all projects'
    stage = state.get("stage")
    if stage in (None, "start"):
        # If we need company info, do that before showcase
        if state.get("needs_company_more_info"):
            return "ask_company"
        # If user already asked for all projects, go directly there
        if requested_all:
            return "ask_more_projects"
        return "showcase"

    if stage == "ask_company":
        # After asking company, either go showcase or end this turn
        if requested_all:
            return "ask_more_projects"
        return "showcase"

    if stage == "showcase":
        return "cta"

    if stage == "ask_more_projects":
        return "cta"

    if stage == "cta":
        return "END"

    return "END"

# ---------- Builder ----------
def build_graph():
    g = StateGraph(FlowState)

    # Nodes
    g.add_node("start", start_node)
    g.add_node("ask_company", ask_company_node)
    g.add_node("showcase", showcase_node)
    g.add_node("ask_more_projects", ask_more_projects_node)
    g.add_node("cta", cta_node)

    # Entry
    g.set_entry_point("start")

    # Conditional edges from every node via the router
    for node in ["start", "ask_company", "showcase", "ask_more_projects", "cta"]:
        g.add_conditional_edges(node, router, {
            "ask_company": "ask_company",
            "ask_more_projects": "ask_more_projects",
            "showcase": "showcase",
            "cta": "cta",
            "END": END,
        })

    return g.compile()
