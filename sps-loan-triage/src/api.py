import sys
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from state import initial_state
from orchestrator import build_graph
from tools.output_handler import assemble_final_output, log_pipeline_record, log_pending_review

app = FastAPI(title="Loan Triage Decision-Support Tool")

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ---------------------------------------------------------------------------
# API graph — compiled with MemorySaver checkpointer for interrupt() support
# ---------------------------------------------------------------------------
try:
    from langgraph.checkpoint.memory import MemorySaver
    _memory = MemorySaver()
    _api_graph = build_graph(checkpointer=_memory)
except Exception:
    # Fallback: no checkpointer (HITL interrupt won't fire, but API still works)
    _api_graph = build_graph()
    _memory = None

# In-memory store of pending cases: case_id → {thread_id, state_snapshot, timestamp}
_pending_cases: dict = {}


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class LoanApplicationRequest(BaseModel):
    credit_score: int
    monthly_income: float
    debt_to_income_ratio: float
    recent_delinquencies: int
    loan_amount_requested: float


class ReviewRequest(BaseModel):
    human_decision: str  # "approve" | "override_escalate" | "override_decline"
    reviewer_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/triage")
def run_triage(request: LoanApplicationRequest):
    """
    Run the full loan triage pipeline.
    Returns the final output dict. If the case requires human review,
    the response includes pending_review=True and a case_id.
    """
    case_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": case_id}}
    application_input = request.model_dump()

    state = initial_state(application_input)

    try:
        final_state = _api_graph.invoke(state, config=config)
    except Exception as e:
        # Handle LangGraph interrupt — graph paused for human review
        error_str = str(e)
        if "interrupt" in error_str.lower() or "GraphInterrupt" in type(e).__name__:
            # Retrieve the last committed state from the checkpointer
            try:
                committed = _api_graph.get_state(config)
                partial_state = committed.values if committed else {}
            except Exception:
                partial_state = {}

            _pending_cases[case_id] = {
                "case_id": case_id,
                "thread_id": case_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "partial_state": partial_state,
                "review_reason": partial_state.get("review_reason", "Manual review required."),
            }
            log_pending_review(partial_state, case_id)

            return {
                "case_id": case_id,
                "pending_review": True,
                "review_reason": partial_state.get("review_reason", "Manual review required."),
                "risk_score": partial_state.get("risk_score"),
                "risk_tier": partial_state.get("risk_tier"),
                "triage_recommendation": partial_state.get("triage_recommendation"),
                "borderline_flag": partial_state.get("borderline_flag", False),
                "llm_status": partial_state.get("llm_status", "unknown"),
                "fallback_used": partial_state.get("fallback_used", False),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        raise HTTPException(status_code=500, detail=str(e))

    # Check if the state itself signals pending_review (CLI-mode interrupt pass-through)
    result = final_state.get("final_output", {})
    if result and result.get("pending_review"):
        _pending_cases[case_id] = {
            "case_id": case_id,
            "thread_id": case_id,
            "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "partial_state": final_state,
            "review_reason": result.get("review_reason", "Manual review required."),
        }
        result["case_id"] = case_id

    return result


@app.get("/api/pending")
def get_pending_reviews():
    """
    Return all cases currently pending human review.
    Reads from in-memory store (active session) and audit log (historical).
    """
    from tools.output_handler import read_pending_reviews
    log_pending = read_pending_reviews()

    # Merge in-memory pending cases (more up-to-date) with log-based ones
    seen_ids = set()
    combined = []

    for case in _pending_cases.values():
        cid = case.get("case_id", "")
        seen_ids.add(cid)
        combined.append({
            "case_id": cid,
            "review_reason": case.get("review_reason"),
            "timestamp": case.get("timestamp"),
            "risk_score": case.get("partial_state", {}).get("risk_score"),
            "risk_tier": case.get("partial_state", {}).get("risk_tier"),
            "triage_recommendation": case.get("partial_state", {}).get("triage_recommendation"),
            "source": "active",
        })

    for record in log_pending:
        cid = record.get("case_id", "")
        if cid not in seen_ids:
            combined.append({
                "case_id": cid,
                "review_reason": record.get("review_reason"),
                "timestamp": record.get("timestamp"),
                "risk_score": record.get("risk_score"),
                "risk_tier": record.get("risk_tier"),
                "triage_recommendation": record.get("triage_recommendation"),
                "source": "log",
            })

    return {"pending_count": len(combined), "cases": combined}


@app.post("/api/review/{case_id}")
def submit_review(case_id: str, review: ReviewRequest):
    """
    Submit a human decision for a pending review case.
    Accepts: approve | override_escalate | override_decline
    Resumes the LangGraph pipeline with the human decision if checkpointer is active.
    """
    valid_decisions = {"approve", "override_escalate", "override_decline"}
    if review.human_decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"human_decision must be one of: {valid_decisions}"
        )

    if case_id not in _pending_cases:
        raise HTTPException(
            status_code=404,
            detail=f"Case '{case_id}' not found in pending review queue."
        )

    case = _pending_cases[case_id]
    config = {"configurable": {"thread_id": case["thread_id"]}}

    try:
        from langgraph.types import Command
        final_state = _api_graph.invoke(
            Command(resume=review.human_decision),
            config=config,
        )
        result = final_state.get("final_output", {})
        if result:
            result["human_decision"] = review.human_decision
            result["reviewer_notes"] = review.reviewer_notes
        del _pending_cases[case_id]
        return result

    except Exception:
        # Checkpointer unavailable — apply human decision directly to stored state
        partial = case.get("partial_state", {})
        partial = {
            **partial,
            "human_decision": review.human_decision,
            "pending_review": False,
        }
        final_output = assemble_final_output(partial)
        final_output["human_decision"] = review.human_decision
        final_output["reviewer_notes"] = review.reviewer_notes
        log_pipeline_record(partial, final_output)
        del _pending_cases[case_id]
        return final_output


@app.get("/api/health")
def health_check():
    from observability import is_langfuse_available
    return {
        "status": "ok",
        "service": "Loan Triage Decision-Support Tool",
        "pending_reviews": len(_pending_cases),
        "langfuse": is_langfuse_available(),
    }
