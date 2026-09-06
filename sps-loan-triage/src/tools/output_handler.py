# tools/output_handler.py
# Mode 3 — Output Structuring, Validation, and Logging
# Combines Mode 1 deterministic results and Mode 2 LLM explanation into a
# validated final output artifact, and logs the complete pipeline record locally.

import json
import os
from datetime import datetime, timezone
from state import AgentState
from schemas import TriageOutput, ErrorOutput

LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "evaluation", "results"
)
LOG_FILE = os.path.join(LOG_DIR, "audit_log.jsonl")


# ---------------------------------------------------------------------------
# Output Assembly
# ---------------------------------------------------------------------------

def assemble_final_output(state: AgentState) -> dict:
    """
    Assemble and validate the final structured output artifact from AgentState.
    Returns a validated TriageOutput dict on success, or ErrorOutput dict on failure.
    """
    try:
        output = TriageOutput(
            risk_score=state["risk_score"],
            risk_tier=state["risk_tier"],
            borderline_flag=state["borderline_flag"],
            triage_recommendation=state["triage_recommendation"],
            decision_explanation=state["decision_explanation"],
            policy_references=state["policy_references"],
            llm_status=state["llm_status"],
            fallback_used=state["fallback_used"],
            pending_review=state.get("pending_review", False),
            review_reason=state.get("review_reason"),
            human_decision=state.get("human_decision"),
            error_flag=state["error_flag"],
            error_stage=state["error_stage"],
            error_message=state["error_message"],
            timestamp=state["timestamp"],
        )
        return output.model_dump()

    except Exception as e:
        return ErrorOutput(
            error_flag=True,
            error_stage="mode_3",
            error_message=f"Output assembly failed: {str(e)}",
            timestamp=state.get("timestamp", _now()),
        ).model_dump()


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

def log_pipeline_record(state: AgentState, final_output: dict) -> None:
    """
    Persist the complete pipeline record to the local audit log (JSON Lines format).
    One record per line. All fields logged for audit compliance.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_record = {
        "timestamp": state.get("timestamp", _now()),
        "application_input": state.get("application_input", {}),
        "validated_input": state.get("validated_input", {}),
        "risk_score": state.get("risk_score"),
        "risk_tier": state.get("risk_tier"),
        "borderline_flag": state.get("borderline_flag"),
        "triage_recommendation": state.get("triage_recommendation"),
        "policy_retrieval_status": state.get("policy_retrieval_status"),
        "policy_references": state.get("policy_references", []),
        "llm_status": state.get("llm_status"),
        "retry_count": state.get("retry_count"),
        "fallback_used": state.get("fallback_used"),
        "pending_review": state.get("pending_review", False),
        "review_reason": state.get("review_reason"),
        "human_decision": state.get("human_decision"),
        "error_flag": state.get("error_flag"),
        "error_stage": state.get("error_stage"),
        "error_message": state.get("error_message"),
        "final_output": final_output,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_record) + "\n")


def log_pending_review(state: AgentState, case_id: str) -> None:
    """
    Log a pending-review record to the audit log.
    Called by the API when a case is interrupted for human review.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_record = {
        "timestamp": state.get("timestamp", _now()),
        "case_id": case_id,
        "pending_review": True,
        "review_reason": state.get("review_reason"),
        "application_input": state.get("application_input", {}),
        "risk_score": state.get("risk_score"),
        "risk_tier": state.get("risk_tier"),
        "triage_recommendation": state.get("triage_recommendation"),
        "borderline_flag": state.get("borderline_flag"),
        "llm_status": state.get("llm_status"),
        "fallback_used": state.get("fallback_used"),
        "human_decision": None,
        "event": "pending_review_flagged",
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_record) + "\n")


def read_pending_reviews() -> list:
    """
    Read all pending-review records from the audit log.
    Used by GET /api/pending.
    """
    if not os.path.exists(LOG_FILE):
        return []
    results = []
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get("pending_review") and record.get("event") == "pending_review_flagged":
                        results.append(record)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return results


# ---------------------------------------------------------------------------
# Human-Readable CLI Summary
# ---------------------------------------------------------------------------

def format_cli_summary(final_output: dict) -> str:
    """Format the final output as a human-readable CLI summary."""
    if final_output.get("error_flag"):
        return (
            f"\n{'='*60}\n"
            f"  PIPELINE ERROR\n"
            f"{'='*60}\n"
            f"  Stage:   {final_output.get('error_stage', 'unknown')}\n"
            f"  Message: {final_output.get('error_message', 'unknown')}\n"
            f"{'='*60}\n"
        )

    recommendation = final_output.get("triage_recommendation", "unknown")
    tier = final_output.get("risk_tier", "unknown")
    score = final_output.get("risk_score", "N/A")
    borderline = final_output.get("borderline_flag", False)
    fallback = final_output.get("fallback_used", False)
    llm_status = final_output.get("llm_status", "skipped")
    pending = final_output.get("pending_review", False)

    lines = [
        f"\n{'='*60}",
        f"  LOAN TRIAGE RESULT",
        f"{'='*60}",
        f"  Risk Score:        {score:.1f}/100" if isinstance(score, float) else f"  Risk Score:        {score}",
        f"  Risk Tier:         {tier}",
        f"  Borderline Case:   {'Yes' if borderline else 'No'}",
        f"  Recommendation:    {recommendation.replace('_', ' ').title()}",
        f"  LLM Status:        {llm_status}",
        f"  Fallback Used:     {'Yes' if fallback else 'No'}",
        f"  Pending Review:    {'Yes — ' + final_output.get('review_reason', '') if pending else 'No'}",
    ]

    if final_output.get("decision_explanation"):
        lines.append(f"\n  Explanation:")
        lines.append(f"  {final_output['decision_explanation']}")

    if final_output.get("policy_references"):
        lines.append(f"\n  Policy References:")
        for ref in final_output["policy_references"]:
            lines.append(f"    - {ref}")

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
