# observability.py
# Structured pipeline tracing via LangFuse (self-hosted, local Docker instance).
# All functions fail silently — if LangFuse is not running the pipeline is unaffected.
# To enable: start LangFuse via Docker (see README) and set env vars.

import os
from typing import Optional

# ---------------------------------------------------------------------------
# LangFuse client initialisation (singleton, fail-silent)
# ---------------------------------------------------------------------------

_langfuse: Optional[object] = None
_langfuse_available: bool = False


def _init_langfuse():
    """Attempt to initialise a LangFuse client. Returns None if unavailable."""
    global _langfuse, _langfuse_available
    try:
        from langfuse import Langfuse
        client = Langfuse(
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-local"),
        )
        _langfuse = client
        _langfuse_available = True
    except Exception:
        _langfuse = None
        _langfuse_available = False


# Initialise on import; fail silently if LangFuse not installed / not running
_init_langfuse()


# ---------------------------------------------------------------------------
# Public tracing API
# ---------------------------------------------------------------------------

def trace_pipeline_run(final_state: dict, duration_ms: float) -> None:
    """
    Log a complete pipeline run to LangFuse.
    Called by orchestrator.run_pipeline() at the end of every execution.
    Silently does nothing if LangFuse is unavailable.

    Logged fields:
      - Input fields (credit score, income, DTI, delinquencies, loan amount)
      - Risk score and tier
      - Borderline flag
      - Path taken (A = deterministic, B = LLM reasoning)
      - LLM status and retry count
      - Final recommendation
      - Pending review flag
      - Latency in ms
    """
    if not _langfuse_available or _langfuse is None:
        return

    try:
        validated = final_state.get("validated_input", {})
        path = "B" if final_state.get("borderline_flag") else "A"

        trace = _langfuse.trace(
            name="loan_triage_pipeline",
            metadata={
                "credit_score": validated.get("credit_score"),
                "monthly_income": validated.get("monthly_income"),
                "debt_to_income_ratio": validated.get("debt_to_income_ratio"),
                "recent_delinquencies": validated.get("recent_delinquencies"),
                "loan_amount_requested": validated.get("loan_amount_requested"),
                "risk_score": final_state.get("risk_score"),
                "risk_tier": final_state.get("risk_tier"),
                "borderline_flag": final_state.get("borderline_flag"),
                "path": path,
                "llm_status": final_state.get("llm_status"),
                "retry_count": final_state.get("retry_count", 0),
                "fallback_used": final_state.get("fallback_used", False),
                "triage_recommendation": final_state.get("triage_recommendation"),
                "pending_review": final_state.get("pending_review", False),
                "error_flag": final_state.get("error_flag", False),
                "duration_ms": round(duration_ms, 1),
            },
        )

        # Span for Mode 1 (always runs)
        _langfuse.span(
            trace_id=trace.id,
            name="mode_1_scoring",
            metadata={
                "risk_score": final_state.get("risk_score"),
                "risk_tier": final_state.get("risk_tier"),
                "borderline_flag": final_state.get("borderline_flag"),
            },
        )

        # Span for Mode 2 (only on Path B)
        if path == "B":
            _langfuse.span(
                trace_id=trace.id,
                name="mode_2_llm_reasoning",
                metadata={
                    "llm_status": final_state.get("llm_status"),
                    "retry_count": final_state.get("retry_count", 0),
                    "policy_retrieval_status": final_state.get("policy_retrieval_status"),
                },
            )

        # Span for Mode 3 (always runs)
        _langfuse.span(
            trace_id=trace.id,
            name="mode_3_output",
            metadata={
                "fallback_used": final_state.get("fallback_used", False),
                "pending_review": final_state.get("pending_review", False),
                "duration_ms": round(duration_ms, 1),
            },
        )

        _langfuse.flush()

    except Exception:
        # Never let observability failures surface to the caller
        pass


def is_langfuse_available() -> bool:
    """Return True if a LangFuse client was successfully initialised."""
    return _langfuse_available
