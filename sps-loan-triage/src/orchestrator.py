# orchestrator.py
# LangGraph Orchestrator — wires all pipeline components into a conditional graph.
# Implements Path A (non-borderline) and Path B (borderline) routing,
# Mode 2 retry loop, deterministic fallback, HITL checkpoint, and observability hooks.

from datetime import datetime, timezone
from langgraph.graph import StateGraph, END

from state import AgentState, initial_state
from config_loader import get_config
from tools.validator import validate_input
from tools.scoring import run_scoring_engine
from tools.policy_retrieval import retrieve_policy_clauses, format_policy_context
from tools.output_handler import assemble_final_output, log_pipeline_record
from agent.reasoning_agent import reasoning_agent_node


# ---------------------------------------------------------------------------
# Node: Mode 1 — Validation and Scoring
# ---------------------------------------------------------------------------

def mode_1_node(state: AgentState) -> AgentState:
    """
    Validates input, runs deterministic scoring engine, detects borderline cases.
    Halts pipeline on validation or scoring failure.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    is_valid, validated_input, error_message = validate_input(state["application_input"])

    if not is_valid:
        return {
            **state,
            "timestamp": timestamp,
            "error_flag": True,
            "error_stage": "mode_1",
            "error_message": error_message,
        }

    try:
        scoring_results = run_scoring_engine(validated_input)
    except Exception as e:
        return {
            **state,
            "timestamp": timestamp,
            "validated_input": validated_input,
            "error_flag": True,
            "error_stage": "mode_1",
            "error_message": f"Scoring engine error: {str(e)}",
        }

    return {
        **state,
        "timestamp": timestamp,
        "validated_input": validated_input,
        "risk_score": scoring_results["risk_score"],
        "risk_tier": scoring_results["risk_tier"],
        "triage_recommendation": scoring_results["triage_recommendation"],
        "borderline_flag": scoring_results["borderline_flag"],
    }


# ---------------------------------------------------------------------------
# Node: Policy Retrieval (Orchestrator Tool)
# ---------------------------------------------------------------------------

def policy_retrieval_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant policy clauses for borderline cases (semantic → keyword fallback).
    Only called on Path B.
    """
    policy_clauses, retrieval_status = retrieve_policy_clauses(
        risk_tier=state["risk_tier"],
        borderline_flag=state["borderline_flag"],
        validated_input=state["validated_input"],
    )
    return {
        **state,
        "policy_context": format_policy_context(policy_clauses),
        "policy_retrieval_status": retrieval_status,
    }


# ---------------------------------------------------------------------------
# Node: Human-in-the-Loop Review Checkpoint
# ---------------------------------------------------------------------------

def should_flag_for_review(state: AgentState) -> bool:
    """
    Return True when the case needs human review before finalising:
    - LLM fallback was used (reliability concern), OR
    - Borderline case where LLM itself failed (double uncertainty)
    """
    if state.get("fallback_used"):
        return True
    if state.get("borderline_flag") and state.get("llm_status") == "failed_after_retries":
        return True
    return False


def _review_reason(state: AgentState) -> str:
    if state.get("fallback_used"):
        return "LLM fallback triggered — deterministic output used. Human verification recommended."
    if state.get("borderline_flag") and state.get("llm_status") == "failed_after_retries":
        return "Borderline case with LLM failure — no policy justification available."
    return "Manual review required."


def review_checkpoint_node(state: AgentState) -> AgentState:
    """
    Human-in-the-Loop checkpoint node.
    Sets pending_review=True for qualifying cases and, when running under the API
    graph (which supplies a MemorySaver checkpointer), pauses via interrupt().
    In CLI mode (no checkpointer), the flag is set and the pipeline continues so
    the pending_review status is visible in the final output.
    """
    if not should_flag_for_review(state):
        return state

    updated = {
        **state,
        "pending_review": True,
        "review_reason": _review_reason(state),
    }

    try:
        from langgraph.types import interrupt
        human_decision = interrupt(
            {
                "review_reason": updated["review_reason"],
                "risk_score": state.get("risk_score"),
                "risk_tier": state.get("risk_tier"),
                "triage_recommendation": state.get("triage_recommendation"),
                "borderline_flag": state.get("borderline_flag"),
            }
        )
        # Resumed with a human decision
        return {
            **updated,
            "human_decision": human_decision,
            "pending_review": False,
        }
    except Exception:
        # No checkpointer (CLI mode) — continue with pending_review=True recorded in output
        return updated


# ---------------------------------------------------------------------------
# Node: Mode 3 — Output Assembly and Logging
# ---------------------------------------------------------------------------

def mode_3_node(state: AgentState) -> AgentState:
    """
    Assembles final output, validates schema, logs audit record.
    Handles deterministic fallback when Mode 2 failed.
    """
    if state["llm_status"] == "failed_after_retries":
        state = {
            **state,
            "fallback_used": True,
            "decision_explanation": None,
            "policy_references": [],
        }

    final_output = assemble_final_output(state)
    log_pipeline_record(state, final_output)
    return {**state, "final_output": final_output}


# ---------------------------------------------------------------------------
# Terminal error node
# ---------------------------------------------------------------------------

def end_error_node(state: AgentState) -> AgentState:
    final_output = assemble_final_output(state)
    log_pipeline_record(state, final_output)
    return {**state, "final_output": final_output}


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------

def route_after_mode_1(state: AgentState) -> str:
    if state["error_flag"]:
        return "end_error"
    if state["application_input"].get("_no_llm_mode"):
        return "review_checkpoint"
    if state["borderline_flag"]:
        return "policy_retrieval"
    return "review_checkpoint"


def route_after_reasoning(state: AgentState) -> str:
    """
    After Mode 2: success → review_checkpoint; retry → loop back; exhausted → review_checkpoint
    (mode_3 applies fallback when llm_status == 'failed_after_retries').
    """
    cfg = get_config()
    if state["llm_status"] == "success":
        return "review_checkpoint"
    elif state["llm_status"] == "retry" and state["retry_count"] <= cfg.max_retries:
        return "reasoning_agent"
    else:
        return "review_checkpoint"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph pipeline.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver for API use).
                      Without a checkpointer the review_checkpoint node is a pass-through.

    Graph structure:
      mode_1 → [conditional]
        ├── end_error         (Mode 1 failure)
        ├── review_checkpoint → mode_3   (Path A: non-borderline)
        └── policy_retrieval → reasoning_agent → [conditional]
              ├── reasoning_agent    (network retry)
              └── review_checkpoint → mode_3  (Path B complete or fallback)
    """
    graph = StateGraph(AgentState)

    graph.add_node("mode_1", mode_1_node)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("reasoning_agent", reasoning_agent_node)
    graph.add_node("review_checkpoint", review_checkpoint_node)
    graph.add_node("mode_3", mode_3_node)
    graph.add_node("end_error", end_error_node)

    graph.set_entry_point("mode_1")

    graph.add_conditional_edges(
        "mode_1",
        route_after_mode_1,
        {
            "end_error": "end_error",
            "policy_retrieval": "policy_retrieval",
            "review_checkpoint": "review_checkpoint",
        },
    )

    graph.add_edge("policy_retrieval", "reasoning_agent")

    graph.add_conditional_edges(
        "reasoning_agent",
        route_after_reasoning,
        {
            "reasoning_agent": "reasoning_agent",
            "review_checkpoint": "review_checkpoint",
        },
    )

    graph.add_edge("review_checkpoint", "mode_3")
    graph.add_edge("mode_3", END)
    graph.add_edge("end_error", END)

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# CLI graph — no checkpointer; review_checkpoint passes through without interrupt()
_GRAPH = build_graph()


# ---------------------------------------------------------------------------
# Public Run Function (CLI)
# ---------------------------------------------------------------------------

def run_pipeline(application_input: dict) -> dict:
    """Execute the full loan triage pipeline for a single application (CLI)."""
    import time
    from observability import trace_pipeline_run

    state = initial_state(application_input)
    start = time.time()
    final_state = _GRAPH.invoke(state)
    duration_ms = (time.time() - start) * 1000

    trace_pipeline_run(final_state, duration_ms)
    return final_state["final_output"]
