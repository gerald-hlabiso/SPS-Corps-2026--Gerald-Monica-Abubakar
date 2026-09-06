# tools/policy_retrieval.py
# Orchestrator Tool — Policy Retrieval
# Primary path: semantic vector search via ChromaDB + Ollama embeddings.
# Fallback path: keyword/condition-based matching (always available).
# The orchestrator calls this tool; the LLM agent never calls it directly.

import json
import os
from typing import List, Tuple

POLICY_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "lending_policy.json"
)


# ---------------------------------------------------------------------------
# Keyword / condition-based retrieval (fallback, no external dependencies)
# ---------------------------------------------------------------------------

def load_policy_store() -> List[dict]:
    """Load the local policy store from JSON. Returns [] on any failure."""
    try:
        with open(POLICY_STORE_PATH, "r") as f:
            data = json.load(f)
            return data.get("policies", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _get_triggered_conditions(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
) -> List[str]:
    """Identify which policy trigger conditions apply to this application."""
    conditions = []
    if borderline_flag:
        conditions.append("borderline")
    if validated_input.get("debt_to_income_ratio", 0) > 0.43:
        conditions.append("high_dti")
    if validated_input.get("credit_score", 850) < 580:
        conditions.append("low_credit_score")
    if validated_input.get("recent_delinquencies", 0) >= 2:
        conditions.append("recent_delinquencies")
    if risk_tier == "High":
        conditions.append("high_risk_tier")
    if risk_tier == "Moderate":
        conditions.append("moderate_risk_tier")
    return conditions


def _keyword_retrieve(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
) -> Tuple[List[str], str]:
    """Keyword/condition-based policy retrieval — always available as fallback."""
    policies = load_policy_store()
    if not policies:
        return [], "none_found"

    triggered = _get_triggered_conditions(risk_tier, borderline_flag, validated_input)
    matched = []
    for p in policies:
        tier_match = risk_tier in p.get("trigger_tiers", [])
        cond_match = any(c in triggered for c in p.get("trigger_conditions", []))
        if tier_match or cond_match:
            matched.append(f"{p['id']}: {p['clause']}")

    return (matched, "found") if matched else ([], "none_found")


# ---------------------------------------------------------------------------
# Semantic retrieval (primary path via vector_store.py)
# ---------------------------------------------------------------------------

def _build_semantic_query(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
) -> str:
    """Build a natural-language query from application attributes for vector search."""
    parts = [
        f"Risk tier: {risk_tier}.",
        f"Credit score: {validated_input.get('credit_score')}.",
        f"DTI ratio: {validated_input.get('debt_to_income_ratio', 0) * 100:.1f}%.",
        f"Recent delinquencies: {validated_input.get('recent_delinquencies', 0)}.",
    ]
    if borderline_flag:
        parts.append("Borderline case near escalation threshold.")
    if validated_input.get("debt_to_income_ratio", 0) > 0.43:
        parts.append("High debt-to-income ratio requiring policy review.")
    if validated_input.get("credit_score", 850) < 580:
        parts.append("Subprime credit score below 580.")
    return " ".join(parts)


def _semantic_retrieve(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
    n_results: int = 4,
) -> Tuple[List[str], str]:
    """
    Semantic policy retrieval using the vector store.
    Returns (clauses, status) or ([], "none_found") if unavailable.
    """
    try:
        from tools.vector_store import retrieve_similar_clauses, is_vector_store_available
        if not is_vector_store_available():
            return [], "none_found"
        query = _build_semantic_query(risk_tier, borderline_flag, validated_input)
        clauses = retrieve_similar_clauses(query, n_results=n_results)
        return (clauses, "found") if clauses else ([], "none_found")
    except Exception:
        return [], "none_found"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_policy_clauses(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
) -> Tuple[List[str], str]:
    """
    Retrieve relevant policy clauses for a borderline application.

    Primary path: semantic vector search (ChromaDB + Ollama embeddings).
    Fallback:     keyword/condition-based matching from lending_policy.json.

    Returns:
        (policy_clauses: list[str], retrieval_status: "found" | "none_found")
    """
    clauses, status = _semantic_retrieve(risk_tier, borderline_flag, validated_input)
    if status == "found":
        return clauses, status

    # Keyword fallback
    return _keyword_retrieve(risk_tier, borderline_flag, validated_input)


def format_policy_context(policy_clauses: List[str]) -> str:
    """Format retrieved policy clauses into a single string for LLM prompt injection."""
    if not policy_clauses:
        return ""
    lines = ["Relevant Lending Policy Clauses:"]
    for clause in policy_clauses:
        lines.append(f"- {clause}")
    return "\n".join(lines)
