# tools/policy_retrieval.py
# Deterministic policy applicability gate with semantic ranking.
# Similarity search may rank policies, but it may never decide applicability.

import json
import os
import re
from typing import List, Optional, Tuple

POLICY_STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "lending_policy.json"
)


def load_policy_store() -> List[dict]:
    """Load the local policy store from JSON. Returns [] on any failure."""
    try:
        with open(POLICY_STORE_PATH, "r") as f:
            return json.load(f).get("policies", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _condition_values(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
    triage_recommendation: Optional[str],
) -> dict:
    dti = validated_input.get("debt_to_income_ratio", 0)
    return {
        "always": True,
        "borderline": borderline_flag,
        "high_dti": dti > 0.43,
        "dti_above_36": dti > 0.36,
        "dti_43_to_50": 0.43 < dti <= 0.50,
        "low_credit_score": validated_input.get("credit_score", 850) < 580,
        "recent_delinquencies": validated_input.get("recent_delinquencies", 0) >= 2,
        "high_risk_tier": risk_tier == "High",
        "moderate_or_high_risk_tier": risk_tier in ("Moderate", "High"),
        "decline_recommendation": triage_recommendation == "recommend_decline",
    }


def _is_applicable(policy: dict, values: dict) -> bool:
    """Evaluate structured policy conditions; unknown conditions fail closed."""
    rule = policy.get("applicability")
    if not rule:
        # Backward-compatible safe default: every declared condition must hold.
        conditions = policy.get("trigger_conditions", [])
        return bool(conditions) and all(values.get(name, False) for name in conditions)

    all_conditions = rule.get("all", [])
    any_conditions = rule.get("any", [])
    all_match = all(values.get(name, False) for name in all_conditions)
    any_match = not any_conditions or any(values.get(name, False) for name in any_conditions)
    return all_match and any_match


def get_applicable_policy_records(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
    triage_recommendation: Optional[str] = None,
) -> List[dict]:
    """Return only policies whose explicit, machine-readable conditions are true."""
    values = _condition_values(
        risk_tier, borderline_flag, validated_input, triage_recommendation
    )
    return [p for p in load_policy_store() if _is_applicable(p, values)]


def _build_semantic_query(
    risk_tier: str, borderline_flag: bool, validated_input: dict
) -> str:
    return " ".join([
        f"Risk tier: {risk_tier}.",
        f"Credit score: {validated_input.get('credit_score')}.",
        f"DTI ratio: {validated_input.get('debt_to_income_ratio', 0) * 100:.1f}%.",
        f"Recent delinquencies: {validated_input.get('recent_delinquencies', 0)}.",
        "Borderline case." if borderline_flag else "",
    ])


def _semantic_policy_ids(
    risk_tier: str, borderline_flag: bool, validated_input: dict
) -> List[str]:
    """Use semantic search for ordering only, never for applicability."""
    try:
        from tools.vector_store import retrieve_similar_clauses, is_vector_store_available
        if not is_vector_store_available():
            return []
        clauses = retrieve_similar_clauses(
            _build_semantic_query(risk_tier, borderline_flag, validated_input),
            n_results=10,
        )
        ids = []
        for clause in clauses:
            match = re.search(r"\bPOL-\d{3}\b", clause)
            if match and match.group(0) not in ids:
                ids.append(match.group(0))
        return ids
    except Exception:
        return []


def retrieve_policy_clauses(
    risk_tier: str,
    borderline_flag: bool,
    validated_input: dict,
    triage_recommendation: Optional[str] = None,
) -> Tuple[List[str], str]:
    """
    Retrieve applicable policies.

    Deterministic rules establish applicability. Semantic search only orders the
    already-applicable set, preventing similar-but-false policies from reaching
    the reasoning model.
    """
    applicable = get_applicable_policy_records(
        risk_tier,
        borderline_flag,
        validated_input,
        triage_recommendation,
    )
    if not applicable:
        return [], "none_found"

    semantic_order = _semantic_policy_ids(
        risk_tier, borderline_flag, validated_input
    )
    rank = {policy_id: index for index, policy_id in enumerate(semantic_order)}
    applicable.sort(key=lambda p: (rank.get(p["id"], len(rank)), p["id"]))
    clauses = [f"{p['id']}: {p['clause']}" for p in applicable]
    return clauses, "found"


def required_policy_action(policy_clauses: List[str]) -> Optional[str]:
    """Resolve mandatory policy action from the applicable clause IDs."""
    policy_by_id = {p["id"]: p for p in load_policy_store()}
    actions = []
    for clause in policy_clauses:
        match = re.match(r"(POL-\d{3}):", clause)
        if match:
            action = policy_by_id.get(match.group(1), {}).get("required_action")
            if action:
                actions.append(action)
    if "escalate_to_underwriting" in actions:
        return "escalate_to_underwriting"
    return actions[0] if actions else None


def format_policy_context(policy_clauses: List[str]) -> str:
    if not policy_clauses:
        return ""
    return "\n".join(
        ["Applicable Lending Policy Clauses:"]
        + [f"- {clause}" for clause in policy_clauses]
    )
