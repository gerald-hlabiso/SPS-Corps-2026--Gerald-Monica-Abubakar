# evaluation/llm_judge.py
# LLM-as-Judge scorer for pipeline output quality assessment.
# Evaluates decision_explanation and policy_references for:
#   - factual consistency with the input data
#   - grounding in retrieved policy clauses
#   - reasoning clarity (1–5 scale)
#   - hallucination detection
# Uses phi4-mini via Ollama. Only called for borderline cases in batch evaluation.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pydantic import BaseModel, Field
from typing import List, Optional

from llm_client import call_llm, PRIMARY_MODEL

# ---------------------------------------------------------------------------
# Judge Output Schema
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    factual_consistency: bool = Field(
        description=(
            "True if the decision_explanation is consistent with the provided input data. "
            "False if the explanation contradicts the actual credit score, DTI, income, "
            "delinquencies, risk score, or risk tier."
        )
    )
    policy_grounding: bool = Field(
        description=(
            "True if all policy_references cited in the explanation appear verbatim "
            "in the provided policy store. False if any cited policy cannot be verified."
        )
    )
    reasoning_clarity: int = Field(
        description=(
            "Integer 1–5 rating of how clearly the explanation justifies the recommendation. "
            "5 = concise, specific, references exact input factors; "
            "1 = vague, generic, or incoherent."
        )
    )
    hallucination_detected: bool = Field(
        description=(
            "True if the explanation invents facts, policy clauses, or risk factors "
            "not present in the input or policy context. False otherwise."
        )
    )
    judge_notes: str = Field(
        description=(
            "Brief note (1–2 sentences) explaining the scores, highlighting the key "
            "strength or weakness of the explanation."
        )
    )


# ---------------------------------------------------------------------------
# Judge Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of loan triage decision explanations.
Your task is to assess the quality of an AI-generated justification for a loan triage decision.

You will be given:
1. The original loan application data (inputs)
2. The scoring engine outputs (risk score, tier, recommendation)
3. The AI-generated decision_explanation
4. The AI-cited policy_references
5. The available policy clauses from the policy store

Evaluate the explanation against four criteria and return a structured JSON assessment.
Be strict: factual inconsistencies and hallucinated policies are serious failures.
Reasoning clarity is about how well a risk associate can act on the explanation, not about style."""


def _build_judge_message(
    pipeline_output: dict,
    original_input: dict,
    policy_clauses: List[str],
) -> str:
    explanation = pipeline_output.get("decision_explanation", "None")
    references = pipeline_output.get("policy_references", [])
    policy_store_text = "\n".join(f"  - {c}" for c in policy_clauses) or "  (none)"

    return f"""ORIGINAL LOAN APPLICATION INPUT:
  Credit Score:          {original_input.get('credit_score')}
  Monthly Income:        ${original_input.get('monthly_income', 0):,.2f}
  Debt-to-Income Ratio:  {original_input.get('debt_to_income_ratio', 0) * 100:.1f}%
  Recent Delinquencies:  {original_input.get('recent_delinquencies', 0)}
  Loan Amount Requested: ${original_input.get('loan_amount_requested', 0):,.2f}

SCORING ENGINE OUTPUTS:
  Risk Score:            {pipeline_output.get('risk_score', 'N/A')}
  Risk Tier:             {pipeline_output.get('risk_tier', 'N/A')}
  Recommendation:        {pipeline_output.get('triage_recommendation', 'N/A')}
  Borderline Flag:       {pipeline_output.get('borderline_flag', False)}

AI-GENERATED DECISION EXPLANATION:
  {explanation}

AI-CITED POLICY REFERENCES:
{chr(10).join(f'  - {r}' for r in references) if references else '  (none cited)'}

AVAILABLE POLICY CLAUSES IN POLICY STORE:
{policy_store_text}

Evaluate the decision_explanation against the four criteria and return your structured assessment."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_explanation(
    pipeline_output: dict,
    original_input: dict,
    policy_clauses: Optional[List[str]] = None,
) -> Optional[JudgeScore]:
    """
    Call phi4-mini to evaluate the quality of a pipeline explanation.

    Args:
        pipeline_output: The final output dict from run_pipeline()
        original_input:  The original validated input dict (5 loan fields)
        policy_clauses:  All available policy clauses from lending_policy.json

    Returns:
        JudgeScore Pydantic object, or None if LLM call fails.
    """
    if not pipeline_output.get("decision_explanation"):
        return None  # Nothing to evaluate — LLM was skipped or fallback used

    if policy_clauses is None:
        policy_clauses = _load_policy_clauses()

    user_message = _build_judge_message(pipeline_output, original_input, policy_clauses)

    try:
        score: JudgeScore = call_llm(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=user_message,
            response_schema=JudgeScore,
            model=PRIMARY_MODEL,
            temperature=0.1,  # Low temperature for consistent scoring
        )
        return score
    except Exception as e:
        print(f"[llm_judge] Evaluation failed: {e}")
        return None


def _load_policy_clauses() -> List[str]:
    """Load all policy clause strings from lending_policy.json."""
    policy_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "lending_policy.json"
    )
    try:
        with open(policy_path, "r") as f:
            data = json.load(f)
        policies = data.get("policies", [])
        return [f"{p['id']}: {p['clause']}" for p in policies]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Aggregate Metrics
# ---------------------------------------------------------------------------

def compute_judge_metrics(scores: List[Optional[JudgeScore]]) -> dict:
    """
    Compute aggregate LLM judge metrics across a batch of evaluations.

    Args:
        scores: List of JudgeScore objects (or None if evaluation was skipped)

    Returns:
        Dict with pct_factually_consistent, pct_policy_grounded,
        pct_hallucination_detected, avg_reasoning_clarity
    """
    evaluated = [s for s in scores if s is not None]

    if not evaluated:
        return {
            "evaluated_count": 0,
            "pct_factually_consistent": None,
            "pct_policy_grounded": None,
            "pct_hallucination_detected": None,
            "avg_reasoning_clarity": None,
        }

    n = len(evaluated)
    return {
        "evaluated_count": n,
        "pct_factually_consistent": round(
            sum(1 for s in evaluated if s.factual_consistency) / n, 4
        ),
        "pct_policy_grounded": round(
            sum(1 for s in evaluated if s.policy_grounding) / n, 4
        ),
        "pct_hallucination_detected": round(
            sum(1 for s in evaluated if s.hallucination_detected) / n, 4
        ),
        "avg_reasoning_clarity": round(
            sum(s.reasoning_clarity for s in evaluated) / n, 2
        ),
    }
