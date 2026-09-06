# Demo Script: Local Agentic Loan Triage System

## Objective

Demonstrate how the system supports a fintech risk operations associate in triaging unsecured personal consumer loan applications, with emphasis on borderline-case handling and policy-aware reasoning.

---

## Demo Scenario Overview

We will demonstrate three application types:

1. Clearly Low-Risk Applicant
2. Clearly High-Risk Applicant
3. Borderline Applicant (Near Escalation Threshold)

---

## Scenario 1: Clearly Low-Risk Application

### Input (JSON)

{
  "credit_score": 780,
  "monthly_income": 8500,
  "debt_to_income_ratio": 0.18,
  "recent_delinquencies": 0,
  "loan_amount_requested": 10000
}

### Expected System Behavior

- Validation passes.
- Deterministic scoring produces low risk score.
- Risk tier: Low.
- Recommendation: Escalate to manual underwriting (standard processing).
- Explanation references strong credit profile and low debt ratio.
- Policy citation confirms low-risk criteria.

### Key Demonstration Points

- Deterministic scoring transparency.
- Structured output JSON.
- Logged audit record.

---

## Scenario 2: Clearly High-Risk Application

### Input (JSON)

{
  "credit_score": 540,
  "monthly_income": 2800,
  "debt_to_income_ratio": 0.62,
  "recent_delinquencies": 3,
  "loan_amount_requested": 20000
}

### Expected System Behavior

- Validation passes.
- Deterministic scoring produces high risk score.
- Risk tier: High.
- Recommendation: Recommend decline at intake.
- Explanation references high DTI and delinquencies.
- Policy citation confirms decline thresholds.

### Key Demonstration Points

- Deterministic recommendation.
- Clear policy-aligned reasoning.
- Safe handling without ambiguity.

---

## Scenario 3: Borderline Application

### Input (JSON)

{
  "credit_score": 660,
  "monthly_income": 4200,
  "debt_to_income_ratio": 0.42,
  "recent_delinquencies": 1,
  "loan_amount_requested": 15000
}

### Expected System Behavior

- Validation passes.
- Deterministic scoring produces near-threshold score.
- Risk tier: Moderate (borderline).
- Borderline flag activated.
- Policy retrieval identifies escalation guidance for near-threshold cases.
- LLM synthesizes structured explanation referencing borderline condition.
- Final recommendation: Escalate to manual underwriting due to policy-defined borderline handling rule.

### Key Demonstration Points

- Borderline detection logic.
- Policy retrieval.
- Stage 2 reasoning layer.
- Structured explanation with policy reference.
- Logged decision artifact.

---

## Closing Demonstration Summary

- Show deterministic-only baseline mode.
- Compare baseline vs hybrid decision on borderline case.
- Highlight improvement in borderline-case consistency.
- Display evaluation metrics (accuracy, latency, consistency).

The demo will conclude by emphasizing auditability, CPU-only execution, and offline operation.