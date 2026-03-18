# Project Specification
## Local Agentic Loan Triage Decision-Support Tool

---

## 1. System Name and Core Task

**System Name:** Local Agentic Loan Triage Decision-Support Tool

**Core Task:** Given a structured loan application, the system must assign a risk tier and determine whether to escalate the application to manual underwriting or recommend decline at intake, while providing a policy-aligned justification with explicit handling of borderline cases.

---

## 2. Executive Summary

Fintech lending platforms process large volumes of structured loan applications that require preliminary triage before manual underwriting. Risk operations associates must quickly assess applicant financial profiles, apply institutional risk thresholds, and determine whether an application should be escalated for full underwriting review. Borderline cases — those near risk thresholds — are particularly susceptible to inconsistent interpretation, leading to misclassification, unnecessary escalations, or overlooked risk exposure.

This project builds a local, CPU-compatible, agentic decision-support system that assists fintech risk operations associates in structured loan triage. The system operates as a conditional multi-stage pipeline: a deterministic engine validates inputs, computes risk scores, assigns risk tiers, and detects borderline cases; a local LLM agent is invoked only for borderline applications to retrieve relevant policy context and generate a structured, policy-aligned justification; and a deterministic output validator enforces schema compliance and logs all decisions locally. Non-borderline cases bypass the LLM entirely for speed and reliability. All inference and data storage occur locally to ensure offline capability and privacy compliance.

The primary performance objective is to reduce borderline-case misclassification by improving consistency in how near-threshold applications are evaluated. System effectiveness will be measured through controlled scenario testing, including agreement with a predefined decision rubric, consistency under edge-case variations, latency under CPU-only execution, and improvement over a deterministic-only baseline.

---

## 3. Business Problem Statement

Fintech platforms offering unsecured personal consumer loans process high volumes of structured applications that require preliminary triage before full manual underwriting. Risk operations associates must evaluate key financial indicators — including credit score, monthly income, debt-to-income ratio, recent delinquencies, and requested loan amount — to determine whether an application should proceed to manual underwriting or be withheld at the intake stage.

Borderline applications, particularly those near predefined risk thresholds, present a persistent operational challenge. Over-reliance on a single dominant metric, inconsistent interpretation of policy thresholds, and increasing policy complexity contribute to variability in triage decisions. Similar applicant profiles may receive different escalation outcomes depending on the reviewer, workload conditions, or subjective interpretation of lending guidelines.

Misclassification of borderline cases has measurable business consequences in both directions. If a high-risk application is wrongly recommended for decline, the manual underwriting team will never review a risky applicant in detail, increasing credit risk exposure and potential financial losses. If a high-risk application is wrongly escalated or a low-risk one is wrongly declined, risk operations associates and lending operations teams will waste time reviewing unqualified applicants or reject creditworthy customers, increasing operational costs and losing lending revenue.

The core problem addressed by this project is the need for a structured, policy-aware decision-support mechanism that standardizes preliminary triage for unsecured personal loan applications, with particular emphasis on improving consistency in borderline-case evaluation.

---

## 4. Stakeholders

### Primary User
Fintech Risk Operations Associate responsible for preliminary triage of unsecured personal loan applications in a high-volume digital lending environment. This user evaluates structured applicant data and determines whether applications should proceed to manual underwriting review or be declined at intake.

### Secondary Stakeholders
- **Manual Underwriting Team:** Receives escalated applications and benefits from improved triage consistency and reduced unnecessary workload.
- **Compliance and Risk Oversight Team:** Monitors adherence to lending policy thresholds and seeks transparent, auditable decision rationale.
- **Lending Operations Manager:** Responsible for processing efficiency, cost control, and risk exposure management.

### Indirect Stakeholders
- Applicants, who are affected by processing timelines and decision consistency.
- Executive leadership, concerned with portfolio risk and operational scalability.

---

## 5. Decision Being Supported

The system supports fintech risk operations associates in determining whether an unsecured personal consumer loan application should be escalated to manual underwriting review or recommended for decline at intake, while simultaneously assigning a structured risk tier based on predefined financial criteria.

The system does not issue final loan approvals or denials. It supports triage decisions only.

---

## 6. System Scope

### In Scope

- Structured intake of unsecured personal consumer loan application data via CLI, including:
  - Credit score (integer)
  - Monthly income (numeric)
  - Debt-to-income ratio (float)
  - Recent delinquencies (integer)
  - Loan amount requested (numeric)
- Input validation and structured error handling.
- Deterministic risk scoring based on predefined financial thresholds.
- Assignment of a structured risk tier (Low, Moderate, High).
- Borderline case detection using a predefined quantitative margin around decision thresholds.
- Conditional policy retrieval from a local policy store for borderline cases.
- LLM-generated structured explanation for borderline cases only.
- Deterministic triage recommendation (escalate to manual underwriting or recommend decline) for all cases.
- Structured JSON output and human-readable CLI summary.
- Local audit logging of all inputs, intermediate results, and final outputs.
- Fully offline, CPU-compatible operation using a locally hosted language model.
- Deterministic execution mode (no LLM) for baseline evaluation and debugging.

### Out of Scope

- Final automated loan approval or denial decisions.
- Integration with external credit bureaus, banking APIs, or live financial systems.
- Real financial deployment or execution authority.
- Model fine-tuning or supervised retraining in the initial prototype phase.
- Support for multiple loan products beyond unsecured personal consumer loans.
- Parsing of unstructured documents such as PDFs or uploaded forms.
- Cloud-based inference or external AI service dependencies.
- Multi-user or concurrent request handling.
- A graphical user interface or web application frontend.
- FastAPI layer (optional extension, out of scope for initial prototype).

---

## 7. Agent Roster

The system uses one LLM agent and two deterministic tools. An LLM agent is assigned only where genuine language model reasoning is required.

### Mode 1 — Structured Decision Execution (Deterministic Tool, No LLM)
Performs input validation, normalization, deterministic risk scoring, risk tier assignment, base triage recommendation, and borderline detection using predefined rules and thresholds. A perfect output is a fully deterministic, schema-valid intermediate state object containing validated input fields, a computed numeric risk score, an assigned risk tier strictly consistent with threshold definitions, a clearly computed borderline flag based on explicit margin logic, and a base triage recommendation that follows rule-based criteria exactly. All values must be reproducible for identical inputs with no reliance on generative reasoning.

### Mode 2 — Policy-Aware Reasoning and Justification (LLM Agent: phi4-mini, fallback: gemma3:2b)
Triggered only for borderline cases. Receives the full structured output from Mode 1 plus retrieved policy text from the orchestrator, and generates a structured, policy-aligned explanation and justification. A perfect output accurately incorporates deterministic results, explicitly references relevant retrieved policy clauses, avoids hallucinated policies, clearly cites applicable policy statements, and presents a concise, logically coherent justification suitable for audit by a risk operations associate. Mode 2 does not override deterministic scoring outputs.

### Mode 3 — Output Structuring and Verification (Deterministic Tool, No LLM)
Combines Mode 1 deterministic results and (where applicable) Mode 2 LLM explanation into a unified, schema-compliant JSON output. Enforces schema validation, handles retry on malformed LLM output, and logs the final artifact locally. A perfect output passes all validation checks without errors and is safely logged, ensuring downstream systems and human users always receive a complete, reliable, and machine-parseable result.

---

## 8. Execution Graph

### Conditional Pipeline

Every application enters through Mode 1. After scoring and borderline detection, the orchestrator routes the application along one of two paths:

**Path A — Non-Borderline Cases (majority of applications):**
```
Entry → Mode 1 (Validation + Scoring + Borderline Detection)
      → Mode 3 (Output Validation + Logging)
      → END
```
Expected latency: ~1 second.

**Path B — Borderline Cases:**
```
Entry → Mode 1 (Validation + Scoring + Borderline Detection)
      → Policy Retrieval (Orchestrator — keyword/rule-based lookup)
      → Mode 2 (LLM Reasoning + Justification, phi4-mini)
      → Mode 3 (Output Validation + Logging)
      → END
```
Expected latency: ~2–6 seconds.

### Retry Logic

Mode 2 output is validated after each attempt. If validation fails (malformed JSON, missing fields, inconsistent policy references), the orchestrator re-prompts Mode 2 with corrective instructions. Maximum retries: 2 (3 total attempts). `retry_count` is tracked in shared state and enforced as a named constant.

### Termination Conditions

| Scenario | Condition | System Returns |
|---|---|---|
| Successful termination | Valid schema-compliant JSON produced | Full JSON output |
| Retry loop | Mode 2 validation failure, retries remaining | Re-prompt Mode 2 |
| Controlled failure | Mode 2 exceeds max retries | Deterministic fallback output with `fallback_used: true` |
| Hard failure | Mode 1 error or invalid input | Error JSON, pipeline halts |

---

## 9. Shared State Schema

The `AgentState` object is the single source of truth passed between all pipeline components within a single run. Each run is fully independent — no state persists across loan application evaluations.

```python
class AgentState(TypedDict):
    # Input
    application_input: dict          # Raw JSON input
    validated_input: dict            # Cleaned and normalized fields

    # Mode 1 outputs
    risk_score: float                # Computed numeric risk score
    risk_tier: str                   # Low / Moderate / High
    borderline_flag: bool            # True if near threshold margin
    triage_recommendation: str       # escalate_to_underwriting / recommend_decline

    # Orchestrator outputs
    policy_context: str              # Retrieved policy text or empty string
    policy_retrieval_status: str     # found / none_found

    # Mode 2 outputs
    decision_explanation: Optional[str]   # LLM-generated explanation
    policy_references: List[str]          # Cited policy statements

    # Mode 3 / orchestration control
    llm_status: str                  # success / retry / failed_after_retries
    fallback_used: bool              # True if deterministic fallback triggered
    retry_count: int                 # Current retry count
    max_retries: int                 # Maximum allowed retries (constant: 2)
    error_flag: bool                 # True if pipeline error occurred
    error_stage: Optional[str]       # mode_1 / mode_2 / mode_3 / none
    error_message: Optional[str]     # Description of error
    timestamp: str                   # ISO format execution timestamp
    final_output: Optional[dict]     # Terminal output artifact
```

---

## 10. External Tools

| Tool | Owner | Type | Description |
|---|---|---|---|
| Scoring Function | Mode 1 | Deterministic code | Computes risk score from validated input fields using predefined weights and thresholds |
| Borderline Detector | Mode 1 | Deterministic code | Flags applications within a predefined quantitative margin around decision thresholds |
| Policy Retrieval | Orchestrator | Keyword/rule-based lookup | Retrieves relevant policy clauses from local JSON/CSV policy store based on triggered conditions |
| Schema Validator | Mode 3 | Deterministic code | Enforces JSON output schema compliance and triggers retry on failure |
| Audit Logger | Mode 3 | Deterministic code | Persists full pipeline artifact to local structured log file |

---

## 11. Memory and Context

### Cross-Session Memory
The system does not require cross-session memory for decision-making. Each loan application is evaluated independently. Past decisions are persisted in a structured audit log (Mode 3) for traceability and compliance. A risk operations associate retrieves prior decisions from audit logs, not from the agent.

### Within-Run Context Sharing
- **Mode 2** requires the full structured output from Mode 1 (not a summary) because exact values (risk score, thresholds, borderline flag) are needed for precise policy-aligned reasoning.
- **Mode 3** requires both Mode 1 and Mode 2 outputs to validate consistency before logging.
- All context is carried through the pipeline via the `AgentState` object.

### Domain Knowledge Availability
- **Scoring rules, tier thresholds, and borderline margins** are stored in deterministic configuration files loaded by Mode 1 at runtime. These are always available and do not depend on retrieval.
- **Policy text** is stored in a local structured repository (JSON/CSV with optional lightweight vector indexing) and retrieved dynamically by the orchestrator based on triggered conditions (borderline flag, risk tier, specific rule violations).

---

## 12. Functional Requirements

1. The system shall accept a single unsecured personal loan application as structured JSON input via a command-line interface, either as a direct input string or as a path to a local JSON file.

2. The system shall validate incoming JSON inputs for required fields (credit score, monthly income, debt-to-income ratio, recent delinquencies, loan amount requested), correct data types, and allowable value ranges.

3. The system shall return explicit, structured validation error messages when required fields are missing, malformed, or outside defined operational bounds, and shall halt the pipeline immediately.

4. The system shall compute a deterministic risk score using a predefined scoring function based on validated structured input features.

5. The system shall assign a discrete risk tier (Low, Moderate, High) based on the computed risk score and predefined tier thresholds.

6. The system shall determine a base triage recommendation of either escalate to manual underwriting or recommend decline at intake, based on the assigned risk tier and defined policy thresholds.

7. The system shall identify borderline cases using a predefined quantitative margin around escalation or decline thresholds and shall explicitly flag outputs as borderline when this condition is met.

8. The system shall route borderline cases to the policy retrieval and LLM reasoning pipeline (Path B) and shall route non-borderline cases directly to output validation and logging (Path A), bypassing LLM invocation.

9. The system shall retrieve relevant policy statements from a locally stored lending policy document based on triggered rules or threshold conditions, prior to LLM invocation.

10. The system shall generate a structured decision explanation for borderline cases that includes key input factors, computed risk score and tier, triage recommendation, and references to applicable policy statements.

11. The system shall enforce structured JSON output via schema validation after every LLM invocation, and shall re-prompt the LLM with corrective instructions on validation failure, up to a maximum of 2 retries.

12. The system shall fall back to deterministic-only output if Mode 2 exceeds the maximum retry limit, returning a valid JSON with fallback_used: true and llm_status: failed_after_retries.

13. The system shall output results in structured JSON format containing, at minimum: risk_score, risk_tier, borderline_flag, triage_recommendation, decision_explanation, policy_references, llm_status, fallback_used, error_flag, and timestamp.

14. The system shall output a human-readable summary to the CLI in addition to the structured JSON output.

15. The system shall log each evaluation locally, including input payload, validation outcomes, intermediate state, final output, and timestamp.

16. The system shall provide a deterministic execution mode that bypasses LLM reasoning entirely, to support debugging, benchmarking, and baseline evaluation.

17. The system shall refuse to produce a recommendation when inputs are incomplete, contradictory, or outside defined operational constraints, and shall return a structured error JSON specifying the error stage and message.

---

## 13. Nonfunctional Requirements

1. The system shall operate entirely offline and shall not require any external API calls or cloud-based inference services.

2. The system shall be compatible with CPU-only environments and shall not require GPU acceleration for core functionality.

3. The system shall complete a non-borderline loan triage evaluation within a target latency of approximately 1 second on a standard consumer-grade CPU environment.

4. The system shall complete a borderline loan triage evaluation (including LLM invocation) within a target latency of 2–6 seconds on a standard consumer-grade CPU environment.

5. The system shall require approximately 16 GB RAM and a modern multi-core CPU to run the primary local model (phi4-mini) via Ollama reliably.

6. The system shall store all input data, intermediate computations, and output logs locally, without transmitting or persisting data to external services.

7. The system shall maintain reproducibility such that identical input JSON payloads produce identical risk scores, risk tiers, and triage recommendations when executed in deterministic mode.

8. The system shall maintain auditability by logging all decision inputs, computed intermediate values, policy references, final outputs, status flags, and timestamps in a structured local log file.

9. The system shall clearly separate deterministic scoring logic from language model-generated explanations to ensure transparency in decision mechanics.

10. The system shall implement structured error handling and safe failure behavior, ensuring that invalid or incomplete inputs do not produce silent or undefined outputs.

11. The system shall support modular architecture such that input validation, scoring logic, policy retrieval, LLM reasoning, and output validation components are independently testable.

12. The system shall support single-user operation only; concurrent request handling is not supported in the initial prototype.

13. The system shall be documented sufficiently to allow independent reproduction of results by another developer using the same hardware class and local model configuration.

---

## 14. Failure Handling

### Mode 1 Failure (Validation or Scoring Error)
If Mode 1 fails due to malformed input, missing required fields, or a scoring exception, the pipeline terminates immediately and returns a minimal error JSON. No downstream steps are executed.
```json
{
  "error_flag": true,
  "error_stage": "mode_1",
  "error_message": "<description>",
  "timestamp": "<ISO timestamp>",
  "risk_score": null,
  "risk_tier": null,
  "triage_recommendation": null
}
```

### Policy Retrieval Failure (No Results Found)
If no relevant policy clauses are retrieved, the pipeline does not fail. Mode 2 is invoked with a fallback context and instructed to generate a justification based on scoring rules only, without citing policy.
```json
{ "policy_retrieval_status": "none_found", "policy_references": [] }
```

### Mode 2 Failure (Exceeds Maximum Retries)
If Mode 2 exceeds 2 retries, the system falls back to deterministic output.
```json
{
  "llm_status": "failed_after_retries",
  "fallback_used": true,
  "decision_explanation": null,
  "policy_references": []
}
```

### Mode 3 Failure (Validation or Logging Error)
If Mode 3 fails to validate schema or log the output, a final safeguard retry is triggered. If failure persists, a sanitized error JSON is returned with partial results where available.
```json
{
  "error_flag": true,
  "error_stage": "mode_3",
  "error_message": "<description>"
}
```

---

## 15. Operational Requirements

- **Delivery:** CLI (primary interface). FastAPI layer is optional and out of scope for the initial prototype but architecturally supported for future extension.
- **Users:** Single-user only. Operated by developers and project evaluators for testing and demonstration in the prototype phase.
- **Output format:** Structured JSON plus human-readable CLI summary on every run.
- **Hardware minimum:** 16 GB RAM, modern multi-core CPU.
- **Primary model:** phi4-mini via Ollama.
- **Fallback model:** gemma3:2b (reduced reasoning quality, maintained functionality).
- **Concurrency:** Not supported. Single-user sequential execution only.

---

## 16. Evaluation Plan

### Objectives
Measure whether the system improves consistency and reduces borderline-case misclassification in preliminary loan triage, and whether the LLM reasoning layer adds measurable value over the deterministic-only baseline.

### Dataset Strategy
A hybrid dataset constructed from:
1. A publicly available consumer loan dataset (e.g., Kaggle structured lending dataset) adapted to include the five structured input features.
2. Synthetic augmentation generating controlled borderline cases by perturbing feature values near defined risk thresholds.

### Ground Truth Definition
- **Layer 1 — Deterministic Scoring Rubric:** Predefined scoring formula assigns risk scores and tiers based on structured feature weights and thresholds.
- **Layer 2 — Borderline Handling Rules:** Explicit decision rules for applications within the predefined borderline margin.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Overall Decision Accuracy | Percentage agreement between system output and ground truth rubric |
| Borderline-Case Accuracy | Agreement rate specifically for borderline-flagged applications |
| False Escalation Rate | Percentage of cases escalated that ground truth defines as decline |
| False Decline Rate | Percentage of cases declined that ground truth defines as escalation |
| Consistency Score | Deterministic repeatability for identical inputs |
| Latency | Average execution time per application under CPU-only execution |

### Baseline Comparison
The LLM-augmented system (Path B) will be compared against the deterministic-only baseline (Path A applied to all cases). Improvement in borderline-case accuracy over the deterministic baseline measures the value added by the LLM reasoning layer.

### Test Scenario Suite
A curated scenario suite including:
- Clearly low-risk applications
- Clearly high-risk applications
- Near-threshold borderline cases
- Edge-case input errors and malformed inputs

Each scenario includes predefined expected outputs to validate system behavior and failure handling.

---

## 17. Nine-Week Implementation Timeline

### Week 1 — Finalize Specification and Evaluation Framework
- Lock business scope, feature definitions, and scoring formula.
- Define deterministic scoring rules, tier thresholds, and borderline margin.
- Identify and acquire dataset; draft synthetic case generation plan.
- Finalize `AgentState` schema and failure handling flags.
- **Deliverable:** Locked specification + dataset schema + scoring rubric.

### Week 2 — Repository Structure and Core Infrastructure
- Implement project structure and modular architecture.
- Implement CLI input handler (JSON string and file path).
- Implement validation module with structured error handling.
- Set up Ollama environment; verify phi4-mini and fallback model respond correctly.
- Create logging framework and audit log schema.
- **Deliverable:** Validated JSON intake + logging + test harness skeleton.

### Week 3 — Deterministic Scoring Engine (Mode 1)
- Implement scoring formula and risk tier assignment.
- Implement base triage recommendation logic.
- Implement borderline detection logic with configurable margin.
- Write unit tests for scoring determinism and reproducibility.
- **Deliverable:** Fully functional deterministic Mode 1 component with passing tests.

### Week 4 — Vertical Slice Prototype (End-to-End, No LLM)
- Connect validation → scoring → borderline detection → output formatter → logger (Path A).
- Implement deterministic execution mode.
- Run first evaluation on dataset subset.
- Measure baseline accuracy and borderline-case performance.
- **Deliverable:** End-to-end working triage engine without LLM reasoning.

### Week 5 — Policy Store and Retrieval Layer
- Create local lending policy document (JSON/CSV).
- Implement rule-to-policy mapping and keyword-based retrieval function.
- Integrate policy retrieval into orchestrator pipeline.
- Handle policy retrieval failure gracefully (none_found fallback).
- **Deliverable:** Deterministic pipeline with integrated policy retrieval.

### Week 6 — LLM Integration (Mode 2)
- Integrate phi4-mini via Ollama into the pipeline.
- Design and implement structured prompt template for Mode 2.
- Implement schema validation and retry logic (max 2 retries).
- Implement deterministic fallback on retry exhaustion.
- Confirm Mode 2 does not override Mode 1 scoring outputs.
- **Deliverable:** Full Path B pipeline producing structured explanations.

### Week 7 — Reliability, Guardrails, and Stability
- Stress-test borderline cases and edge-case inputs.
- Refine borderline detection margin if necessary.
- Improve explanation clarity and policy citation consistency.
- Implement and test all failure handling scenarios.
- Polish CLI human-readable output.
- **Deliverable:** Stable, reliable system with tested failure handling.

### Week 8 — Systematic Evaluation and Ablation Study
- Run full dataset evaluation across both Path A and Path B.
- Measure all defined metrics: accuracy, borderline accuracy, false rates, consistency, latency.
- Compare deterministic baseline against LLM-augmented system.
- Document failure modes and edge cases.
- **Deliverable:** Evaluation report draft + metrics tables + ablation results.

### Week 9 — Finalization, Documentation, and Demo Preparation
- Final code cleanup and modularization.
- Prepare reproducibility instructions.
- Prepare demo script with scenario walkthrough.
- Document limitations and future improvements.
- **Deliverable:** Final system, evaluation summary, and demonstration-ready artifact.
