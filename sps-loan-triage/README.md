# Local Agentic Loan Triage System

A fully local, CPU-compatible, LLM-powered decision-support tool for structured loan application triage. Built with LangGraph, Ollama (phi4-mini), FastAPI, and Pydantic.

This system does **not** issue final loan approvals. It provides structured, policy-aligned triage recommendations for risk operations review.

---

## Architecture

```
Input JSON
    │
    ▼
Mode 1 — Deterministic Scoring Engine
    │                               │
    │ (non-borderline: Path A)      │ (borderline: Path B)
    │                               ▼
    │                     Policy Retrieval (semantic → keyword)
    │                               │
    │                               ▼
    │                     Mode 2 — LLM Reasoning (phi4-mini)
    │                               │
    └───────────────────────────────┤
                                    ▼
                          Review Checkpoint (HITL interrupt)
                                    │
                                    ▼
                          Mode 3 — Output Assembly + Audit Log
```

---

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull phi4-mini
ollama pull nomic-embed-text   # for semantic policy search
ollama pull gemma3:2b          # fallback model
```

### CLI Usage

```bash
cd sps-loan-triage/src

# Single application
python3 main.py --input '{"credit_score": 620, "monthly_income": 4500, "debt_to_income_ratio": 0.41, "recent_delinquencies": 1, "loan_amount_requested": 15000}'

# From file
python3 main.py --file ./data/raw/sample_application.json

# Deterministic mode only (no LLM, fastest)
python3 main.py --input '{...}' --no-llm

# Verify Ollama models are available
python3 main.py --check-models
```

### Web Server

```bash
cd sps-loan-triage/src
uvicorn api:app --host 0.0.0.0 --port 8080
# Open: http://localhost:8080
```

---

## Configuration

All scoring thresholds and weights live in [`config/scoring_config.yaml`](config/scoring_config.yaml). **No code changes needed to tune the system** — edit the YAML and restart.

```yaml
weights:
  credit_score:   0.45   # Must sum to 1.0 with the other weights
  dti_ratio:      0.30
  delinquencies:  0.25
  income_to_loan: 0.00

tiers:
  low_max:      22.0     # Scores ≤ 22 → Low
  moderate_max: 55.0     # Scores ≤ 55 → Moderate; above → High

thresholds:
  escalation:        38.0   # Scores ≤ 38 → escalate; above → decline
  borderline_margin: 10.0   # ±10 around threshold → triggers LLM reasoning

retries:
  max_retries: 1            # Network-level retries for Ollama calls
```

The loader validates at startup: weights must sum to 1.0, all thresholds must be 0–100, and tier bounds must be ordered correctly.

---

## Policy Vector Store (Semantic Search)

The system uses ChromaDB + Ollama embeddings (nomic-embed-text) for semantic policy retrieval. It falls back to keyword matching automatically if unavailable.

**Build the vector store** (run after adding/changing policy clauses):

```bash
cd sps-loan-triage
python3 evaluation/build_vector_store.py
```

The store is persisted at `data/processed/policy_vectorstore/`.

---

## Running Tests

```bash
cd sps-loan-triage
pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `tests/test_scoring.py` | Determinism, tier boundaries, borderline edges, weight sum, score range |
| `tests/test_validator.py` | All 5 required fields, type errors, out-of-range values, error messages |
| `tests/test_output_handler.py` | Output schema, fallback structure, HITL fields, error output |
| `tests/test_pipeline.py` | Path A end-to-end, Path B routing, Mode 1 failure handling |

---

## Batch Evaluation

```bash
cd sps-loan-triage

# Deterministic mode (fast)
python3 evaluation/batch_evaluate.py --rows 500

# With LLM reasoning
python3 evaluation/batch_evaluate.py --rows 500 --llm

# With LLM-as-Judge quality scoring on borderline cases
python3 evaluation/batch_evaluate.py --rows 500 --llm --judge
```

**Current performance (500 cases, deterministic):**
- Overall accuracy: 79.0%
- Borderline accuracy: 64.5%
- False escalation rate: 25.3%
- False decline rate: 19.1%
- Avg latency: <2ms

---

## LLM-as-Judge

The `evaluation/llm_judge.py` module evaluates explanation quality on a per-case basis. Metrics added to the batch report when `--judge` is enabled:

| Metric | Description |
|---|---|
| `pct_factually_consistent` | Explanation matches actual input data |
| `pct_policy_grounded` | Cited policies exist in the policy store |
| `pct_hallucination_detected` | Explanation invents facts or policy rules |
| `avg_reasoning_clarity` | 1–5 scale; how actionable the explanation is |

---

## Human-in-the-Loop (HITL)

The LangGraph pipeline includes a `review_checkpoint` node that pauses for human review when:
- The LLM fallback was triggered (reliability concern), OR
- A borderline case had an LLM failure (double uncertainty)

**API endpoints:**

```bash
# Get all cases pending human review
GET /api/pending

# Submit a human decision (approve | override_escalate | override_decline)
POST /api/review/{case_id}
Content-Type: application/json
{"human_decision": "approve", "reviewer_notes": "Verified income documentation"}
```

The web UI shows a **PENDING REVIEW** badge in the header when cases need attention, with approve/override buttons directly in the result view.

---

## Observability with LangFuse (Local)

The system traces every pipeline run to LangFuse when running in self-hosted mode.

**Setup (Docker):**

```bash
# Clone LangFuse
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d

# LangFuse UI: http://localhost:3000
# Create a project and copy the keys
```

**Configure environment variables:**

```bash
export LANGFUSE_HOST="http://localhost:3000"
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
```

If LangFuse is not running the pipeline works normally (fail-silent). Each run logs: input fields, risk score/tier, path taken (A/B), LLM status, retry count, recommendation, pending review flag, and latency.

---

## Project Structure

```
sps-loan-triage/
├── config/
│   └── scoring_config.yaml       # All tunable thresholds and weights
├── src/
│   ├── config_loader.py          # YAML config loader with validation
│   ├── state.py                  # AgentState TypedDict
│   ├── schemas.py                # Pydantic I/O schemas
│   ├── orchestrator.py           # LangGraph pipeline
│   ├── llm_client.py             # Ollama HTTP client
│   ├── observability.py          # LangFuse tracing (fail-silent)
│   ├── api.py                    # FastAPI server + HITL endpoints
│   ├── main.py                   # CLI entry point
│   ├── agent/
│   │   └── reasoning_agent.py    # Mode 2 LLM node
│   ├── tools/
│   │   ├── scoring.py            # Deterministic scoring engine
│   │   ├── validator.py          # Input validation
│   │   ├── policy_retrieval.py   # Semantic + keyword policy retrieval
│   │   ├── vector_store.py       # ChromaDB + Ollama embedding store
│   │   └── output_handler.py     # Output assembly, logging, HITL log
│   └── static/
│       └── index.html            # Web UI with HITL review panel
├── evaluation/
│   ├── batch_evaluate.py         # 500-case batch evaluator + judge metrics
│   ├── llm_judge.py              # LLM-as-Judge quality scorer
│   ├── build_vector_store.py     # Rebuild ChromaDB from policy JSON
│   └── prepare_dataset.py        # Kaggle dataset mapper
├── data/
│   └── processed/
│       ├── lending_policy.json   # 10 synthetic policy clauses
│       ├── evaluation_dataset.csv
│       └── policy_vectorstore/   # ChromaDB persistent store (auto-generated)
└── tests/
    ├── conftest.py
    ├── test_scoring.py
    ├── test_validator.py
    ├── test_output_handler.py
    └── test_pipeline.py
```
