# main.py
# CLI Entry Point — Local Agentic Loan Triage Decision-Support Tool
# Accepts a single loan application as JSON (string or file path) via CLI argument.
# Outputs structured JSON and a human-readable summary.

import sys
import json
import argparse

from orchestrator import run_pipeline
from tools.output_handler import format_cli_summary
from llm_client import verify_models, PRIMARY_MODEL


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Local Agentic Loan Triage Decision-Support Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pass JSON directly as a string
  python main.py --input '{"credit_score": 620, "monthly_income": 4500, "debt_to_income_ratio": 0.41, "recent_delinquencies": 1, "loan_amount_requested": 15000}'

  # Pass a path to a JSON file
  python main.py --file ./data/raw/sample_application.json

  # Run in deterministic mode only (no LLM)
  python main.py --input '{"credit_score": 720, ...}' --no-llm

  # Verify Ollama models are available
  python main.py --check-models
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Loan application as a JSON string"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a local JSON file containing the loan application"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in deterministic mode only — skip LLM reasoning for all cases"
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Verify Ollama models are available and exit"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output structured JSON only — suppress human-readable summary"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input Loading
# ---------------------------------------------------------------------------

def load_application_input(args) -> dict:
    """
    Load loan application input from CLI string or file path.
    Returns parsed dict or exits with error message.
    """
    if args.input:
        try:
            return json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string provided.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.file:
        try:
            with open(args.file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file {args.file}.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Error: Provide input via --input (JSON string) or --file (JSON file path).", file=sys.stderr)
        print("Run with --help for usage examples.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Model Check
# ---------------------------------------------------------------------------

def check_models():
    """
    Verify primary and fallback models are available in Ollama.
    Prints status and exits.
    """
    print(f"\nChecking Ollama model availability...")
    statuses = verify_models()
    for model, available in statuses.items():
        status_str = "✓ READY" if available else "✗ NOT FOUND"
        print(f"  {model}: {status_str}")

    if not statuses.get(PRIMARY_MODEL):
        print(f"\nWarning: Primary model '{PRIMARY_MODEL}' is not available.")
        print(f"Run: ollama pull {PRIMARY_MODEL}")
    print()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Handle --check-models flag
    if args.check_models:
        check_models()

    # Load application input
    application_input = load_application_input(args)

    # Inject no-llm flag into input if deterministic mode is requested
    # The orchestrator reads this flag to skip borderline LLM routing
    if args.no_llm:
        application_input["_no_llm_mode"] = True

    print(f"\n Running loan triage pipeline...\n", file=sys.stderr)

    # Run pipeline
    final_output = run_pipeline(application_input)

    # Output results
    if not args.json_only:
        print(format_cli_summary(final_output))

    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    main()
