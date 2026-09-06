# tools/validator.py
# Mode 1 — Input Validation
# Validates incoming loan application JSON against the LoanApplicationInput schema.
# Returns a clean validated dict on success, or raises a structured error on failure.

from pydantic import ValidationError
from schemas import LoanApplicationInput


def validate_input(raw_input: dict) -> tuple[bool, dict, str]:
    """
    Validate a raw loan application input dict against the defined schema.

    Args:
        raw_input: Raw JSON input dict from CLI.

    Returns:
        Tuple of (is_valid: bool, validated_data: dict, error_message: str)
        - On success: (True, validated_fields_dict, "")
        - On failure: (False, {}, error_description)
    """
    try:
        validated = LoanApplicationInput(**raw_input)
        return True, validated.model_dump(), ""
    except ValidationError as e:
        # Collect all field-level errors into a readable message
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            errors.append(f"Field '{field}': {msg}")
        error_message = "Input validation failed: " + "; ".join(errors)
        return False, {}, error_message
    except Exception as e:
        return False, {}, f"Unexpected validation error: {str(e)}"
