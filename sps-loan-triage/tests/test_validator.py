"""
tests/test_validator.py
Tests for Mode 1 input validation.
"""
import pytest
from tools.validator import validate_input


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

def _valid():
    return {
        "credit_score": 680,
        "monthly_income": 5000.0,
        "debt_to_income_ratio": 0.35,
        "recent_delinquencies": 0,
        "loan_amount_requested": 15000.0,
    }


def test_valid_input_returns_true():
    is_valid, data, err = validate_input(_valid())
    assert is_valid is True
    assert err == ""
    assert isinstance(data, dict)


def test_valid_input_all_fields_present():
    _, data, _ = validate_input(_valid())
    for field in ("credit_score", "monthly_income", "debt_to_income_ratio",
                  "recent_delinquencies", "loan_amount_requested"):
        assert field in data


# ---------------------------------------------------------------------------
# Missing fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "credit_score", "monthly_income", "debt_to_income_ratio",
    "recent_delinquencies", "loan_amount_requested",
])
def test_missing_required_field_fails(field):
    data = _valid()
    del data[field]
    is_valid, _, err = validate_input(data)
    assert is_valid is False
    assert err != ""


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

def test_credit_score_string_fails():
    d = _valid()
    d["credit_score"] = "not_an_int"
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_monthly_income_string_fails():
    d = _valid()
    d["monthly_income"] = "five_thousand"
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_recent_delinquencies_float_coerces_or_fails():
    # Pydantic may coerce float to int for int fields
    d = _valid()
    d["recent_delinquencies"] = 1.5
    # Either coerces to 1 (valid) or fails — both are acceptable behaviours
    is_valid, _, err = validate_input(d)
    # We don't assert a specific outcome, but the pipeline must not crash
    assert isinstance(is_valid, bool)


# ---------------------------------------------------------------------------
# Out-of-range values
# ---------------------------------------------------------------------------

def test_credit_score_below_300_fails():
    d = _valid()
    d["credit_score"] = 299
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_credit_score_above_850_fails():
    d = _valid()
    d["credit_score"] = 851
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_credit_score_300_passes():
    d = _valid()
    d["credit_score"] = 300
    is_valid, _, _ = validate_input(d)
    assert is_valid is True


def test_credit_score_850_passes():
    d = _valid()
    d["credit_score"] = 850
    is_valid, _, _ = validate_input(d)
    assert is_valid is True


def test_dti_zero_passes():
    d = _valid()
    d["debt_to_income_ratio"] = 0.0
    is_valid, _, _ = validate_input(d)
    assert is_valid is True


def test_dti_one_passes():
    d = _valid()
    d["debt_to_income_ratio"] = 1.0
    is_valid, _, _ = validate_input(d)
    assert is_valid is True


def test_dti_above_one_fails():
    d = _valid()
    d["debt_to_income_ratio"] = 1.01
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_dti_negative_fails():
    d = _valid()
    d["debt_to_income_ratio"] = -0.1
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_monthly_income_zero_fails():
    d = _valid()
    d["monthly_income"] = 0.0
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_monthly_income_negative_fails():
    d = _valid()
    d["monthly_income"] = -100.0
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_recent_delinquencies_negative_fails():
    d = _valid()
    d["recent_delinquencies"] = -1
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_loan_amount_zero_fails():
    d = _valid()
    d["loan_amount_requested"] = 0.0
    is_valid, _, err = validate_input(d)
    assert is_valid is False


def test_loan_amount_negative_fails():
    d = _valid()
    d["loan_amount_requested"] = -5000.0
    is_valid, _, err = validate_input(d)
    assert is_valid is False


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------

def test_error_message_mentions_field():
    d = _valid()
    d["credit_score"] = 900  # out of range
    is_valid, _, err = validate_input(d)
    assert is_valid is False
    assert "credit_score" in err


def test_empty_input_fails_with_message():
    is_valid, _, err = validate_input({})
    assert is_valid is False
    assert "Input validation failed" in err or len(err) > 0
