# config_loader.py
# Loads and validates config/scoring_config.yaml at startup.
# Raises clear errors with actionable messages if values are missing or invalid.
# Returns a frozen ScoringConfig dataclass so values cannot be mutated at runtime.

import os
from typing import Optional
import yaml
from dataclasses import dataclass

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "scoring_config.yaml"
)

_WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ScoringConfig:
    weight_credit_score: float
    weight_dti_ratio: float
    weight_delinquencies: float
    weight_income_to_loan: float
    tier_low_max: float
    tier_moderate_max: float
    escalation_threshold: float
    borderline_margin: float
    max_retries: int


def _validate(cfg: ScoringConfig) -> None:
    weight_sum = (
        cfg.weight_credit_score
        + cfg.weight_dti_ratio
        + cfg.weight_delinquencies
        + cfg.weight_income_to_loan
    )
    if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"Scoring weights must sum to 1.0, got {weight_sum:.8f}. "
            "Edit config/scoring_config.yaml [weights] and recheck."
        )

    for name, val in [
        ("tiers.low_max", cfg.tier_low_max),
        ("tiers.moderate_max", cfg.tier_moderate_max),
        ("thresholds.escalation", cfg.escalation_threshold),
        ("thresholds.borderline_margin", cfg.borderline_margin),
    ]:
        if not (0.0 <= val <= 100.0):
            raise ValueError(
                f"config/{name} must be in [0, 100], got {val}."
            )

    if cfg.tier_low_max >= cfg.tier_moderate_max:
        raise ValueError(
            f"tiers.low_max ({cfg.tier_low_max}) must be less than "
            f"tiers.moderate_max ({cfg.tier_moderate_max})."
        )

    if cfg.max_retries < 0:
        raise ValueError(
            f"retries.max_retries must be >= 0, got {cfg.max_retries}."
        )


def load_config() -> ScoringConfig:
    """
    Load and validate scoring configuration from YAML.
    Raises FileNotFoundError or ValueError with a clear message on failure.
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scoring config not found at: {os.path.abspath(CONFIG_PATH)}\n"
            "Create config/scoring_config.yaml before starting the pipeline."
        )

    try:
        weights = raw["weights"]
        tiers = raw["tiers"]
        thresholds = raw["thresholds"]
        retries = raw["retries"]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"Malformed scoring_config.yaml — missing section: {e}"
        )

    cfg = ScoringConfig(
        weight_credit_score=float(weights["credit_score"]),
        weight_dti_ratio=float(weights["dti_ratio"]),
        weight_delinquencies=float(weights["delinquencies"]),
        weight_income_to_loan=float(weights["income_to_loan"]),
        tier_low_max=float(tiers["low_max"]),
        tier_moderate_max=float(tiers["moderate_max"]),
        escalation_threshold=float(thresholds["escalation"]),
        borderline_margin=float(thresholds["borderline_margin"]),
        max_retries=int(retries["max_retries"]),
    )
    _validate(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Module-level singleton — loaded once, reused everywhere.
# ---------------------------------------------------------------------------
_CONFIG: Optional[ScoringConfig] = None


def get_config() -> ScoringConfig:
    """Return the validated ScoringConfig singleton, loading it on first call."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG
