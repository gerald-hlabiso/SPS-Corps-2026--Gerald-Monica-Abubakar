# evaluation/prepare_dataset.py
# Maps Kaggle "Give Me Some Credit" dataset columns to the system's input schema.
# Produces a clean CSV ready for batch evaluation.
# Run this once before running batch_evaluate.py.

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cs-training.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "evaluation_dataset.csv")


# ---------------------------------------------------------------------------
# Column Mapping
# ---------------------------------------------------------------------------
# Kaggle column → System field mapping:
#
# SeriousDlqin2yrs                    → ground_truth (1=default/high-risk, 0=good)
# RevolvingUtilizationOfUnsecuredLines → debt_to_income_ratio (proxy)
# MonthlyIncome                        → monthly_income
# NumberOfTime30-59DaysPastDueNotWorse
#   + NumberOfTime60-89DaysPastDueNotWorse
#   + NumberOfTimes90DaysLate          → recent_delinquencies (sum)
# Derived from utilization + delinquencies → credit_score (proxy, 300–850)
# Fixed synthetic value                → loan_amount_requested
# ---------------------------------------------------------------------------


def derive_credit_score(row: pd.Series) -> int:
    """
    Derive a proxy credit score (300–850) from available Kaggle features.
    Higher utilization and more delinquencies → lower credit score.
    This is a heuristic proxy, not a real FICO score.
    """
    base = 750

    # Penalize for high revolving utilization
    utilization = min(row["RevolvingUtilizationOfUnsecuredLines"], 1.0)
    base -= int(utilization * 200)

    # Penalize for delinquencies
    delinquencies = (
        row["NumberOfTime30-59DaysPastDueNotWorse"] +
        row["NumberOfTime60-89DaysPastDueNotWorse"] +
        row["NumberOfTimes90DaysLate"]
    )
    base -= int(min(delinquencies, 10) * 20)

    # Penalize for high debt ratio
    debt_ratio = min(row["DebtRatio"], 1.0)
    base -= int(debt_ratio * 50)

    return max(300, min(850, base))


def derive_loan_amount(monthly_income: float) -> float:
    """
    Derive a realistic loan amount request based on monthly income.
    Assumes applicant requests ~3x monthly income (common personal loan pattern).
    """
    if pd.isna(monthly_income) or monthly_income <= 0:
        return 10000.0
    return round(monthly_income * 3, -2)  # Round to nearest 100


def prepare_dataset(
    sample_size: int = 500,
    borderline_boost: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Load, clean, map, and sample the Kaggle dataset.

    Args:
        sample_size:      Number of general cases to sample
        borderline_boost: Additional synthetic borderline cases to generate
        random_seed:      For reproducibility

    Returns:
        Clean DataFrame ready for batch evaluation
    """
    print(f"Loading dataset from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, index_col=0)
    print(f"Raw dataset: {len(df)} rows, {len(df.columns)} columns")

    # ---------------------------------------------------------------------------
    # Step 1 — Drop rows with missing critical values
    # ---------------------------------------------------------------------------
    df = df.dropna(subset=["MonthlyIncome", "SeriousDlqin2yrs"])
    print(f"After dropping missing values: {len(df)} rows")

    # ---------------------------------------------------------------------------
    # Step 2 — Clip extreme outliers
    # ---------------------------------------------------------------------------
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
    df["DebtRatio"] = df["DebtRatio"].clip(0, 1)
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(500, 50000)

    # ---------------------------------------------------------------------------
    # Step 3 — Map columns to system input fields
    # ---------------------------------------------------------------------------
    mapped = pd.DataFrame()

    mapped["credit_score"] = df.apply(derive_credit_score, axis=1)
    mapped["monthly_income"] = df["MonthlyIncome"].round(2)
    mapped["debt_to_income_ratio"] = df["RevolvingUtilizationOfUnsecuredLines"].round(4)
    mapped["recent_delinquencies"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] +
        df["NumberOfTime60-89DaysPastDueNotWorse"] +
        df["NumberOfTimes90DaysLate"]
    ).clip(0, 10).astype(int)
    mapped["loan_amount_requested"] = df["MonthlyIncome"].apply(derive_loan_amount)
    mapped["ground_truth"] = df["SeriousDlqin2yrs"].astype(int)

    # ---------------------------------------------------------------------------
    # Step 4 — Sample dataset with stratification
    # ---------------------------------------------------------------------------
    good = mapped[mapped["ground_truth"] == 0]
    bad  = mapped[mapped["ground_truth"] == 1]

    # Balance classes: 70% good, 30% bad (reflects realistic loan portfolio)
    n_good = int(sample_size * 0.70)
    n_bad  = int(sample_size * 0.30)

    sampled = pd.concat([
        good.sample(min(n_good, len(good)), random_state=random_seed),
        bad.sample(min(n_bad, len(bad)), random_state=random_seed),
    ]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"Sampled {len(sampled)} cases ({n_good} good, {n_bad} bad)")

    # ---------------------------------------------------------------------------
    # Step 5 — Add case type label for evaluation tracking
    # ---------------------------------------------------------------------------
    sampled["case_type"] = "standard"

    # ---------------------------------------------------------------------------
    # Step 6 — Save processed dataset
    # ---------------------------------------------------------------------------
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    sampled.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved processed dataset to {PROCESSED_PATH}")
    print(f"\nColumn summary:")
    print(sampled.describe().round(2))

    return sampled


if __name__ == "__main__":
    df = prepare_dataset(sample_size=500)
    print(f"\nDataset ready: {len(df)} rows")
    print(df.head())
