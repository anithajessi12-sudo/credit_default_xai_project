import pandas as pd

# Columns used in training — must match exactly
EXPECTED_FEATURES = [
    "credit_score",
    "annual_income",
    "debt_to_income_ratio",
    "loan_amount",
    "loan_term_months",
    "late_payments",
    "previous_defaults",
    "employment_status_Self-Employed",
    "loan_purpose_Personal",
    "loan_purpose_Car",
    "loan_purpose_Education",
    "loan_purpose_Home"
]


def preprocess_single_record(df):
    """
    Ensures prediction input matches the format of training data.
    Adds missing one-hot columns with 0.
    """
    # Add missing columns
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Keep only needed columns (correct order)
    df = df[EXPECTED_FEATURES]

    return df
