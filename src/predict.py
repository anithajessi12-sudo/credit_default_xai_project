import joblib
import pandas as pd
from utils import preprocess_single_record


def load_model(model_path="best_model.pkl"):
    """Load the trained LightGBM model"""
    return joblib.load(model_path)


def predict_single(input_data):
    """
    input_data = dictionary format
    Example:
    {
        "credit_score": 600,
        "annual_income": 450000,
        "debt_to_income_ratio": 0.4,
        "loan_amount": 250000,
        "loan_term_months": 60,
        "late_payments": 1,
        "previous_defaults": 0,
        "employment_status_Self-Employed": 0,
        "loan_purpose_Personal": 1
    }
    """
    model = load_model()

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Pre-process like training data
    processed = preprocess_single_record(df)

    # Predict probability
    proba = model.predict_proba(processed)[0]
    prediction = model.predict(processed)[0]

    return {
        "prediction": int(prediction),
        "prob_default": float(proba[1]),
        "prob_no_default": float(proba[0])
    }


if __name__ == "__main__":
    sample_input = {
        "credit_score": 500,
        "annual_income": 300000,
        "debt_to_income_ratio": 0.6,
        "loan_amount": 200000,
        "loan_term_months": 48,
        "late_payments": 2,
        "previous_defaults": 0,
        "employment_status_Self-Employed": 0,
        "loan_purpose_Personal": 1
    }

    result = predict_single(sample_input)
    print("Prediction:", result)
