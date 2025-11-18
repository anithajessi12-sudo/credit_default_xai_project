import joblib
import numpy as np
import pandas as pd

def load_model(model_path="best_model.pkl"):
    model = joblib.load(model_path)
    return model

def predict_single(input_data):
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    return prediction, probability

if __name__ == "__main__":
    sample = {
        "credit_score": 720,
        "loan_amount": 12000,
        "annual_income": 45000,
        "debt_to_income_ratio": 0.25,
        "age": 32,
        "loan_term_months": 36,
        "late_payments": 0,
        "employment_status_Self-Employed": 0,
        "loan_purpose_Education": 0,
        "previous_defaults": 0
    }

    pred, prob = predict_single(sample)
    print("Predicted:", pred)
    print("Probability:", prob)
