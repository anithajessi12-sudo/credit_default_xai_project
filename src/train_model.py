import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils import load_data, preprocess_data


def train_model():
    print("🔹 Loading dataset...")
    df = load_data("credit_data_jessi.csv")   # your dataset

    print("🔹 Preprocessing...")
    X, y = preprocess_data(df)

    print("🔹 Train / Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔹 Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("🔹 Predicting on test set...")
    y_pred = model.predict(X_test)

    print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("🔹 Saving model...")
    joblib.dump(model, "best_model.pkl")

    print("✅ Model training complete & saved as best_model.pkl")


if __name__ == "__main__":
    train_model()
