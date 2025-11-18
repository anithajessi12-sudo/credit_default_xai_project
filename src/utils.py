"""
utils.py

Helper functions for the Credit Default XAI project.
- Loading data
- Preprocessing (train/test split + one-hot encoding)
- Evaluation metrics printing
"""

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET_COL = "default"


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the credit default dataset.

    Parameters
    ----------
    csv_path : str
        Relative or absolute path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Split the data into train / test and apply one-hot encoding to categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe including the target column `default`.
    test_size : float
        Fraction of rows to use as test set.

    Returns
    -------
    X_train, X_test : np.ndarray
        Preprocessed feature matrices.
    y_train, y_test : np.ndarray
        Target arrays.
    preprocessor : ColumnTransformer
        Fitted preprocessor that can be reused for predictions.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data!")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    # Detect categorical & numerical features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    # One-hot encode categoricals, pass through numeric features
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,  # keeps class balance
    )

    # Fit transformer on train only, then transform train & test
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, preprocessor


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Print classification metrics and return them as a dict.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    metrics : dict
        Dictionary containing AUC and F1.
    """
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_true, y_pred))

    print(f"\nROC-AUC : {auc:.4f}")
    print(f"F1-score: {f1:.4f}")

    return {"auc": auc, "f1": f1}


def save_object(obj, path: str) -> None:
    """
    Save a Python object with joblib.

    Parameters
    ----------
    obj : any
        Object to save.
    path : str
        File path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"Saved: {path}")


def load_object(path: str):
    """
    Load an object saved with joblib.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    obj : any
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)
