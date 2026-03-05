"""
utils/preprocessing.py
=======================
End-to-end data cleaning, encoding, and scaling pipeline.
Designed so the same transformations can be applied identically
to training data and new inference rows.

ROOT CAUSE FIX
--------------
The original design used module-level LabelEncoder / StandardScaler globals
that were fitted in-process during training but were never persisted.
When the Streamlit app (a separate process) called preprocess_single(), it
imported fresh, unfitted instances → NotFittedError.

Solution: save_preprocessors() / load_preprocessors() explicitly persist the
fitted transformers to models/preprocessors.pkl after training, and
preprocess_single() loads them from disk before transforming.
"""

import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .feature_engineering import engineer_features, ENGINEERED_FEATURES

logger = logging.getLogger(__name__)

# ── Artefact path ─────────────────────────────────────────────────────────────
# Resolved relative to this file so it works regardless of cwd.
_MODELS_DIR        = os.path.join(os.path.dirname(__file__), "..", "models")
PREPROCESSORS_PATH = os.path.join(_MODELS_DIR, "preprocessors.pkl")


# ── Column groups ─────────────────────────────────────────────────────────────

RAW_NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCreditCard", "IsActiveMember", "EstimatedSalary",
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]
TARGET_COL           = "Exited"

MODEL_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES + [
    "Geography_encoded", "Gender_encoded",
]

SCALE_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
    "balance_to_salary_ratio", "customer_value_score", "products_per_tenure",
]


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop junk columns, deduplicate, impute, and clamp ranges."""
    df = df.copy()

    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.info(f"Dropped identifier columns: {drop_cols}")

    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows.")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    df["CreditScore"]     = df["CreditScore"].clip(300, 900)
    df["Age"]             = df["Age"].clip(18, 100)
    df["Tenure"]          = df["Tenure"].clip(0, 10)
    df["NumOfProducts"]   = df["NumOfProducts"].clip(1, 4)
    df["Balance"]         = df["Balance"].clip(lower=0)
    df["EstimatedSalary"] = df["EstimatedSalary"].clip(lower=0)

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df


# ── Persist / restore fitted transformers ────────────────────────────────────

def save_preprocessors(geo_enc: LabelEncoder, gender_enc: LabelEncoder,
                        scaler: StandardScaler) -> None:
    """
    Persist the three fitted transformers to disk so that inference
    processes (Streamlit, FastAPI, etc.) can reload them without needing
    to re-run training.  Called once at the end of train_model.py.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)
    bundle = {
        "geo_encoder":    geo_enc,
        "gender_encoder": gender_enc,
        "scaler":         scaler,
    }
    joblib.dump(bundle, PREPROCESSORS_PATH)
    logger.info(f"Preprocessors saved → {PREPROCESSORS_PATH}")


def load_preprocessors() -> Tuple[LabelEncoder, LabelEncoder, StandardScaler]:
    """
    Load fitted transformers from disk.
    Raises FileNotFoundError with a helpful message if the model has not
    been trained yet.
    """
    if not os.path.exists(PREPROCESSORS_PATH):
        raise FileNotFoundError(
            f"Preprocessors not found at '{PREPROCESSORS_PATH}'.\n"
            "Run training first:  python -m ml.train_model --data data/bank_churn.csv"
        )
    bundle = joblib.load(PREPROCESSORS_PATH)
    return bundle["geo_encoder"], bundle["gender_encoder"], bundle["scaler"]


# ── Encoding (training-time: fit=True) ───────────────────────────────────────

def encode_categoricals(df: pd.DataFrame,
                         fit: bool = True,
                         geo_enc:    LabelEncoder = None,
                         gender_enc: LabelEncoder = None) -> Tuple[pd.DataFrame,
                                                                     LabelEncoder,
                                                                     LabelEncoder]:
    """
    Label-encode Geography and Gender.

    Args:
        df:         DataFrame with Geography and Gender columns.
        fit:        If True, create and fit new encoders (training).
                    If False, use the provided geo_enc / gender_enc (inference).
        geo_enc:    Pre-fitted Geography encoder (required when fit=False).
        gender_enc: Pre-fitted Gender encoder (required when fit=False).

    Returns:
        (transformed_df, geo_encoder, gender_encoder)
        — the encoders are returned so callers can persist them.
    """
    df = df.copy()

    if fit:
        geo_enc    = LabelEncoder()
        gender_enc = LabelEncoder()
        df["Geography_encoded"] = geo_enc.fit_transform(df["Geography"])
        df["Gender_encoded"]    = gender_enc.fit_transform(df["Gender"])
    else:
        if geo_enc is None or gender_enc is None:
            raise ValueError("geo_enc and gender_enc must be supplied when fit=False.")
        df["Geography_encoded"] = geo_enc.transform(df["Geography"])
        df["Gender_encoded"]    = gender_enc.transform(df["Gender"])

    return df, geo_enc, gender_enc


# ── Scaling (training-time: fit=True) ────────────────────────────────────────

def scale_features(df: pd.DataFrame,
                   fit: bool = True,
                   scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale continuous features.

    Returns:
        (transformed_df, scaler)
    """
    df           = df.copy()
    cols_present = [c for c in SCALE_FEATURES if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        if scaler is None:
            raise ValueError("scaler must be supplied when fit=False.")
        df[cols_present] = scaler.transform(df[cols_present])

    return df, scaler


# ── Master training pipeline ──────────────────────────────────────────────────

def preprocess_train(df: pd.DataFrame, test_size: float = 0.2,
                     random_state: int = 42) -> Tuple:
    """
    Full training preprocessing pipeline.
    Fits all transformers, saves them to disk, and returns splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = clean_data(df)
    df = engineer_features(df)

    df, geo_enc, gender_enc = encode_categoricals(df, fit=True)
    df, scaler              = scale_features(df, fit=True)

    # ── Persist fitted transformers immediately ───────────────────────────────
    save_preprocessors(geo_enc, gender_enc, scaler)

    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Train size: {len(X_train)} | Test size: {len(X_test)} | "
        f"Churn rate: {y.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ── Inference pipeline ────────────────────────────────────────────────────────

def preprocess_single(input_dict: dict) -> pd.DataFrame:
    """
    Preprocess a single customer dict for real-time inference.
    Loads persisted transformers from disk — no in-memory state required.

    Args:
        input_dict: Keys match original column names (e.g. 'CreditScore').

    Returns:
        Single-row DataFrame ready for model.predict_proba().
    """
    # Load the transformers that were saved during training
    geo_enc, gender_enc, scaler = load_preprocessors()

    df = pd.DataFrame([input_dict])
    df = clean_data(df)
    df = engineer_features(df)

    df, _, _ = encode_categoricals(df, fit=False, geo_enc=geo_enc, gender_enc=gender_enc)
    df, _    = scale_features(df, fit=False, scaler=scaler)

    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    return df[feature_cols]


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_geo_classes() -> list:
    geo_enc, _, _ = load_preprocessors()
    return list(geo_enc.classes_)


def get_gender_classes() -> list:
    _, gender_enc, _ = load_preprocessors()
    return list(gender_enc.classes_)
