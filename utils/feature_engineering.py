"""
utils/feature_engineering.py
=============================
Domain-driven feature engineering for the Bank Churn dataset.
All transformations are pure functions that accept and return DataFrames,
making them easy to test and reuse in both training and inference.
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

AGE_BINS   = [0, 30, 45, 60, 100]
AGE_LABELS = ["Young", "Middle-Aged", "Senior", "Elderly"]

# Weights for the composite customer_value_score
VALUE_WEIGHTS = {
    "balance_norm":    0.35,
    "tenure_norm":     0.25,
    "products_norm":   0.20,
    "active_bonus":    0.10,
    "salary_norm":     0.10,
}


# ── Individual feature functions ──────────────────────────────────────────────

def add_balance_to_salary_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    balance_to_salary_ratio — how much of a customer's salary is in their account.
    A high ratio may indicate greater financial dependence on the bank.
    """
    df = df.copy()
    df["balance_to_salary_ratio"] = np.where(
        df["EstimatedSalary"] > 0,
        df["Balance"] / df["EstimatedSalary"],
        0.0,
    )
    return df


def add_customer_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    customer_value_score — composite 0-100 score reflecting overall customer value.
    Combines balance, tenure, product count, activity status, and salary.
    """
    df = df.copy()

    # Min-max normalise each component within [0,1]
    def norm(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / rng if rng > 0 else series * 0.0

    score = (
        norm(df["Balance"])           * VALUE_WEIGHTS["balance_norm"]
        + norm(df["Tenure"])          * VALUE_WEIGHTS["tenure_norm"]
        + norm(df["NumOfProducts"])   * VALUE_WEIGHTS["products_norm"]
        + df["IsActiveMember"]        * VALUE_WEIGHTS["active_bonus"]
        + norm(df["EstimatedSalary"]) * VALUE_WEIGHTS["salary_norm"]
    )
    df["customer_value_score"] = (score * 100).round(2)
    return df


def add_inactivity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    inactivity_flag — 1 if the customer is inactive AND has zero balance.
    Captures 'ghost' accounts that are at high churn risk.
    """
    df = df.copy()
    df["inactivity_flag"] = (
        (df["IsActiveMember"] == 0) & (df["Balance"] == 0)
    ).astype(int)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    age_group — ordinal category: Young / Middle-Aged / Senior / Elderly.
    Also adds age_group_encoded (integer) for models that need numeric input.
    """
    df = df.copy()
    df["age_group"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    )
    # Ordinal encoding (preserves natural order)
    age_order = {label: i for i, label in enumerate(AGE_LABELS)}
    df["age_group_encoded"] = df["age_group"].map(age_order).astype(int)
    return df


def add_products_per_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    products_per_tenure — rate at which a customer adopted products.
    Customers with short tenure and many products may be more promiscuous.
    """
    df = df.copy()
    df["products_per_tenure"] = np.where(
        df["Tenure"] > 0,
        df["NumOfProducts"] / df["Tenure"],
        df["NumOfProducts"].astype(float),
    )
    return df


def add_high_balance_inactive(df: pd.DataFrame) -> pd.DataFrame:
    """
    high_balance_inactive — interaction feature.
    High-balance inactive customers are a paradoxical but important segment.
    """
    df = df.copy()
    balance_median = df["Balance"].median()
    df["high_balance_inactive"] = (
        (df["IsActiveMember"] == 0) & (df["Balance"] > balance_median)
    ).astype(int)
    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps in sequence.

    Args:
        df: Raw DataFrame with original column names (CreditScore, Age, …).

    Returns:
        DataFrame enriched with engineered features.
    """
    df = add_balance_to_salary_ratio(df)
    df = add_customer_value_score(df)
    df = add_inactivity_flag(df)
    df = add_age_group(df)
    df = add_products_per_tenure(df)
    df = add_high_balance_inactive(df)
    return df


# ── Convenience: list all engineered feature names ────────────────────────────

ENGINEERED_FEATURES = [
    "balance_to_salary_ratio",
    "customer_value_score",
    "inactivity_flag",
    "age_group_encoded",
    "products_per_tenure",
    "high_balance_inactive",
]
