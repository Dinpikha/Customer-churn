"""
ml/predict.py
=============
Inference layer: loads the trained model and returns churn probabilities
with risk classification and SHAP explanations.
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocessing import preprocess_single

logger = logging.getLogger(__name__)

# ── Model artefact paths ──────────────────────────────────────────────────────
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
EXPLAINER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "shap_explainer.pkl")
FEATURES_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "feature_names.pkl")

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_model     = None
_explainer = None
_features  = None


def _load_artefacts():
    global _model, _explainer, _features
    if _model is None:
        _model     = joblib.load(MODEL_PATH)
        _features  = joblib.load(FEATURES_PATH)
        try:
            _explainer = joblib.load(EXPLAINER_PATH)
        except FileNotFoundError:
            _explainer = shap.TreeExplainer(_model)
        logger.info("Model artefacts loaded.")


# ── Risk thresholds ───────────────────────────────────────────────────────────

def classify_risk(prob: float) -> str:
    """Map churn probability to a human-readable risk bucket."""
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


# ── Single-customer prediction ────────────────────────────────────────────────

def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn for one customer.

    Args:
        customer_data: Dict of raw feature values (original column names).

    Returns:
        {
          "churn_probability": float,
          "risk_level":        str,
          "shap_values":       dict[feature → shap_value],
          "feature_values":    dict[feature → scaled_value],
        }
    """
    _load_artefacts()

    X = preprocess_single(customer_data)

    prob      = float(_model.predict_proba(X)[0, 1])
    risk      = classify_risk(prob)

    # SHAP explanation for this single row
    shap_vals  = _explainer.shap_values(X)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    shap_dict    = dict(zip(_features, sv.tolist()))
    feature_vals = dict(zip(_features, X.values[0].tolist()))

    return {
        "churn_probability": round(prob, 4),
        "risk_level":        risk,
        "shap_values":       shap_dict,
        "feature_values":    feature_vals,
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a raw DataFrame (e.g., loaded from DB or CSV).

    Returns:
        Original DataFrame augmented with churn_probability and risk_level.
    """
    _load_artefacts()

    results = []
    for _, row in df_raw.iterrows():
        try:
            res = predict_churn(row.to_dict())
            results.append({
                "churn_probability": res["churn_probability"],
                "risk_level":        res["risk_level"],
            })
        except Exception as exc:
            logger.warning(f"Prediction failed for row: {exc}")
            results.append({"churn_probability": np.nan, "risk_level": "Unknown"})

    return df_raw.assign(**pd.DataFrame(results).to_dict(orient="list"))


# ── Top SHAP drivers ──────────────────────────────────────────────────────────

def get_top_shap_drivers(customer_data: dict, top_n: int = 8) -> pd.DataFrame:
    """
    Return the top-N features driving churn (positive = increases churn risk).

    Args:
        customer_data: Raw customer feature dict.
        top_n:         Number of features to return.

    Returns:
        DataFrame with columns: feature, shap_value, direction.
    """
    result = predict_churn(customer_data)
    shap_dict = result["shap_values"]

    drivers = pd.DataFrame({
        "feature":    list(shap_dict.keys()),
        "shap_value": list(shap_dict.values()),
    })
    drivers["abs_shap"]   = drivers["shap_value"].abs()
    drivers["direction"]  = drivers["shap_value"].apply(
        lambda v: "↑ Increases Risk" if v > 0 else "↓ Decreases Risk"
    )
    drivers = drivers.nlargest(top_n, "abs_shap").reset_index(drop=True)
    return drivers


def get_feature_names() -> list:
    _load_artefacts()
    return _features
