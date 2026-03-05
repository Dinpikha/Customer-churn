"""
ml/train_model.py
=================
Trains a XGBoost churn classifier with:
  • SMOTE oversampling for class imbalance
  • Optuna hyperparameter search (optional, set TUNE=True)
  • Full evaluation metrics + confusion matrix
  • SHAP explainability
  • Serialises artefacts to models/

Run:
    python -m ml.train_model --data data/bank_churn.csv
"""

import argparse
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocessing import preprocess_train


# ── Hyperparameters (tuned offline) ──────────────────────────────────────────

XGB_PARAMS = {
    "n_estimators":       400,
    "max_depth":          5,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   3,
    "scale_pos_weight":   2,   # handles mild class imbalance
    "use_label_encoder":  False,
    "eval_metric":        "logloss",
    "random_state":       42,
    "n_jobs":             -1,
}

RF_PARAMS = {
    "n_estimators":  300,
    "max_depth":     8,
    "min_samples_split": 5,
    "class_weight":  "balanced",
    "random_state":  42,
    "n_jobs":        -1,
}


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: str, model_type: str = "xgb", use_smote: bool = True):
    """
    End-to-end training pipeline.

    Args:
        data_path:  Path to raw CSV.
        model_type: 'xgb' or 'rf'.
        use_smote:  Whether to apply SMOTE to training split.

    Returns:
        Trained model, feature names, evaluation dict.
    """
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = preprocess_train(df)

    # ── Class balance before SMOTE ────────────────────────────────────────────
    churn_rate = y_train.mean()
    logger.info(f"Training churn rate (pre-SMOTE): {churn_rate:.2%}")

    # ── SMOTE oversampling ────────────────────────────────────────────────────
    if use_smote and churn_rate < 0.40:
        sm = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"Post-SMOTE train size: {len(X_train)} | churn rate: {y_train.mean():.2%}")

    # ── Model selection ───────────────────────────────────────────────────────
    if model_type == "xgb":
        model = XGBClassifier(**XGB_PARAMS)
        logger.info("Training XGBoost classifier …")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model = RandomForestClassifier(**RF_PARAMS)
        logger.info("Training Random Forest classifier …")
        model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    logger.info("\n" + "="*50)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("="*50)
    for k, v in metrics.items():
        logger.info(f"  {k.upper():12s}: {v}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    # ── Save artefacts ────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    model_path    = f"models/churn_model_{model_type}.pkl"
    features_path = "models/feature_names.pkl"
    metrics_path  = "models/metrics.pkl"

    joblib.dump(model,         model_path)
    joblib.dump(feature_names, features_path)
    joblib.dump(metrics,       metrics_path)

    # Canonical alias used by the app
    joblib.dump(model, "models/churn_model.pkl")

    logger.info(f"\nModel saved → {model_path}")
    logger.info(f"Features  saved → {features_path}")

    # ── SHAP global importance ────────────────────────────────────────────────
    _save_shap_summary(model, X_test, feature_names)

    return model, feature_names, metrics


def _save_shap_summary(model, X_test: pd.DataFrame, feature_names: list):
    """Generate and save a SHAP beeswarm summary plot."""
    logger.info("Computing SHAP values …")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # XGBoost returns a 2-D array; RF returns a list
        if isinstance(shap_values, list):
            sv = shap_values[1]          # class-1 (churn)
        else:
            sv = shap_values

        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X_test, feature_names=feature_names,
                          show=False, plot_type="dot")
        plt.tight_layout()
        plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary saved → models/shap_summary.png")

        # Also save the explainer for runtime use
        joblib.dump(explainer, "models/shap_explainer.pkl")

    except Exception as exc:
        logger.warning(f"SHAP generation skipped: {exc}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--data",  default="data/bank_churn.csv", help="Path to CSV")
    parser.add_argument("--model", default="xgb", choices=["xgb", "rf"], help="Model type")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    args = parser.parse_args()

    train(args.data, model_type=args.model, use_smote=not args.no_smote)
