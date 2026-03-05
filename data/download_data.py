"""
data/download_data.py
=====================
Downloads the Bank Customer Churn dataset from Kaggle or a public mirror.

Usage:
    python data/download_data.py

Requires:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables (or use kaggle.json).
"""

import os
import sys
import urllib.request
import zipfile
import logging

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "shubhendra7/bank-customer-churn-prediction"
OUTPUT_DIR     = os.path.dirname(__file__)
CSV_FILENAME   = "bank_churn.csv"


def download_via_kaggle():
    """Use Kaggle CLI if credentials are available."""
    import kaggle
    logger.info("Downloading via Kaggle API …")
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=OUTPUT_DIR,
        unzip=True,
    )
    logger.info(f"Dataset downloaded to {OUTPUT_DIR}/")


def generate_demo_csv():
    """
    Generate a realistic synthetic dataset if Kaggle is unavailable.
    Mirrors the structure of the original Bank Churn dataset exactly.
    """
    import numpy as np
    import pandas as pd

    logger.info("Generating synthetic demo dataset …")
    rng = np.random.default_rng(42)
    n   = 10_000

    geography = rng.choice(["France", "Germany", "Spain"], n, p=[0.50, 0.25, 0.25])
    gender    = rng.choice(["Male", "Female"], n, p=[0.54, 0.46])
    age       = rng.integers(18, 75, n)
    balance   = np.where(
        rng.random(n) < 0.35, 0,
        rng.uniform(5_000, 250_000, n)
    ).round(2)
    tenure       = rng.integers(0, 11, n)
    num_products = rng.choice([1, 2, 3, 4], n, p=[0.50, 0.46, 0.03, 0.01])
    credit_score = rng.integers(350, 851, n)
    salary       = rng.uniform(11_500, 200_000, n).round(2)
    has_cc       = rng.integers(0, 2, n)
    is_active    = rng.integers(0, 2, n)

    # Churn depends on age, geography, activity, balance, products
    churn_logit = (
        -1.5
        + 0.04  * (age - 36)
        + 0.60  * (geography == "Germany")
        - 0.50  * is_active
        + 0.80  * (num_products >= 3)
        + 0.30  * (balance == 0)
        - 0.002 * (credit_score - 500)
        + rng.normal(0, 0.5, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    exited = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "RowNumber":       range(1, n + 1),
        "CustomerId":      rng.integers(15_000_000, 16_000_000, n),
        "Surname":         ["Smith"] * n,
        "CreditScore":     credit_score,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCreditCard":   has_cc,
        "IsActiveMember":  is_active,
        "EstimatedSalary": salary,
        "Exited":          exited,
    })

    out_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)
    df.to_csv(out_path, index=False)
    logger.info(f"Synthetic dataset saved → {out_path}  ({n} rows, churn rate: {exited.mean():.1%})")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    csv_path = os.path.join(OUTPUT_DIR, CSV_FILENAME)

    if os.path.exists(csv_path):
        logger.info(f"Dataset already exists at {csv_path}")
        sys.exit(0)

    try:
        download_via_kaggle()
    except Exception as e:
        logger.warning(f"Kaggle download failed ({e}). Generating synthetic data instead.")
        generate_demo_csv()
