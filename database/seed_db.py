"""
database/seed_db.py
===================
One-time script: initialise the Neon database, load the CSV,
run the model on all customers, and store predictions.

Usage:
    python -m database.seed_db --data data/bank_churn.csv
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.db_connection import init_db, insert_customers, upsert_prediction
from utils.preprocessing import clean_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def seed(csv_path: str, run_predictions: bool = True):
    logger.info("Initialising database schema …")
    init_db()

    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = clean_data(df)

    # Map CSV column names → ORM model column names
    col_map = {
        "CreditScore":     "credit_score",
        "Geography":       "geography",
        "Gender":          "gender",
        "Age":             "age",
        "Tenure":          "tenure",
        "Balance":         "balance",
        "NumOfProducts":   "num_products",
        "HasCreditCard":   "has_credit_card",
        "IsActiveMember":  "is_active",
        "EstimatedSalary": "estimated_salary",
        "Exited":          "exited",
    }
    df.rename(columns=col_map, inplace=True)
    df["customer_id"] = range(1, len(df) + 1)

    logger.info(f"Inserting {len(df)} customer records …")
    insert_customers(df)

    if run_predictions:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
        if not os.path.exists(model_path):
            logger.warning("Model not found. Skipping predictions. Train first with ml/train_model.py")
            return

        from ml.predict import predict_churn
        logger.info("Generating predictions for all customers …")

        # Reverse-map for predict_churn (expects original names)
        rev_map = {v: k for k, v in col_map.items()}
        df_orig = df.rename(columns=rev_map)

        for i, row in df_orig.iterrows():
            cid = int(df.loc[i, "customer_id"])
            try:
                res = predict_churn(row.to_dict())
                upsert_prediction(cid, res["churn_probability"], res["risk_level"])
            except Exception as e:
                logger.debug(f"Skipping customer {cid}: {e}")

            if (i + 1) % 500 == 0:
                logger.info(f"  … {i+1} / {len(df_orig)} customers scored")

        logger.info("All predictions stored ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           default="data/bank_churn.csv")
    parser.add_argument("--no-predictions", action="store_true")
    args = parser.parse_args()
    seed(args.data, run_predictions=not args.no_predictions)
