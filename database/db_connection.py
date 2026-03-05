"""
database/db_connection.py
=========================
PostgreSQL connection manager using SQLAlchemy + Neon.
Handles table creation, data insertion, and query retrieval.
"""

import os
import logging
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, text,
    Column, Integer, Float, String, DateTime, Boolean,
    MetaData, Table
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class Customer(Base):
    """Raw customer demographic and banking data."""
    __tablename__ = "customers"

    customer_id      = Column(Integer, primary_key=True, index=True)
    credit_score     = Column(Integer, nullable=False)
    geography        = Column(String(50), nullable=False)
    gender           = Column(String(10), nullable=False)
    age              = Column(Integer, nullable=False)
    tenure           = Column(Integer, nullable=False)
    balance          = Column(Float, nullable=False)
    num_products     = Column(Integer, nullable=False)
    has_credit_card  = Column(Boolean, default=False)
    is_active        = Column(Boolean, default=False)
    estimated_salary = Column(Float, nullable=False)
    exited           = Column(Boolean, default=False)   # ground-truth label
    created_at       = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """ML model predictions stored per customer."""
    __tablename__ = "predictions"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    customer_id        = Column(Integer, index=True, nullable=False)
    churn_probability  = Column(Float, nullable=False)
    risk_level         = Column(String(10), nullable=False)   # Low / Medium / High
    prediction_date    = Column(DateTime, default=datetime.utcnow)
    model_version      = Column(String(30), default="v1.0")


# ── Engine factory ────────────────────────────────────────────────────────────

def get_engine():
    """
    Build a SQLAlchemy engine from the DATABASE_URL env var.
    Uses NullPool to avoid connection issues on Neon's serverless tier.
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError(
            "DATABASE_URL not set. Copy .env.example → .env and fill in credentials."
        )
    # Neon requires SSL; psycopg2 is happy with ?sslmode=require in the URL.
    engine = create_engine(db_url, poolclass=NullPool, echo=False)
    logger.info("Database engine created successfully.")
    return engine


def init_db():
    """Create all tables if they don't already exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified / created.")
    return engine


# ── Session helper ────────────────────────────────────────────────────────────

@contextmanager
def get_session():
    """Context-managed SQLAlchemy session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Session error: {exc}")
        raise
    finally:
        session.close()


# ── Data insertion helpers ────────────────────────────────────────────────────

def insert_customers(df: pd.DataFrame) -> int:
    """
    Bulk-insert a DataFrame of customers.

    Args:
        df: DataFrame whose columns match the Customer ORM model.

    Returns:
        Number of rows inserted.
    """
    records = df.to_dict(orient="records")
    with get_session() as session:
        session.bulk_insert_mappings(Customer, records)
    logger.info(f"Inserted {len(records)} customer records.")
    return len(records)


def upsert_prediction(customer_id: int, churn_prob: float, risk_level: str,
                       model_version: str = "v1.0"):
    """
    Insert (or replace) a prediction row for a given customer.
    Simple delete-then-insert strategy for Neon compatibility.
    """
    with get_session() as session:
        # Remove stale prediction for this customer
        session.query(Prediction).filter(
            Prediction.customer_id == customer_id
        ).delete()

        pred = Prediction(
            customer_id=customer_id,
            churn_probability=round(churn_prob, 4),
            risk_level=risk_level,
            model_version=model_version,
        )
        session.add(pred)
    logger.debug(f"Prediction stored — customer {customer_id}: {risk_level} ({churn_prob:.2%})")


# ── Query helpers ─────────────────────────────────────────────────────────────

def fetch_all_customers() -> pd.DataFrame:
    """Return full customers table as a DataFrame."""
    engine = get_engine()
    return pd.read_sql("SELECT * FROM customers ORDER BY customer_id", engine)


def fetch_predictions() -> pd.DataFrame:
    """Return predictions joined with customer data."""
    engine = get_engine()
    query = """
        SELECT
            c.customer_id,
            c.geography,
            c.gender,
            c.age,
            c.balance,
            c.num_products,
            c.is_active,
            c.estimated_salary,
            c.exited,
            p.churn_probability,
            p.risk_level,
            p.prediction_date
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        ORDER BY p.churn_probability DESC
    """
    return pd.read_sql(query, engine)


def fetch_high_risk_customers(top_n: int = 100) -> pd.DataFrame:
    """Return the top-N customers with highest churn probability."""
    engine = get_engine()
    query = f"""
        SELECT
            c.customer_id,
            c.geography,
            c.gender,
            c.age,
            c.credit_score,
            c.balance,
            c.num_products,
            c.tenure,
            c.is_active,
            p.churn_probability,
            p.risk_level
        FROM customers c
        JOIN predictions p ON c.customer_id = p.customer_id
        WHERE p.risk_level IN ('High', 'Medium')
        ORDER BY p.churn_probability DESC
        LIMIT {top_n}
    """
    return pd.read_sql(query, engine)


def get_summary_stats() -> dict:
    """Return key dashboard metrics from the database."""
    engine = get_engine()
    with engine.connect() as conn:
        total      = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
        churned    = conn.execute(text("SELECT COUNT(*) FROM customers WHERE exited = TRUE")).scalar()
        avg_bal    = conn.execute(text("SELECT ROUND(AVG(balance)::numeric, 2) FROM customers")).scalar()
        high_risk  = conn.execute(
            text("SELECT COUNT(*) FROM predictions WHERE risk_level = 'High'")
        ).scalar()

    return {
        "total_customers": total or 0,
        "churned_customers": churned or 0,
        "churn_rate": round((churned / total * 100), 2) if total else 0,
        "avg_balance": float(avg_bal or 0),
        "high_risk_count": high_risk or 0,
    }
