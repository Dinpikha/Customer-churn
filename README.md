# 🏦 ChurnSight — Bank Customer Churn Prediction System

A production-grade machine learning platform that predicts customer churn,
explains predictions with SHAP, and surfaces actionable insights through a
professional Streamlit analytics dashboard backed by Neon PostgreSQL.

---

## 📐 Architecture

```
customer_churn_project/
├── data/
│   ├── bank_churn.csv          ← Kaggle dataset (or generated demo)
│   └── download_data.py        ← Kaggle API / synthetic data generator
│
├── models/
│   ├── churn_model.pkl         ← Trained XGBoost model
│   ├── churn_model_xgb.pkl
│   ├── feature_names.pkl       ← Ordered feature list
│   ├── metrics.pkl             ← Evaluation metrics dict
│   ├── shap_explainer.pkl      ← TreeExplainer for runtime SHAP
│   └── shap_summary.png        ← Global feature importance plot
│
├── database/
│   ├── __init__.py
│   ├── db_connection.py        ← SQLAlchemy ORM + Neon helpers
│   └── seed_db.py              ← One-time DB initialisation script
│
├── ml/
│   ├── __init__.py
│   ├── train_model.py          ← Training pipeline (XGBoost / RF)
│   └── predict.py              ← Inference + SHAP explanation layer
│
├── app/
│   └── streamlit_app.py        ← 4-page Streamlit dashboard
│
├── utils/
│   ├── __init__.py
│   ├── feature_engineering.py  ← Domain feature construction
│   └── preprocessing.py        ← Cleaning, encoding, scaling
│
├── .streamlit/
│   ├── config.toml             ← Dark-theme Streamlit config
│   └── secrets.toml.example    ← Deployment secrets template
│
├── .env.example                ← Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourname/churnsight.git
cd churnsight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Kaggle (recommended)**
```bash
# Install kaggle and set KAGGLE_USERNAME + KAGGLE_KEY
pip install kaggle
python data/download_data.py
```

**Option B — Generate synthetic demo data**
```bash
python -c "from data.download_data import generate_demo_csv; generate_demo_csv()"
```

### 3. Train the model

```bash
python -m ml.train_model --data data/bank_churn.csv --model xgb
```

Expected output:
```
ACCURACY  : 0.8640
PRECISION : 0.7512
RECALL    : 0.5834
F1        : 0.6567
ROC_AUC   : 0.8810
```

### 4. Set up Neon PostgreSQL (optional)

1. Create a free database at [neon.tech](https://neon.tech)
2. Copy the connection string into `.env`:
   ```
   DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/churndb?sslmode=require
   ```
3. Seed the database:
   ```bash
   python -m database.seed_db --data data/bank_churn.csv
   ```

### 5. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI cards, age distribution, geography donut, balance scatter |
| **Churn Analysis** | Churn rate by geography / age group / products, SHAP importance |
| **Predict Churn** | Real-time prediction form with SHAP waterfall explanation |
| **High Risk Customers** | Filterable table, AUM-at-risk estimate, risk distribution |

---

## 🤖 Model Details

| Component | Detail |
|-----------|--------|
| Algorithm | XGBoost (gradient boosted trees) |
| Imbalance handling | SMOTE oversampling |
| Explainability | SHAP TreeExplainer |
| Persistence | joblib `.pkl` |

**Engineered features:**

| Feature | Description |
|---------|-------------|
| `balance_to_salary_ratio` | Account balance / estimated salary |
| `customer_value_score` | Composite 0–100 value index |
| `inactivity_flag` | Inactive member with zero balance |
| `age_group_encoded` | Ordinal: Young / Middle-Aged / Senior / Elderly |
| `products_per_tenure` | Product adoption rate over tenure |
| `high_balance_inactive` | High balance but inactive member |

---

## ☁️ Deploy to Streamlit Cloud

1. Push repo to GitHub (include `models/` in the repo or use DVC/S3).
2. Connect repo at [share.streamlit.io](https://share.streamlit.io).
3. Set the entry point: `app/streamlit_app.py`
4. Add `DATABASE_URL` in **Secrets** (Settings → Secrets):
   ```toml
   [database]
   DATABASE_URL = "postgresql://..."
   ```

---

## 💡 Portfolio Enhancement Ideas

### 🔥 High Impact
- **MLflow tracking** — log every experiment run (params, metrics, artefacts)
- **Feature store** — Feast or a simple custom store to version features
- **A/B model comparison** — compare XGBoost vs LightGBM vs CatBoost in the UI
- **Drift detection** — monitor feature distributions over time with Evidently AI
- **CI/CD pipeline** — GitHub Actions: test → train → evaluate → promote model

### 📈 Business-Oriented
- **CLV (Customer Lifetime Value)** modelling alongside churn probability
- **Intervention ROI calculator** — estimate revenue saved per intervention
- **Cohort retention analysis** — track retained customers over time
- **Segment-level churn forecasting** — time-series churn prediction per segment

### 🛠 Engineering Depth
- **FastAPI serving layer** — REST endpoint wrapping the model for production inference
- **Redis caching** — cache SHAP explanations to reduce latency
- **Dockerise** — `docker-compose up` the full stack (app + postgres)
- **pytest suite** — unit tests for preprocessing, feature engineering, predictions

---

## 📜 License

MIT — free for personal and commercial use.
