"""
app/streamlit_app.py
====================
ChurnSight — Premium Fintech Analytics Dashboard
Aesthetic: Dark luxury · Glassmorphism · Electric cyan gradients
Run with:  streamlit run app/streamlit_app.py
"""

import os
import sys
import warnings
import logging
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight | Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#080c14",
    "surface":  "#0d1321",
    "card":     "#111827",
    "card2":    "#141d2e",
    "border":   "rgba(255,255,255,0.07)",
    "border2":  "rgba(255,255,255,0.13)",
    "cyan":     "#00d4ff",
    "teal":     "#00b4aa",
    "emerald":  "#10b981",
    "amber":    "#f59e0b",
    "rose":     "#f43f5e",
    "violet":   "#8b5cf6",
    "text":     "#f0f4ff",
    "text2":    "#94a3b8",
    "text3":    "#4b5e7a",
    "success":  "#10b981",
    "warning":  "#f59e0b",
    "danger":   "#f43f5e",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {{
    background-color: {C['bg']} !important;
    font-family: 'DM Sans', sans-serif;
    color: {C['text']};
}}
.main .block-container {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
}}

[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 10% 20%, rgba(0,212,255,0.045) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(16,185,129,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(139,92,246,0.025) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0b1120 0%, #080c14 100%) !important;
    border-right: 1px solid {C['border2']} !important;
}}
[data-testid="stSidebar"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, {C['cyan']}, transparent);
}}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}
[data-testid="stSidebar"] .stRadio label {{ font-size: 12px !important; color: {C['text2']} !important; }}

.sidebar-logo {{
    padding: 28px 20px 20px;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 8px;
}}
.logo-mark {{
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, {C['cyan']}, {C['emerald']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.logo-sub {{
    font-size: 9px; letter-spacing: 0.2em; text-transform: uppercase;
    color: {C['text3']}; margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}}
.sidebar-badge {{
    display: inline-flex; align-items: center; gap: 5px;
    margin-top: 10px; padding: 3px 8px;
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 4px; font-size: 9px; letter-spacing: 0.1em;
    text-transform: uppercase; color: {C['cyan']};
    font-family: 'JetBrains Mono', monospace;
}}
.dot-live {{
    width: 5px; height: 5px; background: {C['cyan']};
    border-radius: 50%; animation: pulse-dot 2s infinite;
}}
@keyframes pulse-dot {{
    0%, 100% {{ opacity:1; box-shadow: 0 0 0 0 rgba(0,212,255,0.4); }}
    50% {{ opacity:0.7; box-shadow: 0 0 0 4px rgba(0,212,255,0); }}
}}

.kpi-grid {{
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 14px; margin-bottom: 28px;
}}
.kpi-card {{
    position: relative; background: {C['card']};
    border-radius: 14px; padding: 20px 22px 18px;
    border: 1px solid {C['border']}; overflow: hidden;
    transition: transform 0.2s, border-color 0.3s; cursor: default;
}}
.kpi-card::before {{
    content: ''; position: absolute; inset: 0; border-radius: 14px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(16,185,129,0.1), transparent);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity: 0; transition: opacity 0.3s;
}}
.kpi-card:hover {{ transform: translateY(-2px); }}
.kpi-card:hover::before {{ opacity: 1; }}
.kpi-icon {{
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; margin-bottom: 12px;
}}
.kpi-label {{
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: {C['text3']}; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Syne', sans-serif; font-size: 26px;
    font-weight: 700; letter-spacing: -0.02em; color: {C['text']}; line-height: 1;
}}
.kpi-delta {{
    font-size: 11px; margin-top: 8px; display: flex;
    align-items: center; gap: 4px; color: {C['text3']};
}}
.kpi-glow {{
    position: absolute; width: 80px; height: 80px; border-radius: 50%;
    top: -20px; right: -20px; filter: blur(30px); opacity: 0.15;
}}

.chart-card {{
    background: {C['card']}; border: 1px solid {C['border']};
    border-radius: 16px; padding: 22px 24px; margin-bottom: 16px;
    position: relative; overflow: hidden;
}}
.chart-card::after {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,212,255,0.4) 50%, transparent 100%);
}}
.chart-title {{
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600;
    letter-spacing: 0.01em; color: {C['text']}; margin-bottom: 4px;
}}
.chart-subtitle {{
    font-size: 11px; color: {C['text3']}; margin-bottom: 18px;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
}}

.page-hero {{
    margin-bottom: 32px; padding-bottom: 24px;
    border-bottom: 1px solid {C['border']}; position: relative;
}}
.page-hero .breadcrumb {{
    font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase;
    color: {C['cyan']}; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px; opacity: 0.8;
}}
.page-hero h1 {{
    font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800;
    letter-spacing: -0.03em; color: {C['text']}; margin: 0 0 6px 0;
    background: linear-gradient(120deg, {C['text']} 60%, {C['cyan']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.page-hero .subtitle {{ font-size: 13px; color: {C['text2']}; font-weight: 300; }}
.page-hero .timestamp {{
    position: absolute; top: 0; right: 0;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: {C['text3']}; letter-spacing: 0.08em;
}}

.section-label {{
    display: flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: {C['text3']}; margin: 28px 0 16px 0;
}}
.section-label::after {{ content: ''; flex: 1; height: 1px; background: {C['border']}; }}

.risk-high {{
    display:inline-flex;align-items:center;gap:5px;padding:4px 10px;
    border-radius:6px;font-size:11px;font-weight:600;
    background:rgba(244,63,94,0.12);color:{C['danger']};
    border:1px solid rgba(244,63,94,0.3);
    font-family:'JetBrains Mono',monospace;letter-spacing:.05em;
}}
.risk-medium {{
    display:inline-flex;align-items:center;gap:5px;padding:4px 10px;
    border-radius:6px;font-size:11px;font-weight:600;
    background:rgba(245,158,11,0.12);color:{C['warning']};
    border:1px solid rgba(245,158,11,0.3);
    font-family:'JetBrains Mono',monospace;letter-spacing:.05em;
}}
.risk-low {{
    display:inline-flex;align-items:center;gap:5px;padding:4px 10px;
    border-radius:6px;font-size:11px;font-weight:600;
    background:rgba(16,185,129,0.12);color:{C['success']};
    border:1px solid rgba(16,185,129,0.3);
    font-family:'JetBrains Mono',monospace;letter-spacing:.05em;
}}

.result-panel {{
    background: {C['card2']}; border: 1px solid {C['border2']};
    border-radius: 18px; padding: 28px; margin: 24px 0;
    position: relative; overflow: hidden;
}}
.result-panel::before {{
    content: ''; position: absolute; inset: 0; border-radius: 18px;
    background: linear-gradient(135deg, rgba(0,212,255,0.03), rgba(16,185,129,0.03));
}}
.prob-big {{
    font-family: 'Syne', sans-serif; font-size: 52px;
    font-weight: 800; letter-spacing: -0.04em;
    text-align: center; line-height: 1; margin: 8px 0;
}}
.prob-bar-wrap {{
    background: rgba(255,255,255,0.05); border-radius: 100px;
    height: 6px; margin: 14px 0; overflow: hidden;
}}

.action-item {{
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; background: rgba(255,255,255,0.02);
    border: 1px solid {C['border']}; border-radius: 10px; margin-bottom: 8px;
    font-size: 13px; transition: background 0.2s, border-color 0.2s;
}}
.action-item:hover {{ background: rgba(255,255,255,0.04); border-color: {C['border2']}; }}
.action-num {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {C['text3']}; min-width: 16px; }}

.status-row {{
    display: flex; align-items: center; gap: 20px;
    padding: 10px 16px; background: rgba(255,255,255,0.02);
    border: 1px solid {C['border']}; border-radius: 8px; margin-bottom: 24px;
}}
.status-item {{
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: {C['text3']}; letter-spacing: 0.08em;
}}
.status-dot {{ width: 6px; height: 6px; border-radius: 50%; }}

.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid {C['border2']} !important;
    border-radius: 8px !important; color: {C['text']} !important;
}}
label, .stRadio label span {{
    color: {C['text3']} !important; font-size: 11px !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.stButton > button {{
    background: linear-gradient(135deg, {C['cyan']}, {C['teal']}) !important;
    color: #050d1a !important; border: none !important;
    border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 13px !important;
    letter-spacing: 0.05em !important; padding: 12px 28px !important;
    transition: opacity 0.2s, transform 0.15s !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; transform: translateY(-1px) !important; }}

[data-testid="stForm"] {{
    background: {C['card']} !important; border: 1px solid {C['border']} !important;
    border-radius: 16px !important; padding: 24px !important;
}}
[data-testid="stDataFrameContainer"] {{
    border: 1px solid {C['border']} !important; border-radius: 12px !important; overflow: hidden;
}}
[data-baseweb="tag"] {{
    background: rgba(0,212,255,0.12) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: 4px !important; color: {C['cyan']} !important;
}}
hr {{ border: none !important; border-top: 1px solid {C['border']} !important; margin: 24px 0 !important; }}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {C['text3']}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)


# ── Plotly theme ──────────────────────────────────────────────────────────────
def pgo(fig, height=360, title=""):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=C["text2"], size=11),
        height=height,
        title=dict(text=title, font=dict(family="Syne",size=13,color=C["text"]),x=0),
        margin=dict(l=8,r=8,t=36 if title else 8,b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=11,color=C["text2"]),
                    bordercolor=C["border"],borderwidth=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(size=10,color=C["text3"]),showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(size=10,color=C["text3"]),showline=False),
    )
    return fig

RISK_CLR = {"High": C["danger"], "Medium": C["warning"], "Low": C["success"]}


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_badge(risk):
    cls = {"High":"risk-high","Medium":"risk-medium","Low":"risk-low"}.get(risk,"risk-low")
    dot = {"High":"#f43f5e","Medium":"#f59e0b","Low":"#10b981"}.get(risk,"#10b981")
    return f'<span class="{cls}"><span style="width:5px;height:5px;border-radius:50%;background:{dot};display:inline-block"></span>{risk}</span>'


def kpi_card(icon, icon_bg, label, value, delta="", glow_color=C["cyan"]):
    return f"""
    <div class="kpi-card">
        <div class="kpi-glow" style="background:{glow_color}"></div>
        <div class="kpi-icon" style="background:{icon_bg}">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-delta">' + delta + '</div>' if delta else ''}
    </div>"""


def section(text):
    st.markdown(f'<div class="section-label"><span>{text}</span></div>', unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "bank_churn.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        df.drop(columns=[c for c in ["RowNumber","CustomerId","Surname"] if c in df.columns], inplace=True)
        return df
    rng = np.random.default_rng(42)
    n   = 10_000
    geo = rng.choice(["France","Germany","Spain"], n, p=[.5,.25,.25])
    age = rng.integers(18, 75, n)
    logit = (-1.4 + 0.04*(age-36) + 0.6*(geo=="Germany")
             - 0.5*rng.integers(0,2,n) + rng.normal(0,.5,n))
    exited = (1/(1+np.exp(-logit)) > 0.5).astype(int)
    return pd.DataFrame({
        "CreditScore": rng.integers(350,851,n),
        "Geography": geo, "Gender": rng.choice(["Male","Female"],n),
        "Age": age, "Tenure": rng.integers(0,11,n),
        "Balance": np.where(rng.random(n)<.35,0,rng.uniform(5000,250000,n)).round(2),
        "NumOfProducts": rng.choice([1,2,3,4],n,p=[.5,.46,.03,.01]),
        "HasCreditCard": rng.integers(0,2,n), "IsActiveMember": rng.integers(0,2,n),
        "EstimatedSalary": rng.uniform(11500,200000,n).round(2), "Exited": exited,
    })


# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-mark">◈ ChurnSight</div>
        <div class="logo-sub">Intelligence Platform</div>
        <div class="sidebar-badge"><span class="dot-live"></span>Live · v2.0</div>
    </div>""", unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Overview","Churn Analysis","Predict Churn","High Risk"],
        icons=["grid","graph-up","cpu","exclamation-diamond"],
        default_index=0,
        styles={
            "container": {"padding":"8px 12px","background":"transparent"},
            "icon": {"color":C["text3"],"font-size":"13px"},
            "nav-link": {
                "font-size":"12px","font-family":"DM Sans","color":C["text2"],
                "padding":"9px 14px","border-radius":"8px","margin":"2px 0",
                "--hover-color":"rgba(0,212,255,0.06)",
            },
            "nav-link-selected": {
                "background":"linear-gradient(135deg,rgba(0,212,255,0.12),rgba(16,185,129,0.08))",
                "color":C["cyan"],"font-weight":"600",
                "border":"1px solid rgba(0,212,255,0.2)",
            },
        },
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:{C["text3"]};margin-bottom:8px;font-family:JetBrains Mono,monospace">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("", ["CSV / Demo Data","PostgreSQL (Neon)"], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)

    model_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__),"..","models","churn_model.pkl"))
    mc = C["success"] if model_exists else C["warning"]
    mt = "Model Ready" if model_exists else "Demo Mode — No Model"
    st.markdown(f"""
    <div style="padding:10px 14px;background:rgba(255,255,255,0.02);border:1px solid {C['border']};border-radius:8px;">
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="width:6px;height:6px;background:{mc};border-radius:50%;display:block"></span>
            <span style="font-family:JetBrains Mono,monospace;font-size:10px;color:{mc};letter-spacing:.06em">{mt}</span>
        </div>
        <div style="font-size:9px;color:{C['text3']};margin-top:4px;font-family:JetBrains Mono,monospace">XGBoost · SHAP Explainer</div>
    </div>""", unsafe_allow_html=True)

# Load data
if data_source == "PostgreSQL (Neon)":
    try:
        from database.db_connection import fetch_all_customers
        df = fetch_all_customers()
        df.rename(columns={"credit_score":"CreditScore","geography":"Geography","gender":"Gender",
            "age":"Age","tenure":"Tenure","balance":"Balance","num_products":"NumOfProducts",
            "has_credit_card":"HasCreditCard","is_active":"IsActiveMember",
            "estimated_salary":"EstimatedSalary","exited":"Exited"}, inplace=True)
        st.sidebar.success("✓ Connected")
    except Exception as e:
        st.sidebar.error(f"DB: {e}")
        df = load_data()
else:
    df = load_data()

try:
    from utils.feature_engineering import engineer_features
    dv = engineer_features(df)
except Exception:
    dv = df.copy()

dv["age_group"]   = pd.cut(dv["Age"], bins=[0,30,45,60,100], labels=["18–30","31–45","46–60","60+"])
dv["churn_label"] = dv["Exited"].map({0:"Retained",1:"Churned"})
NOW = datetime.datetime.now().strftime("%d %b %Y  %H:%M UTC")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    total   = len(df)
    churned = int(df["Exited"].sum())
    rate    = churned/total*100
    active  = df["IsActiveMember"].mean()*100
    avg_bal = df["Balance"].mean()
    aum_r   = df.loc[df["Exited"]==1,"Balance"].sum()

    st.markdown(f"""
    <div class="page-hero">
        <div class="timestamp">{NOW}</div>
        <div class="breadcrumb">◈ ChurnSight / Overview</div>
        <h1>Executive Dashboard</h1>
        <div class="subtitle">Real-time churn intelligence across your entire customer portfolio</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-row">
        <div class="status-item"><span class="status-dot" style="background:{C['success']}"></span>System Operational</div>
        <div class="status-item"><span class="status-dot" style="background:{C['cyan']}"></span>{total:,} Records Loaded</div>
        <div class="status-item"><span class="status-dot" style="background:{'#f43f5e' if rate>25 else '#f59e0b' if rate>15 else '#10b981'}"></span>Churn Rate {rate:.1f}%</div>
        <div class="status-item"><span class="status-dot" style="background:{C['violet']}"></span>3 Geographies</div>
    </div>""", unsafe_allow_html=True)

    kc = (
        kpi_card("👥","rgba(0,212,255,0.1)","Total Customers",f"{total:,}",
                 f'<span style="color:{C["cyan"]}">↗</span> Full portfolio',C["cyan"])
      + kpi_card("⚠️","rgba(244,63,94,0.1)","Churned",f"{churned:,}",
                 f'<span style="color:{C["danger"]}">{rate:.1f}% churn rate</span>',C["danger"])
      + kpi_card("💰","rgba(16,185,129,0.1)","Avg Balance",f"${avg_bal:,.0f}",
                 f'Portfolio average',C["emerald"])
      + kpi_card("📊","rgba(139,92,246,0.1)","Active Members",f"{active:.1f}%",
                 f'Engagement index',C["violet"])
      + kpi_card("🔥","rgba(245,158,11,0.1)","AUM at Risk",f"${aum_r/1e6:.1f}M",
                 f'Churned balance',C["amber"])
    )
    st.markdown(f'<div class="kpi-grid">{kc}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.6,1], gap="medium")
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Age Distribution by Churn Status</div><div class="chart-subtitle">Histogram · Overlay · All customers</div>', unsafe_allow_html=True)
        fig = px.histogram(dv, x="Age", color="churn_label", nbins=35, opacity=0.75, barmode="overlay",
            color_discrete_map={"Retained":C["cyan"],"Churned":C["danger"]})
        fig.update_traces(marker_line_width=0)
        pgo(fig, 300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Portfolio by Geography</div><div class="chart-subtitle">Donut · Customer share</div>', unsafe_allow_html=True)
        geo_c = dv["Geography"].value_counts().reset_index()
        fig = px.pie(geo_c, values="count", names="Geography", hole=0.65,
            color_discrete_sequence=[C["cyan"],C["violet"],C["teal"]])
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=11,color=C["text2"]),
                          marker=dict(line=dict(color=C["bg"],width=2)))
        fig.add_annotation(text=f"<b>{total:,}</b>",x=0.5,y=0.52,
            font=dict(family="Syne",size=22,color=C["text"]),showarrow=False)
        fig.add_annotation(text="customers",x=0.5,y=0.42,
            font=dict(family="DM Sans",size=11,color=C["text3"]),showarrow=False)
        pgo(fig, 300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3, gap="medium")
    with col3:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn by Tenure</div><div class="chart-subtitle">Bar · Yearly cohort</div>', unsafe_allow_html=True)
        ten = dv.groupby("Tenure")["Exited"].mean().reset_index()
        ten["rate"] = ten["Exited"]*100
        fig = go.Figure(go.Bar(x=ten["Tenure"],y=ten["rate"],
            marker=dict(color=ten["rate"].tolist(),
                        colorscale=[[0,C["teal"]],[0.5,C["cyan"]],[1,C["danger"]]],line_width=0),
            text=ten["rate"].round(1).astype(str)+"%",textposition="outside",
            textfont=dict(size=9,color=C["text3"])))
        pgo(fig,270)
        fig.update_layout(xaxis_title="Tenure (yrs)",yaxis_title="Churn %",
                          xaxis=dict(tickmode="linear",dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card"><div class="chart-title">Active vs Inactive</div><div class="chart-subtitle">Ring · Member status</div>', unsafe_allow_html=True)
        act = dv["IsActiveMember"].value_counts().reset_index()
        act["label"] = act["IsActiveMember"].map({1:"Active",0:"Inactive"})
        fig = px.pie(act,values="count",names="label",hole=0.7,
            color_discrete_sequence=[C["emerald"],"#1e293b"])
        fig.update_traces(textposition="outside",textinfo="percent+label",
                          textfont=dict(size=11),
                          marker=dict(line=dict(color=C["bg"],width=2)))
        fig.add_annotation(text=f"<b>{active:.0f}%</b>",x=0.5,y=0.53,
            font=dict(family="Syne",size=20,color=C["success"]),showarrow=False)
        fig.add_annotation(text="active",x=0.5,y=0.43,
            font=dict(family="DM Sans",size=11,color=C["text3"]),showarrow=False)
        pgo(fig,270)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="chart-card"><div class="chart-title">Products Held</div><div class="chart-subtitle">Distribution · All customers</div>', unsafe_allow_html=True)
        prod = dv["NumOfProducts"].value_counts().sort_index().reset_index()
        fig = go.Figure(go.Bar(
            x=prod["NumOfProducts"].astype(str)+" prod",y=prod["count"],
            marker=dict(color=[C["cyan"],C["teal"],C["amber"],C["danger"]],line_width=0),
            text=prod["count"],textposition="outside",
            textfont=dict(size=10,color=C["text3"])))
        pgo(fig,270)
        fig.update_layout(showlegend=False,yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    section("Balance vs. Salary — Churn Overlay")
    st.markdown('<div class="chart-card"><div class="chart-title">Balance vs. Estimated Salary</div><div class="chart-subtitle">Scatter · 2,000 sample · Color = churn status</div>', unsafe_allow_html=True)
    s = dv.sample(min(2000,len(dv)),random_state=42)
    fig = px.scatter(s,x="EstimatedSalary",y="Balance",color="churn_label",opacity=0.45,
        color_discrete_map={"Retained":C["cyan"],"Churned":C["danger"]},
        labels={"EstimatedSalary":"Estimated Salary ($)","Balance":"Account Balance ($)"})
    fig.update_traces(marker_size=4)
    pgo(fig,380)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Churn Analysis":
    st.markdown(f"""
    <div class="page-hero">
        <div class="timestamp">{NOW}</div>
        <div class="breadcrumb">◈ ChurnSight / Analysis</div>
        <h1>Churn Analysis</h1>
        <div class="subtitle">Segment-level breakdown of churn drivers and behavioural patterns</div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Geography</div><div class="chart-subtitle">Bar · % churned per region</div>', unsafe_allow_html=True)
        geo = dv.groupby("Geography")["Exited"].agg(c="sum",t="count").reset_index()
        geo["rate"] = geo["c"]/geo["t"]*100
        fig = go.Figure(go.Bar(x=geo["Geography"],y=geo["rate"],
            marker=dict(color=[C["danger"],C["warning"],C["cyan"]],line_width=0),
            text=geo["rate"].round(1).astype(str)+"%",textposition="outside",
            textfont=dict(size=11,color=C["text"])))
        pgo(fig,320)
        fig.update_layout(yaxis_title="Churn Rate (%)",showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Age Group</div><div class="chart-subtitle">Bar · Senior customers are highest risk</div>', unsafe_allow_html=True)
        ag = dv.groupby("age_group",observed=True)["Exited"].agg(c="sum",t="count").reset_index()
        ag["rate"] = ag["c"]/ag["t"]*100
        fig = go.Figure(go.Bar(x=ag["age_group"].astype(str),y=ag["rate"],
            marker=dict(color=[C["success"] if r<15 else C["warning"] if r<30 else C["danger"] for r in ag["rate"]],
                        line_width=0),
            text=ag["rate"].round(1).astype(str)+"%",textposition="outside",
            textfont=dict(size=11,color=C["text"])))
        pgo(fig,320)
        fig.update_layout(yaxis_title="Churn Rate (%)",showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.markdown('<div class="chart-card"><div class="chart-title">Churn Rate by Number of Products</div><div class="chart-subtitle">Bar · 3-4 product customers show extreme churn</div>', unsafe_allow_html=True)
        pr = dv.groupby("NumOfProducts")["Exited"].agg(c="sum",t="count").reset_index()
        pr["rate"] = pr["c"]/pr["t"]*100
        fig = go.Figure(go.Bar(x=pr["NumOfProducts"].astype(str)+" product(s)",y=pr["rate"],
            marker=dict(color=pr["rate"].tolist(),
                        colorscale=[[0,C["success"]],[0.4,C["amber"]],[1,C["danger"]]],line_width=0),
            text=pr["rate"].round(1).astype(str)+"%",textposition="outside",
            textfont=dict(size=11,color=C["text"])))
        pgo(fig,320)
        fig.update_layout(yaxis_title="Churn Rate (%)",showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card"><div class="chart-title">Credit Score Distribution</div><div class="chart-subtitle">Violin · Retained vs Churned comparison</div>', unsafe_allow_html=True)
        fig = px.violin(dv,x="churn_label",y="CreditScore",box=True,points=False,color="churn_label",
            color_discrete_map={"Retained":C["cyan"],"Churned":C["danger"]})
        fig.update_traces(meanline_visible=True,meanline_color="white",
                          line_color="rgba(255,255,255,0.3)")
        pgo(fig,320)
        fig.update_layout(xaxis_title="",yaxis_title="Credit Score",showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    section("Model Feature Importance")
    model_path = os.path.join(os.path.dirname(__file__),"..","models","churn_model.pkl")
    feat_path  = os.path.join(os.path.dirname(__file__),"..","models","feature_names.pkl")
    st.markdown('<div class="chart-card"><div class="chart-title">Feature Importance</div><div class="chart-subtitle">XGBoost gain · Higher = stronger churn predictor</div>', unsafe_allow_html=True)
    if os.path.exists(model_path) and os.path.exists(feat_path):
        import joblib
        model = joblib.load(model_path)
        feats = joblib.load(feat_path)
        imp   = pd.Series(model.feature_importances_,index=feats).nlargest(12)
        nv    = (imp-imp.min())/(imp.max()-imp.min())
        cols_i = [f"rgba(0,{int(180+75*v)},{int(255*v)},0.85)" for v in nv.values[::-1]]
        fig = go.Figure(go.Bar(x=imp.values[::-1],y=imp.index[::-1],orientation="h",
            marker=dict(color=cols_i,line_width=0),
            text=[f"{v:.4f}" for v in imp.values[::-1]],
            textposition="outside",textfont=dict(size=10,color=C["text3"])))
        pgo(fig,400)
        fig.update_layout(xaxis_title="Importance Score")
    else:
        numeric = dv.select_dtypes(include=[np.number]).columns.tolist()
        corr = dv[numeric].corr()["Exited"].drop("Exited").sort_values()
        fig = go.Figure(go.Bar(x=corr.values,y=corr.index,orientation="h",
            marker=dict(color=[C["danger"] if v>0 else C["teal"] for v in corr.values],line_width=0)))
        pgo(fig,400)
        fig.update_layout(xaxis_title="Pearson Correlation with Churn")
        st.info("💡 Train the model for XGBoost feature importances.")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Churn":
    st.markdown(f"""
    <div class="page-hero">
        <div class="timestamp">{NOW}</div>
        <div class="breadcrumb">◈ ChurnSight / Predict</div>
        <h1>Churn Predictor</h1>
        <div class="subtitle">Enter customer attributes for an instant AI-powered churn risk assessment</div>
    </div>""", unsafe_allow_html=True)

    if not model_exists:
        st.markdown(f"""
        <div style="padding:12px 18px;background:rgba(245,158,11,0.08);
                    border:1px solid rgba(245,158,11,0.25);border-radius:10px;
                    margin-bottom:20px;font-size:12px;color:{C['warning']};
                    font-family:JetBrains Mono,monospace;letter-spacing:.04em">
            ⚡ Demo mode — no trained model found.
            Run: <code>python -m ml.train_model --data data/bank_churn.csv</code>
        </div>""", unsafe_allow_html=True)

    with st.form("predict_form"):
        fa, fb, fc = st.columns(3)
        with fa:
            credit_score = st.number_input("Credit Score",300,900,650)
            age          = st.number_input("Age",18,100,38)
            tenure       = st.number_input("Tenure (years)",0,10,5)
            geography    = st.selectbox("Geography",["France","Germany","Spain"])
        with fb:
            balance      = st.number_input("Account Balance ($)",0,500_000,75_000)
            salary       = st.number_input("Estimated Salary ($)",0,500_000,60_000)
            num_products = st.selectbox("Number of Products",[1,2,3,4])
            gender       = st.selectbox("Gender",["Male","Female"])
        with fc:
            has_cc    = st.radio("Has Credit Card",["Yes","No"],horizontal=True)
            is_active = st.radio("Is Active Member",["Yes","No"],horizontal=True)
            st.markdown(f"""
            <div style="margin-top:20px;padding:14px;background:rgba(0,212,255,0.04);
                        border:1px solid rgba(0,212,255,0.12);border-radius:10px;
                        font-size:11px;color:{C['text3']};line-height:1.8;
                        font-family:JetBrains Mono,monospace">
                MODEL<br><span style="color:{C['cyan']}">XGBoost</span> · 400 trees<br>
                EXPLAINER<br><span style="color:{C['cyan']}">SHAP TreeExplainer</span><br>
                FEATURES<br><span style="color:{C['cyan']}">16 (incl. engineered)</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("◈  Run Churn Analysis", use_container_width=True)

    if submitted:
        customer = {
            "CreditScore":credit_score,"Geography":geography,"Gender":gender,
            "Age":age,"Tenure":tenure,"Balance":balance,"NumOfProducts":num_products,
            "HasCreditCard":1 if has_cc=="Yes" else 0,
            "IsActiveMember":1 if is_active=="Yes" else 0,
            "EstimatedSalary":salary,
        }
        if not model_exists:
            prob = min(0.97,(age>45)*0.20+(geography=="Germany")*0.15
                       +(is_active=="No")*0.25+(num_products>=3)*0.20
                       +(balance==0)*0.10+np.random.uniform(0,.1))
            risk = "High" if prob>=0.70 else "Medium" if prob>=0.40 else "Low"
            shap_df = None
        else:
            from ml.predict import predict_churn, get_top_shap_drivers
            res     = predict_churn(customer)
            prob    = res["churn_probability"]
            risk    = res["risk_level"]
            shap_df = get_top_shap_drivers(customer)

        risk_color = RISK_CLR[risk]
        prob_pct   = prob*100
        grad_bar   = {"High":f"linear-gradient(90deg,{C['danger']},{C['rose']})",
                      "Medium":f"linear-gradient(90deg,{C['amber']},{C['warning']})",
                      "Low":f"linear-gradient(90deg,{C['teal']},{C['emerald']})"}[risk]

        section("Prediction Result")
        rc1, rc2 = st.columns([1,2], gap="medium")

        with rc1:
            age_label = "Senior" if age>60 else "Middle-Aged" if age>45 else "Young Adult"
            st.markdown(f"""
            <div class="result-panel" style="text-align:center">
                <div style="font-family:JetBrains Mono,monospace;font-size:9px;
                            letter-spacing:.15em;text-transform:uppercase;color:{C['text3']}">
                    Churn Probability
                </div>
                <div class="prob-big" style="background:linear-gradient(135deg,{risk_color},{risk_color}99);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
                    {prob_pct:.1f}%
                </div>
                <div class="prob-bar-wrap">
                    <div style="width:{prob_pct:.1f}%;height:100%;border-radius:100px;background:{grad_bar}"></div>
                </div>
                <div style="margin:16px auto 0">{risk_badge(risk)}</div>
                <div style="margin-top:14px;font-size:11px;color:{C['text3']};line-height:1.6">
                    {"🔴 Immediate retention action required" if risk=="High"
                     else "🟡 Monitor and engage proactively" if risk=="Medium"
                     else "🟢 Low risk — standard programme"}
                </div>
                <div style="margin-top:16px;padding-top:16px;border-top:1px solid {C['border']};
                            display:grid;grid-template-columns:1fr 1fr;gap:10px;text-align:left">
                    <div>
                        <div style="font-size:9px;color:{C['text3']};font-family:JetBrains Mono,monospace;
                                    letter-spacing:.1em;text-transform:uppercase">Age Group</div>
                        <div style="font-size:13px;color:{C['text']};font-weight:500;margin-top:3px">{age_label}</div>
                    </div>
                    <div>
                        <div style="font-size:9px;color:{C['text3']};font-family:JetBrains Mono,monospace;
                                    letter-spacing:.1em;text-transform:uppercase">Balance</div>
                        <div style="font-size:13px;color:{C['text']};font-weight:500;margin-top:3px">${balance:,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size:9px;color:{C['text3']};font-family:JetBrains Mono,monospace;
                                    letter-spacing:.1em;text-transform:uppercase">Tenure</div>
                        <div style="font-size:13px;color:{C['text']};font-weight:500;margin-top:3px">{tenure} years</div>
                    </div>
                    <div>
                        <div style="font-size:9px;color:{C['text3']};font-family:JetBrains Mono,monospace;
                                    letter-spacing:.1em;text-transform:uppercase">Products</div>
                        <div style="font-size:13px;color:{C['text']};font-weight:500;margin-top:3px">{num_products}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        with rc2:
            if shap_df is not None and len(shap_df)>0:
                shap_colors = [C["danger"] if v>0 else C["teal"] for v in shap_df["shap_value"]]
                fig = go.Figure(go.Bar(
                    x=shap_df["shap_value"][::-1].reset_index(drop=True),
                    y=shap_df["feature"][::-1].reset_index(drop=True),
                    orientation="h",
                    marker=dict(color=shap_colors[::-1],line_width=0,opacity=0.85),
                    text=shap_df["direction"][::-1].reset_index(drop=True),
                    textposition="outside",textfont=dict(size=9,color=C["text3"])))
                pgo(fig,380,"SHAP Feature Attribution")
                fig.update_layout(
                    xaxis_title="SHAP value (impact on log-odds of churn)",
                    shapes=[dict(type="line",x0=0,x1=0,y0=-0.5,y1=len(shap_df)-0.5,
                                 line=dict(color="rgba(255,255,255,0.15)",dash="dot",width=1))])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f"""
                <div style="padding:40px;text-align:center;color:{C['text3']};font-size:12px;
                            font-family:JetBrains Mono,monospace;border:1px dashed {C['border']};
                            border-radius:12px;margin-top:20px">
                    SHAP explanations available after model training.<br>
                    <code style="color:{C['cyan']}">python -m ml.train_model</code>
                </div>""", unsafe_allow_html=True)

        section("Recommended Interventions")
        actions = {
            "High":   [("01","🚨","Escalate to senior retention specialist immediately"),
                       ("02","💎","Offer VIP rate improvement or fee waiver"),
                       ("03","📅","Schedule relationship manager call within 48 hours")],
            "Medium": [("01","📧","Trigger personalised product recommendation email"),
                       ("02","🎁","Enrol in loyalty rewards accelerator programme"),
                       ("03","📊","Flag for next monthly churn review cycle")],
            "Low":    [("01","✅","Include in quarterly NPS survey"),
                       ("02","📱","Promote digital self-service feature adoption"),
                       ("03","📈","Eligible for cross-sell — high CLV segment")],
        }
        act_html = ""
        for num, icon, text in actions[risk]:
            act_html += f"""
            <div class="action-item">
                <span class="action-num">{num}</span>
                <span style="font-size:16px">{icon}</span>
                <span style="font-size:13px;color:{C['text2']}">{text}</span>
                <span style="margin-left:auto;font-size:10px;color:{C['text3']};
                             font-family:JetBrains Mono,monospace">→</span>
            </div>"""
        st.markdown(act_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HIGH RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "High Risk":
    st.markdown(f"""
    <div class="page-hero">
        <div class="timestamp">{NOW}</div>
        <div class="breadcrumb">◈ ChurnSight / High Risk</div>
        <h1>High Risk Customers</h1>
        <div class="subtitle">Prioritised intervention list — sorted by churn probability</div>
    </div>""", unsafe_allow_html=True)

    if model_exists:
        from ml.predict import predict_batch
        with st.spinner("Scoring …"):
            scored = predict_batch(dv.head(500))
    else:
        rng2 = np.random.default_rng(99)
        dv2  = dv.copy()
        dv2["churn_probability"] = (
            (dv2["Age"]>45)*0.20+(dv2["Geography"]=="Germany")*0.15
            +(dv2["IsActiveMember"]==0)*0.25+(dv2["NumOfProducts"]>=3)*0.20
            +(dv2["Balance"]==0)*0.10+rng2.uniform(0,.15,len(dv2))
        ).clip(0,.99)
        dv2["risk_level"] = dv2["churn_probability"].apply(
            lambda p: "High" if p>=0.70 else "Medium" if p>=0.40 else "Low")
        scored = dv2

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1,1,1])
    with f1: risk_f = st.multiselect("Risk Level",["High","Medium","Low"],default=["High","Medium"])
    with f2: geo_f  = st.multiselect("Geography",scored["Geography"].unique().tolist(),
                                      default=scored["Geography"].unique().tolist())
    with f3: top_n  = st.slider("Top N customers",10,300,75)
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = scored[
        scored["risk_level"].isin(risk_f) & scored["Geography"].isin(geo_f)
    ].nlargest(top_n,"churn_probability")

    high_ct = (filtered["risk_level"]=="High").sum()
    med_ct  = (filtered["risk_level"]=="Medium").sum()
    avg_p   = filtered["churn_probability"].mean()
    aum_r   = (filtered["Balance"]*filtered["churn_probability"]).sum()

    kc = (
        kpi_card("🔴","rgba(244,63,94,0.1)","High Risk",f"{high_ct:,}","Immediate action",C["danger"])
      + kpi_card("🟡","rgba(245,158,11,0.1)","Medium Risk",f"{med_ct:,}","Monitor closely",C["amber"])
      + kpi_card("📉","rgba(139,92,246,0.1)","Avg Churn Prob",f"{avg_p:.1%}","Filtered set",C["violet"])
      + kpi_card("💸","rgba(244,63,94,0.1)","AUM at Risk",f"${aum_r/1e6:.1f}M","Balance × prob",C["danger"])
    )
    st.markdown(f'<div class="kpi-grid" style="grid-template-columns:repeat(4,1fr)">{kc}</div>', unsafe_allow_html=True)

    tc1, tc2 = st.columns([1.6,1], gap="medium")
    with tc1:
        section("Customer Risk Table")
        disp = filtered[["Geography","Gender","Age","Balance","NumOfProducts",
                          "IsActiveMember","churn_probability","risk_level"]].copy()
        disp.columns = ["Geography","Gender","Age","Balance","Products","Active","Churn Prob","Risk"]
        disp["Balance"]    = disp["Balance"].apply(lambda x:f"${x:,.0f}")
        disp["Churn Prob"] = (disp["Churn Prob"]*100).round(1).astype(str)+"%"
        disp["Active"]     = disp["Active"].map({1:"✓ Active",0:"✗ Inactive"})
        st.dataframe(disp, use_container_width=True, height=480)

    with tc2:
        section("Risk Distribution")
        st.markdown('<div class="chart-card"><div class="chart-title">Probability Distribution</div><div class="chart-subtitle">Histogram · Filtered customers</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered,x="churn_probability",color="risk_level",nbins=25,
            barmode="overlay",opacity=0.75,
            color_discrete_map={"High":C["danger"],"Medium":C["amber"],"Low":C["success"]})
        fig.update_traces(marker_line_width=0)
        pgo(fig,240)
        fig.update_layout(xaxis_title="Churn Probability",yaxis_title="# Customers",legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        section("Risk by Geography")
        st.markdown('<div class="chart-card"><div class="chart-title">High Risk by Region</div><div class="chart-subtitle">Stacked bar · Filtered set</div>', unsafe_allow_html=True)
        grisk = filtered.groupby(["Geography","risk_level"]).size().reset_index(name="count")
        fig = px.bar(grisk,x="Geography",y="count",color="risk_level",barmode="stack",
            color_discrete_map={"High":C["danger"],"Medium":C["amber"],"Low":C["success"]})
        fig.update_traces(marker_line_width=0)
        pgo(fig,230)
        fig.update_layout(yaxis_title="Customers",legend_title="Risk")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
