import os
import pandas as pd
import streamlit as st
from fredapi import Fred
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
st.set_page_config(page_title="Macro Dials v2", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=60*60)
def fred_series(api_key: str, series_id: str) -> pd.Series:
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = pd.Series(s).dropna().sort_index()
    return s

def latest_value(s: pd.Series):
    if s is None or len(s) == 0:
        return None
    return float(s.iloc[-1])

def score_thresholds(x: float, low: float, high: float, invert: bool = False) -> int:
    """
    Returns -1 / 0 / +1 based on thresholds:
      +1 if x <= low  (or >= high if invert)
       0 if between
      -1 if x >= high (or <= low if invert)
    """
    if x is None:
        return 0
    if not invert:
        if x <= low:
            return +1
        if x >= high:
            return -1
        return 0
    else:
        # invert meaning: higher is better
        if x >= high:
            return +1
        if x <= low:
            return -1
        return 0
# =============================
# Teil 2 Helpers
# =============================

def _to_weekly(s: pd.Series) -> pd.Series:
    return s.resample("W-FRI").last().dropna()

def _percentile_score(x: pd.Series, lookback: int = 520) -> float:
    x = x.dropna()
    if len(x) < 30:
        return float("nan")
    window = x.iloc[-lookback:] if len(x) > lookback else x
    return float(window.rank(pct=True).iloc[-1] * 100.0)

def make_gauge(title: str, value: float) -> go.Figure:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 33], "color": "rgba(255,0,0,0.25)"},
                {"range": [33, 66], "color": "rgba(255,255,0,0.25)"},
                {"range": [66, 100], "color": "rgba(0,255,0,0.25)"},
            ],
            "bar": {"thickness": 0.25},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def classify_regime(liq_score: float, risk_score: float):
    if np.isnan(liq_score) or np.isnan(risk_score):
        return "Unknown", "⚪"
    liq_hi = liq_score >= 66
    liq_lo = liq_score <= 33
    risk_hi = risk_score >= 66
    risk_lo = risk_score <= 33

    if liq_hi and risk_lo:
        return "Risk-On", "🟢"
    if liq_lo and risk_hi:
        return "Risk-Off", "🔴"
    return "Neutral", "🟡"
def pill(score: int) -> str:
    return "🟢 Tailwind" if score == 1 else "🟡 Neutral" if score == 0 else "🔴 Headwind"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Macro Dials v2")
st.sidebar.caption("Erste lauffähige Version mit echten FRED-Daten + Ampel-Score.")

api_key = st.sidebar.text_input("FRED API Key", value=os.getenv("FRED_API_KEY", ""), type="password")
st.sidebar.markdown("FRED Key holen: FRED Account → API Keys (kostenlos).")

st.sidebar.divider()
st.sidebar.subheader("Serien (FRED IDs)")
series_real_yield = st.sidebar.text_input("10Y Real Yield", "DFII10")
series_fed_funds  = st.sidebar.text_input("Fed Funds Rate", "FEDFUNDS")
series_cpi        = st.sidebar.text_input("CPI YoY", "CPIAUCSL")
series_unrate     = st.sidebar.text_input("Unemployment Rate", "UNRATE")

st.sidebar.divider()
st.sidebar.subheader("Schwellen (kannst du später fein-tunen)")
# Real yield thresholds
ry_low  = st.sidebar.number_input("Real Yield low (<= = grün)", value=1.0, step=0.1)
ry_high = st.sidebar.number_input("Real Yield high (>= = rot)", value=2.0, step=0.1)

# Fed funds thresholds
ff_low  = st.sidebar.number_input("Fed Funds low (<= = grün)", value=2.0, step=0.25)
ff_high = st.sidebar.number_input("Fed Funds high (>= = rot)", value=4.0, step=0.25)

# CPI thresholds (lower inflation = better)
cpi_low  = st.sidebar.number_input("CPI YoY low (<= = grün)", value=3.0, step=0.1)
cpi_high = st.sidebar.number_input("CPI YoY high (>= = rot)", value=4.0, step=0.1)

# Unemployment thresholds (higher unemployment usually bad for growth)
u_low  = st.sidebar.number_input("Unemployment low (<= = grün)", value=4.5, step=0.1)
u_high = st.sidebar.number_input("Unemployment high (>= = rot)", value=5.5, step=0.1)

# -----------------------------
# Main
# -----------------------------
st.title("Macro Dials v2 — Dashboard")
st.caption("Ampel-Score (🟢/🟡/🔴) aus 4 Kernserien. Nächster Schritt: Liquidity + Risk + Matrix wie im Screenshot.")

if not api_key:
    st.warning("Gib links deinen FRED API Key ein, dann laden wir die Daten.")
    st.stop()

# Load data
with st.spinner("Lade FRED-Daten…"):
    real_yield = fred_series(api_key, series_real_yield)
    fedfunds   = fred_series(api_key, series_fed_funds)
    cpi        = fred_series(api_key, series_cpi)
    unrate     = fred_series(api_key, series_unrate)

# Transform CPI to YoY %
cpi_yoy = (cpi.pct_change(12) * 100).dropna()

# Latest values
ry = latest_value(real_yield)
ff = latest_value(fedfunds)
cy = latest_value(cpi_yoy)
ur = latest_value(unrate)

# Scores
score_ry = score_thresholds(ry, ry_low, ry_high, invert=False)   # lower real yield = better
score_ff = score_thresholds(ff, ff_low, ff_high, invert=False)   # lower fed funds = better
score_cy = score_thresholds(cy, cpi_low, cpi_high, invert=False) # lower inflation = better
score_ur = score_thresholds(ur, u_low, u_high, invert=False)     # lower unemployment = better (simple proxy)

total = score_ry + score_ff + score_cy + score_ur

# Header KPIs
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1.2])
c1.metric("10Y Real Yield", f"{ry:.2f}%" if ry is not None else "—", pill(score_ry))
c2.metric("Fed Funds", f"{ff:.2f}%" if ff is not None else "—", pill(score_ff))
c3.metric("CPI YoY", f"{cy:.2f}%" if cy is not None else "—", pill(score_cy))
c4.metric("Unemployment", f"{ur:.2f}%" if ur is not None else "—", pill(score_ur))

regime = "🟢 Risk-On" if total >= 2 else "🟡 Transition" if total >= 0 else "🔴 Risk-Off"
c5.metric("Macro Regime", regime, f"Total Score: {total:+d}")

st.divider()

# Plot helper
def plot_series(title: str, s: pd.Series):
    fig, ax = plt.subplots()
    ax.plot(s.index, s.values)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

left, right = st.columns(2)
with left:
    plot_series("10Y Real Yield (DFII10)", real_yield)
    plot_series("Fed Funds Rate (FEDFUNDS)", fedfunds)

with right:
    plot_series("CPI YoY % (derived from CPIAUCSL)", cpi_yoy)
    plot_series("Unemployment Rate (UNRATE)", unrate)

st.info("Nächster Schritt: Liquidity-Dial (Fed Balance Sheet, RRP/TGA Proxy), Risk-Dial (VIX/HY), und eine farbige Matrix wie im Screenshot.")
st.markdown("---")
st.header("Teil 2: Liquidity-Dial + Risk-Dial + Matrix")

# --- Daten holen
walcl = fred_series(api_key, walcl_id)
rrp   = fred_series(api_key, rrp_id)
tga   = fred_series(api_key, tga_id)

vix = fred_series(api_key, vix_id)
hy  = fred_series(api_key, hy_id)

# --- Weekly
walcl_w = _to_weekly(walcl)
rrp_w   = _to_weekly(rrp)
tga_w   = _to_weekly(tga)

# Liquidity Impulse (ΔWALCL - ΔRRP - ΔTGA), geglättet
liq_imp = (walcl_w.diff() - rrp_w.diff() - tga_w.diff()).dropna()
liq_imp_smooth = liq_imp.rolling(8).mean()

lb = int(lookback_years * 52)
liq_score = _percentile_score(liq_imp_smooth, lookback=lb)

# Risk: VIX + HY (höher = riskiger)
risk_df = pd.DataFrame({
    "vix": _to_weekly(vix),
    "hy":  _to_weekly(hy),
}).dropna()

# simple Combo: Percentile des aktuellen Levels (stabiler als pct_change)
risk_combo = (risk_df["vix"].rank(pct=True) + risk_df["hy"].rank(pct=True)) / 2.0
risk_score = float((risk_combo.iloc[-lb:] if len(risk_combo) > lb else risk_combo).iloc[-1] * 100.0)

regime, dot = classify_regime(liq_score, risk_score)

# --- Anzeigen
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.plotly_chart(make_gauge("Liquidity (Percentile)", liq_score), use_container_width=True)
with c2:
    st.plotly_chart(make_gauge("Risk (Percentile)", risk_score), use_container_width=True)
with c3:
    st.metric("Regime (Teil 2)", f"{dot} {regime}")

st.subheader("Liquidity × Risk Matrix")
st.caption("🟩 Risk-On • 🟥 Risk-Off • 🟨 Neutral")

liq_hi = liq_score >= 50
risk_hi = risk_score >= 50

matrix = {
    "Risk Low": {"Liq Low": "🟨", "Liq High": "🟩"},
    "Risk High": {"Liq Low": "🟥", "Liq High": "🟨"},
}
st.table(matrix)

# optional: kleine Zeitreihe der Liquidity Impulse
st.subheader("Liquidity Impulse (ΔWALCL − ΔRRP − ΔTGA)")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(liq_imp_smooth.index, liq_imp_smooth.values)
ax.set_title("Weekly Liquidity Impulse (8W MA)")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
