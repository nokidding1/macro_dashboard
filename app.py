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
def _to_weekly(s: pd.Series) -> pd.Series:
    return s.resample("W").last()

def _percentile_score(s: pd.Series, lookback: int = 520) -> float:
    if s is None or len(s) < 10:
        return 50.0
    s2 = s.dropna().iloc[-lookback:]
    return float(s2.rank(pct=True).iloc[-1] * 100.0)
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
# --- Serien IDs
walcl_id = "WALCL"
rrp_id   = "RRPONTSYD"
tga_id   = "WTREGEN"
vix_id   = "VIXCLS"
hy_id    = "BAMLH0A0HYM2"
lookback_years = 5
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
# ---------- helpers ----------
@st.cache_resource
def get_fred_client():
    return Fred(api_key=st.secrets["FRED_API_KEY"])

@st.cache_data(ttl=60 * 60 * 6)  # 6h cache
def fred_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    fred = get_fred_client()
    s = fred.get_series(series_id, observation_start=start)
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce")
    s.name = series_id
    return s.dropna()

def sahm_rule(unrate: pd.Series) -> pd.Series:
    """
    Sahm Rule signal: current 3-month avg unemployment rate
    minus minimum 3-month avg over previous 12 months.
    Recession trigger often discussed at >= 0.50 percentage points.
    """
    unrate = unrate.sort_index()
    u3 = unrate.rolling(3).mean()
    u3_min_12m = u3.rolling(12).min()
    sahm = u3 - u3_min_12m
    sahm.name = "SAHM"
    return sahm.dropna()

def make_recession_df(start="1990-01-01") -> pd.DataFrame:
    dgs10 = fred_series("DGS10", start)
    dgs2  = fred_series("DGS2", start)
    spread = (dgs10 - dgs2).rename("10Y-2Y Spread")

    unrate = fred_series("UNRATE", start).rename("Unemployment Rate")
    sahm = sahm_rule(unrate)

    icsa = fred_series("ICSA", start).rename("Initial Claims")
    lei = fred_series("USSLIND", start).rename("Leading Index (USSLIND)")

    df = pd.concat([spread, unrate, sahm, icsa, lei], axis=1).sort_index()
    return df

# ---------- UI ----------
def recession_panel():
    st.subheader("USA Rezessions-Indikatoren")

    with st.sidebar:
        start = st.date_input("Startdatum", value=pd.to_datetime("2000-01-01")).strftime("%Y-%m-%d")

    df = make_recession_df(start=start)

    # KPIs (letzter Wert)
    # KPIs (letzter gültiger Wert pro Serie)
last_spread = df["10Y-2Y Spread"].dropna().iloc[-1] if df["10Y-2Y Spread"].dropna().size else np.nan
last_unrate = df["Unemployment Rate"].dropna().iloc[-1] if df["Unemployment Rate"].dropna().size else np.nan
last_sahm   = df["SAHM"].dropna().iloc[-1] if df["SAHM"].dropna().size else np.nan
last_icsa   = df["Initial Claims"].dropna().iloc[-1] if df["Initial Claims"].dropna().size else np.nan

def fmt_num(x, digits=2, suffix=""):
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}{suffix}"

def fmt_int(x):
    return "n/a" if pd.isna(x) else f"{int(x):,}".replace(",", "'")

c1, c2, c3, c4 = st.columns(4)
c1.metric("10Y-2Y Spread", fmt_num(last_spread, 2, " %-Pkt"))
c2.metric("Arbeitslosigkeit (UNRATE)", fmt_num(last_unrate, 1, " %"))
c3.metric("Sahm Rule", fmt_num(last_sahm, 2, " %-Pkt"))
c4.metric("Initial Claims", fmt_int(last_icsa))
    # einfache Ampel-Logik (nur Orientierung!)
    risk_notes = []
    if last_spread < 0:
        risk_notes.append("🔴 Zinskurve invertiert (Spread < 0)")
    else:
        risk_notes.append("🟢 Zinskurve nicht invertiert")

    if last_sahm >= 0.50:
        risk_notes.append("🔴 Sahm Rule >= 0.50 (klassischer Rezessions-Trigger)")
    elif last_sahm >= 0.35:
        risk_notes.append("🟠 Sahm Rule erhöht (>= 0.35)")
    else:
        risk_notes.append("🟢 Sahm Rule niedrig")

    st.write("**Signal-Check (grob):**")
    st.write("\n".join(risk_notes))

    st.divider()

    st.write("### Charts")
    st.line_chart(df[["10Y-2Y Spread"]].dropna())
    st.line_chart(df[["Unemployment Rate", "SAHM"]].dropna())
    st.line_chart(df[["Initial Claims"]].dropna())
    st.line_chart(df[["Leading Index (USSLIND)"]].dropna())

    with st.expander("Daten anzeigen"):
        st.dataframe(df.tail(50))

# In deiner main.py dann recession_panel() aufrufen
recession_panel ()
