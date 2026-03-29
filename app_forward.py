import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fredapi import Fred

st.set_page_config(page_title="Macro Dials Forward", layout="wide")

# =========================================================
# CONFIG / API KEY
# =========================================================

def resolve_fred_api_key() -> str:
    env_key = os.getenv("FRED_API_KEY", "")
    secret_key = ""
    try:
        secret_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        secret_key = ""
    return env_key or secret_key or ""


# =========================================================
# FRED HELPERS
# =========================================================

@st.cache_resource
def get_fred_client(api_key: str):
    return Fred(api_key=api_key)

@st.cache_data(ttl=60 * 60 * 6)
def fred_series(api_key: str, series_id: str, start: str = "1990-01-01") -> pd.Series:
    fred = get_fred_client(api_key)
    s = fred.get_series(series_id, observation_start=start)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s.dropna().sort_index()


# =========================================================
# GENERIC HELPERS
# =========================================================

def latest_value(s: pd.Series):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.iloc[-1])

def last_valid(s: pd.Series):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[-1]

def to_weekly(s: pd.Series) -> pd.Series:
    return pd.Series(s).dropna().resample("W-FRI").last().dropna()

def to_monthly_avg(s: pd.Series) -> pd.Series:
    return pd.Series(s).dropna().resample("MS").mean().dropna()

def percentile_score(s: pd.Series, lookback: int = 520) -> float:
    s = pd.Series(s).dropna()
    if len(s) < 30:
        return float("nan")
    window = s.iloc[-lookback:] if len(s) > lookback else s
    return float(window.rank(pct=True).iloc[-1] * 100.0)

def score_thresholds(x: float, low: float, high: float, invert: bool = False) -> int:
    if x is None or pd.isna(x):
        return 0
    if not invert:
        if x <= low:
            return +1
        if x >= high:
            return -1
        return 0
    else:
        if x >= high:
            return +1
        if x <= low:
            return -1
        return 0

def pill(score: int) -> str:
    return "🟢 Tailwind" if score == 1 else "🟡 Neutral" if score == 0 else "🔴 Headwind"

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
    return "Neutral / Transition", "🟡"

def make_gauge(title: str, value: float, good_high: bool = True) -> go.Figure:
    if value is None or pd.isna(value):
        value = 0.0

    if good_high:
        steps = [
            {"range": [0, 33], "color": "rgba(255,0,0,0.25)"},
            {"range": [33, 66], "color": "rgba(255,255,0,0.25)"},
            {"range": [66, 100], "color": "rgba(0,255,0,0.25)"},
        ]
    else:
        steps = [
            {"range": [0, 33], "color": "rgba(0,255,0,0.25)"},
            {"range": [33, 66], "color": "rgba(255,255,0,0.25)"},
            {"range": [66, 100], "color": "rgba(255,0,0,0.25)"},
        ]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": steps,
            "bar": {"thickness": 0.25},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def plot_series(title: str, s: pd.Series):
    fig, ax = plt.subplots()
    ax.plot(s.index, s.values)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def count_consecutive_true(cond_series: pd.Series) -> int:
    x = pd.Series(cond_series).dropna().astype(bool)
    count = 0
    for val in reversed(x.tolist()):
        if val:
            count += 1
        else:
            break
    return count

def zscore_last(s: pd.Series, lookback: int = 156) -> float:
    s = pd.Series(s).dropna()
    if len(s) < 20:
        return np.nan
    w = s.iloc[-lookback:] if len(s) > lookback else s
    mu = w.mean()
    sd = w.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((w.iloc[-1] - mu) / sd)

def delta_6m_annualized(s: pd.Series) -> float:
    s = pd.Series(s).dropna()
    if len(s) < 7:
        return np.nan
    last = s.iloc[-1]
    prev = s.iloc[-7]
    if prev == 0 or np.isnan(prev):
        return np.nan
    return float(((last / prev) ** 2 - 1) * 100.0)


# =========================================================
# RECESSION HELPERS
# =========================================================

def sahm_rule(unrate: pd.Series) -> pd.Series:
    unrate = pd.Series(unrate).dropna().sort_index()
    u3 = unrate.rolling(3).mean()
    u3_min_12m = u3.rolling(12).min()
    sahm = u3 - u3_min_12m
    sahm.name = "SAHM"
    return sahm.dropna()

def make_recession_df(api_key: str, start: str = "1990-01-01") -> pd.DataFrame:
    dgs10 = fred_series(api_key, "DGS10", start=start)
    dgs2 = fred_series(api_key, "DGS2", start=start)
    spread = (dgs10 - dgs2).rename("10Y-2Y Spread")

    unrate = fred_series(api_key, "UNRATE", start=start).rename("Unemployment Rate")
    sahm = sahm_rule(unrate)

    icsa = fred_series(api_key, "ICSA", start=start).rename("Initial Claims")
    lei = fred_series(api_key, "USSLIND", start=start).rename("Leading Index (USSLIND)")

    df = pd.concat([spread, unrate, sahm, icsa, lei], axis=1).sort_index()
    return df


# =========================================================
# FORWARD RECESSION RISK PANEL
# =========================================================

def build_forward_recession_data(api_key: str, start: str = "1995-01-01"):
    dgs10 = fred_series(api_key, "DGS10", start)
    dgs2 = fred_series(api_key, "DGS2", start)
    dgs3m = fred_series(api_key, "DGS3MO", start)

    unrate = fred_series(api_key, "UNRATE", start)
    icsa = fred_series(api_key, "ICSA", start)
    lei = fred_series(api_key, "USSLIND", start)
    hy_oas = fred_series(api_key, "BAMLH0A0HYM2", start)

    spread_10y_2y = (dgs10 - dgs2).rename("spread_10y_2y")
    spread_10y_3m = (dgs10 - dgs3m).rename("spread_10y_3m")

    unrate_m = to_monthly_avg(unrate).rename("unrate")
    sahm = sahm_rule(unrate_m).rename("sahm")
    lei_m = to_monthly_avg(lei).rename("lei")

    icsa_w = to_weekly(icsa).rename("icsa")
    icsa_yoy = (((icsa_w / icsa_w.shift(52)) - 1.0) * 100.0).rename("icsa_yoy")

    daily = pd.concat([spread_10y_2y, spread_10y_3m, hy_oas.rename("hy_oas")], axis=1).sort_index()
    monthly = pd.concat([unrate_m, sahm, lei_m], axis=1).sort_index()
    weekly = pd.concat([icsa_w, icsa_yoy], axis=1).sort_index()

    return daily, monthly, weekly

def compute_forward_recession_scores(daily: pd.DataFrame, monthly: pd.DataFrame, weekly: pd.DataFrame):
    last_spread_2y = latest_value(daily["spread_10y_2y"])
    last_spread_3m = latest_value(daily["spread_10y_3m"])
    last_hy = latest_value(daily["hy_oas"])
    last_unrate = latest_value(monthly["unrate"])
    last_sahm = latest_value(monthly["sahm"])
    last_claims = latest_value(weekly["icsa"])
    last_claims_yoy = latest_value(weekly["icsa_yoy"])

    spread2_m = to_monthly_avg(daily["spread_10y_2y"])
    spread3m_m = to_monthly_avg(daily["spread_10y_3m"])
    inv_2y_months = count_consecutive_true(spread2_m < 0)
    inv_3m_months = count_consecutive_true(spread3m_m < 0)

    lei_6m_ann = delta_6m_annualized(monthly["lei"])
    claims_z = zscore_last(weekly["icsa"], lookback=156)
    hy_pct = percentile_score(daily["hy_oas"], lookback=260)

    points = 0

    if not np.isnan(last_spread_3m) and last_spread_3m < 0:
        points += 15
    if inv_3m_months >= 6:
        points += 10
    if inv_3m_months >= 12:
        points += 5

    if not np.isnan(last_spread_2y) and last_spread_2y < 0:
        points += 10
    if inv_2y_months >= 6:
        points += 5

    if not np.isnan(lei_6m_ann) and lei_6m_ann < -2:
        points += 15
    if not np.isnan(lei_6m_ann) and lei_6m_ann < -5:
        points += 10

    if not np.isnan(last_hy) and last_hy > 4.5:
        points += 10
    if not np.isnan(last_hy) and last_hy > 6.0:
        points += 10

    if not np.isnan(last_claims_yoy) and last_claims_yoy > 5:
        points += 5
    if not np.isnan(last_claims_yoy) and last_claims_yoy > 12:
        points += 5

    if not np.isnan(last_sahm) and last_sahm >= 0.35:
        points += 10
    if not np.isnan(last_sahm) and last_sahm >= 0.50:
        points += 15

    recession_probability = float(min(points, 100))

    fr_points = 0

    if not np.isnan(hy_pct):
        fr_points += min(max((hy_pct - 50) * 0.6, 0), 20)

    if not np.isnan(lei_6m_ann):
        if lei_6m_ann < 0:
            fr_points += 10
        if lei_6m_ann < -3:
            fr_points += 10
        if lei_6m_ann < -6:
            fr_points += 10

    if inv_3m_months > 0:
        fr_points += min(inv_3m_months * 1.5, 20)

    if not np.isnan(claims_z):
        if claims_z > 0.5:
            fr_points += 5
        if claims_z > 1.0:
            fr_points += 10
        if claims_z > 1.5:
            fr_points += 10

    if not np.isnan(last_sahm):
        if last_sahm >= 0.25:
            fr_points += 10
        if last_sahm >= 0.35:
            fr_points += 10

    forward_risk_score = float(min(fr_points, 100))

    return {
        "last_spread_2y": last_spread_2y,
        "last_spread_3m": last_spread_3m,
        "last_hy": last_hy,
        "last_unrate": last_unrate,
        "last_sahm": last_sahm,
        "last_claims": last_claims,
        "last_claims_yoy": last_claims_yoy,
        "inv_2y_months": inv_2y_months,
        "inv_3m_months": inv_3m_months,
        "lei_6m_ann": lei_6m_ann,
        "claims_z": claims_z,
        "hy_pct": hy_pct,
        "recession_probability": recession_probability,
        "forward_risk_score": forward_risk_score,
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Macro Dials Forward")
st.sidebar.caption("Test-App mit Forward Risk Overlay")

default_api_key = resolve_fred_api_key()
api_key = st.sidebar.text_input("FRED API Key", value=default_api_key, type="password")
st.sidebar.markdown("FRED Key holen: FRED Account → API Keys")

st.sidebar.divider()
st.sidebar.subheader("Serien Teil 1")
series_real_yield = st.sidebar.text_input("10Y Real Yield", "DFII10")
series_fed_funds = st.sidebar.text_input("Fed Funds Rate", "FEDFUNDS")
series_cpi = st.sidebar.text_input("CPI", "CPIAUCSL")
series_unrate = st.sidebar.text_input("Unemployment Rate", "UNRATE")

st.sidebar.divider()
st.sidebar.subheader("Thresholds Teil 1")
ry_low = st.sidebar.number_input("Real Yield low", value=1.0, step=0.1)
ry_high = st.sidebar.number_input("Real Yield high", value=2.0, step=0.1)
ff_low = st.sidebar.number_input("Fed Funds low", value=2.0, step=0.25)
ff_high = st.sidebar.number_input("Fed Funds high", value=4.0, step=0.25)
cpi_low = st.sidebar.number_input("CPI YoY low", value=3.0, step=0.1)
cpi_high = st.sidebar.number_input("CPI YoY high", value=4.0, step=0.1)
u_low = st.sidebar.number_input("Unemployment low", value=4.5, step=0.1)
u_high = st.sidebar.number_input("Unemployment high", value=5.5, step=0.1)

st.sidebar.divider()
macro_start_dt = st.sidebar.date_input("Historie Startdatum", value=pd.to_datetime("2000-01-01"))
macro_start = pd.to_datetime(macro_start_dt).strftime("%Y-%m-%d")

if not api_key:
    st.warning("Bitte links deinen FRED API Key eingeben.")
    st.stop()


# =========================================================
# MAIN — TEIL 1
# =========================================================

st.title("Macro Dials Forward — Test App")
st.caption("Nowcast + Liquidity/Risk + Recession + Forward Recession Risk")

with st.spinner("Lade FRED-Daten…"):
    real_yield = fred_series(api_key, series_real_yield, start=macro_start)
    fedfunds = fred_series(api_key, series_fed_funds, start=macro_start)
    cpi = fred_series(api_key, series_cpi, start=macro_start)
    unrate = fred_series(api_key, series_unrate, start=macro_start)

cpi_yoy = (cpi.pct_change(12) * 100).dropna()

ry = latest_value(real_yield)
ff = latest_value(fedfunds)
cy = latest_value(cpi_yoy)
ur = latest_value(unrate)

score_ry = score_thresholds(ry, ry_low, ry_high, invert=False)
score_ff = score_thresholds(ff, ff_low, ff_high, invert=False)
score_cy = score_thresholds(cy, cpi_low, cpi_high, invert=False)
score_ur = score_thresholds(ur, u_low, u_high, invert=False)

total = score_ry + score_ff + score_cy + score_ur
regime = "🟢 Risk-On" if total >= 2 else "🟡 Transition" if total >= 0 else "🔴 Risk-Off"

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
c1.metric("10Y Real Yield", f"{ry:.2f}%" if not pd.isna(ry) else "—", pill(score_ry))
c2.metric("Fed Funds", f"{ff:.2f}%" if not pd.isna(ff) else "—", pill(score_ff))
c3.metric("CPI YoY", f"{cy:.2f}%" if not pd.isna(cy) else "—", pill(score_cy))
c4.metric("Unemployment", f"{ur:.2f}%" if not pd.isna(ur) else "—", pill(score_ur))
c5.metric("Macro Regime", regime, f"Total Score: {total:+d}")

st.divider()
left, right = st.columns(2)
with left:
    plot_series("10Y Real Yield (DFII10)", real_yield)
    plot_series("Fed Funds Rate (FEDFUNDS)", fedfunds)
with right:
    plot_series("CPI YoY % (derived from CPIAUCSL)", cpi_yoy)
    plot_series("Unemployment Rate (UNRATE)", unrate)


# =========================================================
# MAIN — TEIL 2
# =========================================================

st.markdown("---")
st.header("Teil 2: Liquidity-Dial + Risk-Dial + Matrix")

walcl = fred_series(api_key, "WALCL", start=macro_start)
rrp = fred_series(api_key, "RRPONTSYD", start=macro_start)
tga = fred_series(api_key, "WTREGEN", start=macro_start)
vix = fred_series(api_key, "VIXCLS", start=macro_start)
hy = fred_series(api_key, "BAMLH0A0HYM2", start=macro_start)

walcl_w = to_weekly(walcl)
rrp_w = to_weekly(rrp)
tga_w = to_weekly(tga)

liq_imp = (walcl_w.diff() - rrp_w.diff() - tga_w.diff()).dropna()
liq_imp_smooth = liq_imp.rolling(8).mean().dropna()

lookback_years = 5
lb = int(lookback_years * 52)

liq_score = percentile_score(liq_imp_smooth, lookback=lb)

risk_df = pd.DataFrame({
    "vix": to_weekly(vix),
    "hy": to_weekly(hy),
}).dropna()

risk_combo = (risk_df["vix"].rank(pct=True) + risk_df["hy"].rank(pct=True)) / 2.0
risk_window = risk_combo.iloc[-lb:] if len(risk_combo) > lb else risk_combo
risk_score = float(risk_window.iloc[-1] * 100.0) if len(risk_window) else np.nan

regime2, dot2 = classify_regime(liq_score, risk_score)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.plotly_chart(make_gauge("Liquidity (Percentile)", liq_score, good_high=True), use_container_width=True)
with c2:
    st.plotly_chart(make_gauge("Risk (Percentile)", risk_score, good_high=False), use_container_width=True)
with c3:
    st.metric("Regime (Teil 2)", f"{dot2} {regime2}")

st.subheader("Liquidity × Risk Matrix")
matrix = {
    "Risk Low": {"Liq Low": "🟨", "Liq High": "🟩"},
    "Risk High": {"Liq Low": "🟥", "Liq High": "🟨"},
}
st.table(matrix)

st.subheader("Liquidity Impulse (ΔWALCL − ΔRRP − ΔTGA)")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(liq_imp_smooth.index, liq_imp_smooth.values)
ax.set_title("Weekly Liquidity Impulse (8W MA)")
ax.grid(True, alpha=0.3)
st.pyplot(fig, clear_figure=True)


# =========================================================
# MAIN — RECESSION PANEL
# =========================================================

st.markdown("---")
st.header("Rezessions-Indikatoren")

df = make_recession_df(api_key, start=macro_start)

last_spread = last_valid(df["10Y-2Y Spread"])
last_unrate = last_valid(df["Unemployment Rate"])
last_sahm = last_valid(df["SAHM"])
last_icsa = last_valid(df["Initial Claims"])

def fmt_num(x, digits=2, suffix=""):
    return "n/a" if pd.isna(x) else f"{x:.{digits}f}{suffix}"

def fmt_int(x):
    return "n/a" if pd.isna(x) else f"{int(x):,}".replace(",", " ")

c1, c2, c3, c4 = st.columns(4)
c1.metric("10Y-2Y Spread", fmt_num(last_spread, 2, " %-Pkt"))
c2.metric("Arbeitslosigkeit (UNRATE)", fmt_num(last_unrate, 1, " %"))
c3.metric("Sahm Rule", fmt_num(last_sahm, 2, " %-Pkt"))
c4.metric("Initial Claims", fmt_int(last_icsa))

risk_notes = []
if not pd.isna(last_spread) and last_spread < 0:
    risk_notes.append("🔴 Zinskurve invertiert (Spread < 0)")
else:
    risk_notes.append("🟢 Zinskurve nicht invertiert")

if not pd.isna(last_sahm) and last_sahm >= 0.50:
    risk_notes.append("🔴 Sahm Rule >= 0.50 (klassischer Rezessions-Trigger)")
elif not pd.isna(last_sahm) and last_sahm >= 0.35:
    risk_notes.append("🟠 Sahm Rule erhöht (>= 0.35)")
else:
    risk_notes.append("🟢 Sahm Rule niedrig")

st.write("**Signal-Check (grob):**")
for note in risk_notes:
    st.write(note)

st.divider()
st.write("### Charts")
st.line_chart(df[["10Y-2Y Spread"]].dropna())
st.line_chart(df[["Unemployment Rate", "SAHM"]].dropna())
st.line_chart(df[["Initial Claims"]].dropna())
st.line_chart(df[["Leading Index (USSLIND)"]].dropna())


# =========================================================
# MAIN — FORWARD RECESSION RISK PANEL
# =========================================================

st.markdown("---")
st.header("Forward Recession Risk")

daily, monthly, weekly = build_forward_recession_data(api_key, start=macro_start)
stats = compute_forward_recession_scores(daily, monthly, weekly)

prob = stats["recession_probability"]
fwd = stats["forward_risk_score"]

def bucket_label(x):
    if pd.isna(x):
        return "⚪ Unknown"
    if x < 30:
        return "🟢 Low"
    if x < 60:
        return "🟡 Medium"
    return "🔴 High"

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.plotly_chart(make_gauge("Recession Probability", prob, good_high=False), use_container_width=True)
with c2:
    st.plotly_chart(make_gauge("Forward Risk Score", fwd, good_high=False), use_container_width=True)
with c3:
    st.metric("Interpretation", bucket_label(prob), f"Forward Risk: {bucket_label(fwd)}")

st.subheader("Treiber")
a1, a2, a3, a4 = st.columns(4)
a1.metric("10Y-2Y", f"{stats['last_spread_2y']:.2f} %-Pkt" if not np.isnan(stats["last_spread_2y"]) else "n/a",
          f"Inversion: {stats['inv_2y_months']}M")
a2.metric("10Y-3M", f"{stats['last_spread_3m']:.2f} %-Pkt" if not np.isnan(stats["last_spread_3m"]) else "n/a",
          f"Inversion: {stats['inv_3m_months']}M")
a3.metric("HY OAS", f"{stats['last_hy']:.2f} %" if not np.isnan(stats["last_hy"]) else "n/a",
          f"PctRank: {stats['hy_pct']:.0f}" if not np.isnan(stats["hy_pct"]) else "PctRank: n/a")
a4.metric("LEI 6M ann.", f"{stats['lei_6m_ann']:.1f} %" if not np.isnan(stats["lei_6m_ann"]) else "n/a",
          "Leading Trend")

b1, b2, b3 = st.columns(3)
b1.metric("Sahm Rule", f"{stats['last_sahm']:.2f}" if not np.isnan(stats["last_sahm"]) else "n/a", "Bestätiger")
b2.metric("Initial Claims", f"{int(stats['last_claims']):,}".replace(",", " ") if not np.isnan(stats["last_claims"]) else "n/a",
          f"YoY: {stats['last_claims_yoy']:.1f}%" if not np.isnan(stats["last_claims_yoy"]) else "YoY: n/a")
b3.metric("Unemployment", f"{stats['last_unrate']:.1f} %" if not np.isnan(stats["last_unrate"]) else "n/a", "Arbeitsmarkt")

notes = []
if stats["inv_3m_months"] >= 6:
    notes.append("🔶 10Y–3M ist seit mehreren Monaten invertiert → klassischer Frühwarnhinweis.")
if not np.isnan(stats["lei_6m_ann"]) and stats["lei_6m_ann"] < 0:
    notes.append("🔶 Leading Index fällt auf 6M-Basis → Vorlaufrisiko steigt.")
if not np.isnan(stats["last_hy"]) and stats["last_hy"] > 4.5:
    notes.append("🔶 Credit Spreads sind erhöht → Finanzierungsstress nimmt zu.")
if not np.isnan(stats["last_sahm"]) and stats["last_sahm"] < 0.35:
    notes.append("🟢 Sahm Rule noch niedrig → Rezession noch nicht bestätigt.")
if not np.isnan(stats["last_claims_yoy"]) and stats["last_claims_yoy"] <= 5:
    notes.append("🟢 Initial Claims zeigen noch keinen klaren Arbeitsmarktbruch.")
if not notes:
    notes.append("🟡 Gemischtes Bild: Noch keine harte Bestätigung, aber einzelne Vorlaufrisiken vorhanden.")

st.write("**Signal-Check:**")
for n in notes:
    st.write(n)

st.subheader("Charts")
ch1, ch2 = st.columns(2)

with ch1:
    st.line_chart(daily[["spread_10y_2y", "spread_10y_3m"]].dropna())
    st.caption("Yield Curve: 10Y–2Y und 10Y–3M")
    st.line_chart(monthly[["lei"]].dropna())
    st.caption("Leading Index (USSLIND)")

with ch2:
    st.line_chart(monthly[["unrate", "sahm"]].dropna())
    st.caption("Arbeitsmarkt + Sahm Rule")
    st.line_chart(weekly[["icsa", "icsa_yoy"]].dropna())
    st.caption("Initial Claims + YoY Trend")

with st.expander("Forward-Recession-Daten anzeigen"):
    preview = pd.DataFrame({
        "10Y-2Y": daily["spread_10y_2y"].tail(20),
        "10Y-3M": daily["spread_10y_3m"].tail(20),
        "HY OAS": daily["hy_oas"].tail(20),
    })
    st.dataframe(preview)

st.markdown("---")
st.info(
    "Diese Test-App trennt bewusst zwischen aktuellem Zustand und Vorlaufrisiken. "
    "So siehst du sauber den Unterschied zwischen bestätigter Schwäche und steigendem Rezessionsrisiko."
)
