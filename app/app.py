"""
app.py  Telecom Churn · Survival Analysis

Design system: Data Terminal.
Near-black ground, JetBrains Mono throughout, neon signal scale (cyan protective /
amber elevated / magenta hazard), big monospace readouts, panel grid, // section
markers, zero rounding. A BI cockpit, not an editorial page.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens · data terminal
# ─────────────────────────────────────────────────────────────────────────────
BONE   = "#0A0E14"   # near-black terminal ground
PANEL  = "#111A24"   # panel ground
PANEL2 = "#0E1620"   # deeper inset
INK    = "#E8F0F6"   # light readout text / values
INK60  = "#7E92A6"   # muted label text
RULE   = "#22323F"   # divider
LINE   = "#1A2530"   # faint grid
BORDER = "#223040"   # panel border
NAVY   = "#8B9CFF"   # violet secondary series

PROTECT = "#2BE8C8"  # cyan · protective / low risk / high survival
CAUTION = "#FFC23D"  # amber · elevated
HAZARD  = "#FF4D8D"  # magenta · high churn risk / hazard
NEUTRAL = "#4A5B6B"  # grey · non-significant

# Chart series: white, magenta, cyan, amber, violet, grey. Neon on dark.
SERIES = [INK, HAZARD, PROTECT, CAUTION, NAVY, NEUTRAL]

# Diverging scale for survival heatmaps: hazard(low survival) → mid → protect(high)
SURV_SCALE = [[0.0, HAZARD], [0.5, "#243240"], [1.0, PROTECT]]
# Diverging scale for SHAP feature value (low → high): protect → mid → hazard
SHAP_SCALE = [[0.0, PROTECT], [0.5, "#243240"], [1.0, HAZARD]]

PLOTLY_TEMPLATE = go.layout.Template(layout=dict(
    xaxis=dict(gridcolor=LINE, zerolinecolor="#243240", linecolor=BORDER, ticks="outside",
               tickcolor=BORDER, tickfont=dict(family="JetBrains Mono, monospace", size=11, color=INK60)),
    yaxis=dict(gridcolor=LINE, zerolinecolor="#243240", linecolor=BORDER, ticks="outside",
               tickcolor=BORDER, tickfont=dict(family="JetBrains Mono, monospace", size=11, color=INK60)),
))

def layout(**kw):
    """Base plotly layout on the terminal palette. Pass overrides via kw."""
    base = dict(
        plot_bgcolor=PANEL, paper_bgcolor=PANEL,
        font=dict(color=INK60, family="JetBrains Mono, monospace", size=12),
        colorway=SERIES, template=PLOTLY_TEMPLATE,
        title=dict(font=dict(family="JetBrains Mono, monospace", size=13, color=INK)),
        margin=dict(l=60, r=30, t=56, b=60),
        legend=dict(font=dict(size=11)),
    )
    base.update(kw)
    return base


def risk_state(prob: float):
    """Map a churn probability to (label, colour, tint)."""
    if prob >= 0.60:
        return "HIGH RISK", HAZARD, "rgba(255,77,141,0.12)"
    if prob >= 0.40:
        return "ELEVATED", CAUTION, "rgba(255,194,61,0.12)"
    return "LOW RISK", PROTECT, "rgba(43,232,200,0.10)"


# ─────────────────────────────────────────────────────────────────────────────
# Page config + global stylesheet
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHURN.term",
    page_icon="◖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

:root {{ --ink:{INK}; --bg:{BONE}; --panel:{PANEL}; --ink60:{INK60}; --bd:{BORDER};
  --hazard:{HAZARD}; --protect:{PROTECT}; }}

html, body, [class*="css"], .stMarkdown, p, li, span, div {{
  font-family:'JetBrains Mono', ui-monospace, monospace; }}
.stApp {{ background:{BONE}; color:{INK}; }}
.stApp *, [data-testid] * {{ border-radius:0 !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
.block-container {{ max-width:1240px; padding-top:2.8rem; padding-bottom:4rem; }}

/* headings · terminal display, big numerals */
h1 {{ font-weight:800; letter-spacing:-0.02em; line-height:0.96;
  font-size:clamp(2.6rem,5.2vw,4.4rem); color:{INK}; margin:0.6rem 0 0.3rem 0; }}
h2 {{ font-weight:700; letter-spacing:-0.01em; font-size:1.4rem; color:{INK}; margin:0.4rem 0 0.7rem 0; }}
h3 {{ font-weight:700; font-size:1.0rem; color:{INK}; }}
h4 {{ font-weight:600; text-transform:uppercase; letter-spacing:0.16em;
  font-size:0.68rem; color:{PROTECT}; }}

/* masthead · status bar */
.masthead {{ display:flex; align-items:center; justify-content:space-between;
  border-bottom:1px solid {BORDER}; padding:0.4rem 0 0.5rem; margin-bottom:0.2rem; }}
.masthead .brand {{ font-weight:800; letter-spacing:0.14em; text-transform:uppercase;
  font-size:0.74rem; color:{PROTECT}; }}
.masthead .meta {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.10em; color:{INK60}; }}
.masthead .meta b {{ color:{INK}; }} .masthead .dot {{ color:{HAZARD}; }}

/* top nav · tab cells */
div[data-testid="stRadio"] > label {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] {{
  display:flex; gap:0; border:1px solid {BORDER}; margin:0.9rem 0 1.8rem; flex-wrap:wrap; background:{PANEL}; }}
div[data-testid="stRadio"] [role="radiogroup"] label {{
  margin:0; padding:0; cursor:pointer; border-right:1px solid {BORDER}; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:last-child p {{
  font-weight:600; text-transform:uppercase; letter-spacing:0.10em;
  font-size:0.72rem; color:{INK60}; padding:0.55rem 1.3rem; margin:0; }}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) > div:last-child p {{
  color:{BONE}; background:{PROTECT}; }}

/* lede */
.lede {{ font-size:1.0rem; line-height:1.65; color:{INK60}; max-width:74ch; font-weight:400; }}
.lede b {{ color:{INK}; font-weight:600; }} .lede i {{ color:{PROTECT}; font-style:normal; }}

/* KPI readout grid */
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:{BORDER};
  border:1px solid {BORDER}; margin:1.4rem 0; }}
.kpi {{ background:{PANEL}; padding:1.1rem 1.2rem; }}
.kpi .v {{ font-weight:800; font-size:2.5rem; line-height:1; color:{INK}; letter-spacing:-0.02em; }}
.kpi .v.haz {{ color:{HAZARD}; }} .kpi .v.pro {{ color:{PROTECT}; }}
.kpi .k {{ font-size:0.62rem; text-transform:uppercase; letter-spacing:0.14em;
  color:{INK60}; margin-top:0.6rem; }}
.kpi .s {{ font-size:0.64rem; color:{PROTECT}; margin-top:0.2rem; }}

/* method / step grid */
.steps {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:{BORDER};
  border:1px solid {BORDER}; margin-top:0.4rem; }}
.step {{ background:{PANEL}; padding:1.1rem 1.15rem; }}
.step .n {{ font-weight:700; font-size:0.66rem; color:{HAZARD}; letter-spacing:0.10em; }}
.step .t {{ color:{INK}; font-weight:700; font-size:0.92rem; margin:0.4rem 0 0.5rem 0; }}
.step .d {{ font-size:0.8rem; color:{INK60}; line-height:1.55; }}
.step .d b {{ color:{INK}; font-weight:600; }}

/* generic block */
.blk {{ border:1px solid {BORDER}; padding:1rem 1.1rem; background:{PANEL}; }}
.blk.fill {{ background:{PANEL2}; }}

/* callout */
.callout {{ border-left:3px solid {NAVY}; padding:0.65rem 0.95rem; margin:0.9rem 0;
  font-size:0.88rem; color:{INK60}; background:{PANEL}; }}
.callout b {{ color:{INK}; }}
.callout.haz {{ border-left-color:{HAZARD}; }}
.callout.pro {{ border-left-color:{PROTECT}; }}
.callout.cau {{ border-left-color:{CAUTION}; }}

/* findings */
.find {{ font-size:0.94rem; line-height:1.7; color:{INK60}; }}
.find b {{ color:{INK}; font-weight:600; }}

/* tags */
.tag {{ display:inline-block; border:1px solid {BORDER}; padding:3px 10px; margin:3px 5px 3px 0;
  font-size:0.7rem; font-weight:500; color:{INK60}; background:{PANEL}; }}
.tag.haz {{ border-color:{HAZARD}; color:{HAZARD}; }}
.tag.pro {{ border-color:{PROTECT}; color:{PROTECT}; }}

/* buttons */
.stButton > button, .stFormSubmitButton > button {{
  border:1px solid {PROTECT}; background:transparent; color:{PROTECT};
  font-family:'JetBrains Mono',monospace; font-weight:600; text-transform:uppercase;
  letter-spacing:0.10em; font-size:0.74rem; padding:0.5rem 1.4rem; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
  background:{PROTECT}; border-color:{PROTECT}; color:{BONE}; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ border-bottom:1px solid {BORDER}; gap:0; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; text-transform:uppercase;
  letter-spacing:0.08em; font-size:0.7rem; padding:0.5rem 1.1rem; color:{INK60}; }}
.stTabs [aria-selected="true"] {{ color:{PROTECT} !important; box-shadow:inset 0 -2px 0 0 {PROTECT}; }}

hr {{ border:none; border-top:1px solid {BORDER}; margin:1.6rem 0; }}
[data-testid="stDataFrame"] {{ border:1px solid {BORDER}; }}
[data-testid="stMetricValue"] {{ font-family:'JetBrains Mono',monospace; color:{INK}; }}
[data-testid="stMetricLabel"] {{ color:{INK60}; }}

/* inputs */
textarea, input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput div[data-baseweb="input"] {{
  border:1px solid {BORDER} !important; background:{PANEL} !important; color:{INK} !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{PROTECT} !important; }}
label, .stSelectbox label, .stSlider label {{ color:{INK60} !important; }}
::placeholder {{ color:{INK60} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable editorial blocks
# ─────────────────────────────────────────────────────────────────────────────
def masthead(meta):
    auc = meta["best_metrics"]["roc_auc"] if meta else 0
    st.markdown(
        f"""<div class="masthead">
        <span class="brand">Telecom Churn · Survival</span>
        <span class="meta"><b>{meta['model_name'] if meta else '·'}</b> ·
        AUC <b>{auc:.3f}</b> · N=<b>{meta['n_samples']:,}</b></span></div>""",
        unsafe_allow_html=True)


def rule(left: str, right: str = ""):
    st.markdown(
        f"""<div style="display:flex; align-items:baseline; justify-content:space-between;
        border-top:1px solid {BORDER}; padding-top:0.5rem; margin:2.2rem 0 1rem 0;">
        <span style="font-weight:700; font-size:1.0rem; color:{INK}; letter-spacing:0.02em;">
        <span style="color:{HAZARD};">// </span>{left}</span>
        <span style="font-size:0.64rem; text-transform:uppercase; letter-spacing:0.14em;
        color:{INK60};">{right}</span></div>""",
        unsafe_allow_html=True)


def callout(text: str, kind: str = ""):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def chart(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ─────────────────────────────────────────────────────────────────────────────
# Data + artifacts
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ["churn_model.pkl", "cox_model.pkl", "km_data.pkl",
                "shap_data.pkl", "model_meta.pkl"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        return None, None, None, None, None, f"Missing artifacts: {missing}"
    model     = joblib.load(MODELS_DIR / "churn_model.pkl")
    cox       = joblib.load(MODELS_DIR / "cox_model.pkl")
    km_data   = joblib.load(MODELS_DIR / "km_data.pkl")
    shap_data = joblib.load(MODELS_DIR / "shap_data.pkl")
    meta      = joblib.load(MODELS_DIR / "model_meta.pkl")
    return model, cox, km_data, shap_data, meta, None


@st.cache_data
def load_telco_csv() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    return df


def _get_lift_at(meta, frac):
    ld = meta["lift_data"]
    return float(np.interp(frac, np.array(ld["pct_targeted"]), np.array(ld["pct_churn_captured"])))


model, cox, km_data, shap_data, meta, load_error = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Header + nav
# ─────────────────────────────────────────────────────────────────────────────
masthead(meta)

if load_error:
    st.error(f"Could not load model artifacts. {load_error}")
    st.info("Run `python src/train.py` from the project root to train models first.")
    st.stop()

NAV = ["Brief", "Survival", "Model & Drivers", "Simulator", "Impact"]
section = st.radio("nav", NAV, horizontal=True, label_visibility="collapsed")


# ═════════════════════════════════════════════════════════════════════════════
# 01 · BRIEF
# ═════════════════════════════════════════════════════════════════════════════
if section == "Brief":
    df = load_telco_csv()
    churned = df[df["Churn_bin"] == 1]
    retained = df[df["Churn_bin"] == 0]
    best_auc = meta["best_metrics"]["roc_auc"]

    st.markdown("<h4>Case file 01 · The brief</h4>", unsafe_allow_html=True)
    st.markdown("# When, not if.")
    st.markdown(
        '<p class="lede">Most churn models answer <i>will this customer leave?</i> · a binary '
        'yes/no. This one answers <b>when</b>, using survival analysis, so a retention team can '
        'triage by <b>urgency</b>, not just likelihood.</p>', unsafe_allow_html=True)

    tenure_gap = churned["tenure"].mean() - retained["tenure"].mean()
    st.markdown(f"""
    <div class="kpis">
      <div class="kpi"><div class="v">{len(df):,}</div><div class="k">Customers</div></div>
      <div class="kpi"><div class="v haz">{meta['churn_rate']:.1%}</div><div class="k">Churn rate</div></div>
      <div class="kpi"><div class="v">{churned['tenure'].mean():.0f} mo</div><div class="k">Avg tenure · churned</div>
        <div class="s">{tenure_gap:+.0f} mo vs retained</div></div>
      <div class="kpi"><div class="v pro">{best_auc:.3f}</div><div class="k">Best AUC-ROC</div></div>
    </div>""", unsafe_allow_html=True)

    rule("The problem", "telecom retention")
    st.markdown(
        '<p class="lede">Acquiring a new telecom customer costs <b>5–25×</b> more than keeping '
        'one, yet sector churn runs <b>15–25%</b> a year. Knowing <i>who</i> is at risk is only '
        'half the lever · knowing <i>when</i> the risk lands is what lets you spend the retention '
        'budget on the customers about to walk, this quarter, not next year.</p>',
        unsafe_allow_html=True)

    rule("Method", "three lenses")
    st.markdown("""
    <div class="steps">
      <div class="step"><div class="n">A · SURVIVAL</div>
        <div class="t">Time-to-event</div>
        <div class="d">Kaplan–Meier curves by segment, log-rank tests, and a
        <b>Cox proportional-hazards</b> model with hazard ratios and CIs. Active customers
        enter as right-censored.</div></div>
      <div class="step"><div class="n">B · CLASSIFY</div>
        <div class="t">Rank by risk</div>
        <div class="d">Logistic Regression, Random Forest and <b>XGBoost</b> benchmarked,
        chosen by AUC-ROC on a held-out test set. Native <b>SHAP</b> contributions for
        global and per-customer attribution.</div></div>
      <div class="step"><div class="n">C · IMPACT</div>
        <div class="t">Money</div>
        <div class="d">Lift curve versus random targeting, an ROI calculator with adjustable
        offer cost and save rate, and a sensitivity sweep for the <b>optimal targeting
        depth</b>.</div></div>
    </div>""", unsafe_allow_html=True)

    rule("What the data says", "key findings")
    _mtm = km_data.get("Contract", {}).get("Month-to-month", {}).get("median")
    _mtm_str = f"~{_mtm:.0f} months" if _mtm else "markedly shorter"
    cmed = df["MonthlyCharges"].median()
    hi = df[df["MonthlyCharges"] > cmed]["Churn_bin"].mean()
    lo = df[df["MonthlyCharges"] <= cmed]["Churn_bin"].mean()
    lift20 = _get_lift_at(meta, 0.20)
    st.markdown(f"""<div class="find">
    <p>① <b>Contract type</b> is the single strongest churn signal · month-to-month customers
    leave at dramatically higher rates than 1- or 2-year holders.</p>
    <p>② <b>Median survival</b> for month-to-month is {_mtm_str}; annual-contract holders never
    reach 50% churn inside the 72-month window.</p>
    <p>③ Targeting the <b>top 20% by churn score captures ~{lift20:.0%} of all churners</b> ·
    a {lift20/0.20:.1f}× lift over random outreach.</p>
    <p>④ <b>Online Security</b> and <b>Tech Support</b> are the strongest protective factors in
    the Cox model: customers with them carry meaningfully lower hazard.</p>
    <p>⑤ Customers billed above the median (${cmed:.0f}/mo) churn at <b>{hi:.0%}</b> versus
    <b>{lo:.0%}</b> below it.</p></div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 02 · SURVIVAL
# ═════════════════════════════════════════════════════════════════════════════
elif section == "Survival":
    st.markdown("<h4>Case file 02 · Time to event</h4>", unsafe_allow_html=True)
    st.markdown("# Survival")
    st.markdown(
        '<p class="lede">How long customers last before they churn · and how that curve splits '
        'by contract, service and tenure. Curves estimated with Kaplan–Meier; group differences '
        'tested with the log-rank statistic; covariate effects from a Cox model.</p>',
        unsafe_allow_html=True)

    # ── Kaplan-Meier ──────────────────────────────────────────────────────────
    rule("Kaplan–Meier curves", "survival by segment")
    seg_options = {
        "Contract Type": "Contract",
        "Internet Service": "InternetService",
        "Payment Method": "PaymentMethod",
        "Senior Status": "SeniorCitizen",
    }
    seg_label = st.selectbox("Segment", list(seg_options.keys()), label_visibility="collapsed")
    seg_col = seg_options[seg_label]
    seg_data = km_data[seg_col]
    pvalue = seg_data.get("_logrank_pvalue", float("nan"))
    groups = {k: v for k, v in seg_data.items() if not k.startswith("_")}

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[j:j+2], 16) for j in (0, 2, 4))

    fig = go.Figure()
    for i, (label, g) in enumerate(groups.items()):
        c = SERIES[i % len(SERIES)]
        r, gr, b = hex_to_rgb(c)
        fig.add_trace(go.Scatter(
            x=g["timeline"] + g["timeline"][::-1],
            y=g["ci_upper"] + g["ci_lower"][::-1],
            fill="toself", fillcolor=f"rgba({r},{gr},{b},0.10)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
        median = g["median"]
        leg = f"{label} · n={g['n']:,} · churn {g['churn_rate']:.0%}"
        if median is not None:
            leg += f" · median {median:.0f}mo"
        fig.add_trace(go.Scatter(
            x=g["timeline"], y=g["survival"], name=leg,
            line=dict(color=c, width=2.4), mode="lines",
            hovertemplate="<b>%{fullData.name}</b><br>Month %{x}<br>Survival %{y:.1%}<extra></extra>"))
        if median is not None:
            fig.add_shape(type="line", x0=median, x1=median, y0=0, y1=0.5,
                          line=dict(color=c, dash="dot", width=1))
            fig.add_shape(type="line", x0=0, x1=median, y0=0.5, y1=0.5,
                          line=dict(color=c, dash="dot", width=1))
    pval_str = f"{pvalue:.1e}" if not np.isnan(pvalue) else "N/A"
    fig.add_hline(y=0.5, line_dash="dot", line_color=NEUTRAL, annotation_text="50% survival")
    fig.update_layout(**layout(
        title=f"Survival function by {seg_label} · log-rank p = {pval_str}",
        xaxis_title="Months as customer", yaxis_title="Survival probability",
        yaxis=dict(tickformat=".0%", range=[0, 1.04], gridcolor=LINE),
        hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.20, font=dict(size=11)),
        margin=dict(l=60, r=30, t=56, b=120)))
    chart(fig)

    rows = [{
        "Group": label, "N": f"{g['n']:,}", "Churned": f"{g['n_events']:,}",
        "Churn rate": f"{g['churn_rate']:.1%}",
        "Median survival": f"{g['median']:.0f} mo" if g["median"] is not None else ">72 mo",
    } for label, g in groups.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if not np.isnan(pvalue):
        if pvalue < 0.001:
            callout(f"<b>Log-rank p &lt; 0.001.</b> Survival curves differ substantially between "
                    f"{seg_label.lower()} groups · this split carries real signal.", "pro")
        elif pvalue < 0.05:
            callout(f"<b>Log-rank p = {pvalue:.4f}.</b> Groups show significantly different "
                    f"survival patterns.", "cau")
        else:
            callout(f"<b>Log-rank p = {pvalue:.4f}.</b> No significant difference · these groups "
                    f"survive alike.", "")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    rule("Survival heatmap", "contract × internet")
    st.markdown('<p class="lede">Survival probability at fixed time points, for the largest '
                'Contract × Internet Service cells.</p>', unsafe_allow_html=True)

    @st.cache_data
    def build_heatmap_data():
        from lifelines import KaplanMeierFitter
        df = load_telco_csv()
        tps = [6, 12, 24, 36, 48, 72]
        combos = []
        for ct in df["Contract"].unique():
            for it in df["InternetService"].unique():
                mask = (df["Contract"] == ct) & (df["InternetService"] == it)
                if mask.sum() >= 30:
                    combos.append((ct, it, int(mask.sum())))
        combos.sort(key=lambda x: -x[2])
        combos = combos[:8]
        rows, labels = [], []
        for ct, it, n in combos:
            sub = df[(df["Contract"] == ct) & (df["InternetService"] == it)]
            kmf = KaplanMeierFitter().fit(sub["tenure"], event_observed=sub["Churn_bin"])
            row = []
            for t in tps:
                try:
                    row.append(round(float(kmf.survival_function_at_times([t]).values[0]), 4))
                except Exception:
                    row.append(np.nan)
            rows.append(row)
            labels.append(f"{ct} · {it} (n={n})")
        return np.array(rows), labels, tps

    hm, hm_labels, tps = build_heatmap_data()
    fig_hm = px.imshow(hm, x=[f"{t}mo" for t in tps], y=hm_labels,
                       color_continuous_scale=SURV_SCALE, zmin=0, zmax=1,
                       text_auto=".0%", aspect="auto")
    fig_hm.update_traces(textfont=dict(family="IBM Plex Mono, monospace", size=11))
    fig_hm.update_layout(**layout(
        title="Survival probability by segment and time",
        xaxis_title="Time point", yaxis_title="",
        yaxis=dict(automargin=True), height=430,
        margin=dict(l=10, r=80, t=56, b=50),
        coloraxis_colorbar=dict(title="Survival", tickformat=".0%")))
    chart(fig_hm)
    callout("Cyan = high survival (low churn risk). Magenta = low survival (high churn risk).", "")

    # ── Cox forest ────────────────────────────────────────────────────────────
    rule("Cox proportional hazards", "hazard ratios")
    st.markdown('<p class="lede">Each covariate\'s hazard ratio with a 95% CI. <b>HR &gt; 1</b> '
                'raises churn risk; <b>HR &lt; 1</b> protects.</p>', unsafe_allow_html=True)

    summary = cox.summary.copy()
    filtered = summary[(summary["p"].fillna(1) < 0.05) | (summary["coef"].abs() > 0.05)].copy()
    filtered = filtered.sort_values("coef")

    fig_cox = go.Figure()
    for covar, row in filtered.iterrows():
        hr = float(np.exp(row["coef"]))
        lo_hr = float(np.exp(row["coef lower 95%"]))
        hi_hr = float(np.exp(row["coef upper 95%"]))
        p = float(row["p"])
        sig = p < 0.05
        col = HAZARD if (hr > 1 and sig) else PROTECT if (hr < 1 and sig) else NEUTRAL
        fig_cox.add_trace(go.Scatter(
            x=[hr], y=[str(covar)], mode="markers",
            marker=dict(color=col, size=9, symbol="square"),
            error_x=dict(type="data", symmetric=False, array=[hi_hr - hr],
                         arrayminus=[hr - lo_hr], color=col, thickness=1.8, width=7),
            showlegend=False,
            hovertemplate=f"<b>{covar}</b><br>HR {hr:.3f}<br>95% CI [{lo_hr:.3f}, {hi_hr:.3f}]"
                          f"<br>p={p:.4f}<extra></extra>"))
    fig_cox.add_vline(x=1.0, line_dash="dash", line_color=NEUTRAL, annotation_text="HR = 1")
    fig_cox.update_layout(**layout(
        title="Hazard ratios · Cox PH model", xaxis_title="Hazard ratio (log scale)",
        xaxis_type="log", yaxis=dict(automargin=True),
        height=max(420, len(filtered) * 30 + 120), margin=dict(l=10, r=30, t=56, b=50)))
    chart(fig_cox)

    risk_f = filtered[filtered["coef"] > 0].sort_values("coef", ascending=False).head(3)
    prot_f = filtered[filtered["coef"] < 0].sort_values("coef").head(3)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4>Top risk factors (HR &gt; 1)</h4>", unsafe_allow_html=True)
        for covar, row in risk_f.iterrows():
            st.markdown(f'<span class="tag haz">{covar} · HR {np.exp(row["coef"]):.2f}</span>',
                        unsafe_allow_html=True)
    with c2:
        st.markdown("<h4>Top protective factors (HR &lt; 1)</h4>", unsafe_allow_html=True)
        for covar, row in prot_f.iterrows():
            st.markdown(f'<span class="tag pro">{covar} · HR {np.exp(row["coef"]):.2f}</span>',
                        unsafe_allow_html=True)

    with st.expander("Full Cox PH summary table"):
        d = summary[["coef", "exp(coef)", "coef lower 95%", "coef upper 95%", "p"]].copy()
        d.columns = ["coef", "HR", "CI low", "CI high", "p"]
        d["HR"] = d["HR"].round(3)
        d["CI low"] = np.exp(d["CI low"]).round(3)
        d["CI high"] = np.exp(d["CI high"]).round(3)
        d["p"] = d["p"].map(lambda x: f"{x:.4f}")
        st.dataframe(d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# 03 · MODEL & DRIVERS
# ═════════════════════════════════════════════════════════════════════════════
elif section == "Model & Drivers":
    st.markdown("<h4>Case file 03 · How good, and why</h4>", unsafe_allow_html=True)
    st.markdown("# Model & drivers")
    st.markdown('<p class="lede">Three classifiers benchmarked on a held-out test set, then the '
                'winner opened up with SHAP to show <b>which features move each prediction</b>.</p>',
                unsafe_allow_html=True)

    all_metrics = meta["metrics"]
    best_name = meta["model_name"]
    churn_rate = meta["churn_rate"]
    names = list(all_metrics.keys())
    mcol = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}

    rule("Discrimination", "ROC & precision–recall")
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=NEUTRAL, dash="dash"), name="Random (0.500)"))
        for n in names:
            m = all_metrics[n]
            fig.add_trace(go.Scatter(x=m["roc_curve"]["fpr"], y=m["roc_curve"]["tpr"], mode="lines",
                          name=f"{n} ({m['roc_auc']:.3f})" + (" ◆" if n == best_name else ""),
                          line=dict(color=mcol[n], width=3 if n == best_name else 1.5)))
        fig.update_layout(**layout(title="ROC curves", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=420,
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10))))
        chart(fig)
    with cc2:
        fig = go.Figure()
        fig.add_hline(y=churn_rate, line_dash="dash", line_color=NEUTRAL,
                      annotation_text=f"Baseline {churn_rate:.0%}")
        for n in names:
            m = all_metrics[n]
            fig.add_trace(go.Scatter(x=m["pr_curve"]["recall"], y=m["pr_curve"]["precision"],
                          mode="lines", name=f"{n} ({m['avg_precision']:.3f})" + (" ◆" if n == best_name else ""),
                          line=dict(color=mcol[n], width=3 if n == best_name else 1.5)))
        fig.update_layout(**layout(title="Precision–recall curves", xaxis_title="Recall",
                          yaxis_title="Precision", height=420,
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10))))
        chart(fig)

    rule("Calibration & comparison", "do the probabilities mean anything")
    cc3, cc4 = st.columns([3, 2])
    with cc3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=NEUTRAL, dash="dash"), name="Perfect"))
        for n in names:
            cal = all_metrics[n]["calibration_curve"]
            fig.add_trace(go.Scatter(x=cal["mean_pred"], y=cal["frac_pos"], mode="lines+markers",
                          name=n + (" ◆" if n == best_name else ""),
                          line=dict(color=mcol[n], width=3 if n == best_name else 1.5),
                          marker=dict(size=7)))
        fig.update_layout(**layout(title="Calibration · predicted vs observed",
                          xaxis_title="Mean predicted probability", yaxis_title="Fraction positive",
                          height=400, legend=dict(orientation="h", y=-0.25, font=dict(size=10))))
        chart(fig)
    with cc4:
        rows = [{
            "Model": ("◆ " if n == best_name else "") + n,
            "AUC": f"{all_metrics[n]['roc_auc']:.3f}",
            "Recall": f"{all_metrics[n]['recall']:.3f}",
            "F1": f"{all_metrics[n]['f1']:.3f}",
            "AP": f"{all_metrics[n]['avg_precision']:.3f}",
        } for n in names]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        callout(f"<b>{best_name}</b> wins on AUC and average precision and stays well-calibrated "
                f"· so its scores can be read as genuine churn probabilities, not just a ranking.",
                "pro")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if shap_data is not None:
        shap_values = shap_data["shap_values"]
        feature_names = shap_data["feature_names"]
        X_test_prep = shap_data["X_test_prep"]

        rule("What drives the score", "SHAP attribution")
        mean_abs = np.abs(shap_values).mean(axis=0)

        sc1, sc2 = st.columns([3, 2])
        with sc1:
            top_n = 14
            idx = np.argsort(mean_abs)[-top_n:][::-1]
            rng = np.random.default_rng(0)
            fig = go.Figure()
            for plot_i, fi in enumerate(idx):
                sv = shap_values[:, fi]
                fv = X_test_prep[:, fi]
                fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
                fig.add_trace(go.Scatter(
                    x=sv, y=plot_i + rng.uniform(-0.34, 0.34, len(sv)), mode="markers",
                    marker=dict(color=fv_norm, colorscale=SHAP_SCALE, size=4, opacity=0.7,
                                showscale=(plot_i == 0),
                                colorbar=dict(title="Feature<br>value", len=0.5, x=1.02)
                                if plot_i == 0 else None),
                    showlegend=False,
                    hovertemplate=f"<b>{feature_names[fi]}</b><br>SHAP %{{x:.4f}}<extra></extra>"))
            fig.add_vline(x=0, line_dash="dash", line_color=NEUTRAL)
            fig.update_layout(**layout(
                title="SHAP beeswarm · impact on churn prediction",
                xaxis_title="SHAP value (→ pushes toward churn)",
                yaxis=dict(tickmode="array", tickvals=list(range(top_n)),
                           ticktext=[feature_names[i] for i in idx], automargin=True),
                height=520, margin=dict(l=10, r=100, t=56, b=56)))
            chart(fig)
            st.markdown('<p style="font-size:0.84rem;color:#7E92A6;">Magenta = high feature value, '
                        'cyan = low. Right of zero pushes the prediction toward churn.</p>',
                        unsafe_allow_html=True)
        with sc2:
            top_b = 12
            ti = np.argsort(mean_abs)[-top_b:]
            fig = go.Figure(go.Bar(x=mean_abs[ti], y=[feature_names[i] for i in ti],
                                   orientation="h", marker=dict(color=INK)))
            fig.update_layout(**layout(title="Mean |SHAP| · global importance",
                              xaxis_title="Mean |SHAP|", yaxis=dict(automargin=True),
                              height=520, margin=dict(l=10, r=20, t=56, b=56)))
            chart(fig)

        with st.expander("SHAP dependence plot"):
            default_feat = feature_names[int(np.argmax(mean_abs))]
            sel = st.selectbox("Feature", feature_names, index=feature_names.index(default_feat))
            fi = feature_names.index(sel)
            sv_sel = shap_values[:, fi]
            cors = []
            for j in range(X_test_prep.shape[1]):
                if j == fi:
                    cors.append(0.0); continue
                try:
                    cors.append(abs(float(np.corrcoef(sv_sel, X_test_prep[:, j])[0, 1])))
                except Exception:
                    cors.append(0.0)
            cfi = int(np.argmax(cors)) if max(cors) > 0 else fi
            st.caption(f"Coloured by interaction feature: {feature_names[cfi]}")
            fig = go.Figure(go.Scatter(
                x=X_test_prep[:, fi], y=shap_values[:, fi], mode="markers",
                marker=dict(color=X_test_prep[:, cfi], colorscale=SHAP_SCALE, size=5, opacity=0.7,
                            colorbar=dict(title=feature_names[cfi]))))
            fig.add_hline(y=0, line_dash="dash", line_color=NEUTRAL)
            fig.update_layout(**layout(title=f"SHAP dependence · {sel}",
                              xaxis_title=f"{sel} (value)", yaxis_title=f"SHAP for {sel}",
                              height=440))
            chart(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 04 · SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif section == "Simulator":
    st.markdown("<h4>Case file 04 · Score one customer</h4>", unsafe_allow_html=True)
    st.markdown("# Simulator")
    st.markdown('<p class="lede">Build a customer profile, get a churn probability, the SHAP '
                'breakdown of <i>why</i>, and a predicted survival curve against the population '
                'baseline.</p>', unsafe_allow_html=True)

    df_ref = load_telco_csv()

    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<h4>Account</h4>", unsafe_allow_html=True)
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", df_ref["Contract"].unique().tolist())
            payment = st.selectbox("Payment method", df_ref["PaymentMethod"].unique().tolist())
            paperless = st.selectbox("Paperless billing", df_ref["PaperlessBilling"].unique().tolist())
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total = st.number_input("Total charges ($)", 0.0, 9000.0, float(65.0 * 12), step=10.0)
        with c2:
            st.markdown("<h4>Services</h4>", unsafe_allow_html=True)
            internet = st.selectbox("Internet service", df_ref["InternetService"].unique().tolist())
            phone = st.selectbox("Phone service", df_ref["PhoneService"].unique().tolist())
            multiple_lines = st.selectbox("Multiple lines", df_ref["MultipleLines"].unique().tolist())
            online_security = st.selectbox("Online security", df_ref["OnlineSecurity"].unique().tolist())
            tech_support = st.selectbox("Tech support", df_ref["TechSupport"].unique().tolist())
            streaming_tv = st.selectbox("Streaming TV", df_ref["StreamingTV"].unique().tolist())
            streaming_movies = st.selectbox("Streaming movies", df_ref["StreamingMovies"].unique().tolist())
            online_backup = st.selectbox("Online backup", df_ref["OnlineBackup"].unique().tolist())
            device_protection = st.selectbox("Device protection", df_ref["DeviceProtection"].unique().tolist())
        with c3:
            st.markdown("<h4>Demographics</h4>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", df_ref["gender"].unique().tolist())
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", df_ref["Partner"].unique().tolist())
            dependents = st.selectbox("Dependents", df_ref["Dependents"].unique().tolist())
        submitted = st.form_submit_button("Score customer", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])
        prob = float(model.predict_proba(input_df)[0][1])
        label, rcol, tint = risk_state(prob)
        if prob >= 0.60:
            action = "Immediate outreach. Offer a meaningful retention incentive now."
        elif prob >= 0.40:
            action = "Proactive check-in within two weeks. Consider a service upgrade offer."
        else:
            action = "Standard engagement. Monitor for usage changes."

        rule("Verdict", f"{label}")
        ga, gb = st.columns([1, 2])
        with ga:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%", "font": {"size": 38, "family": "IBM Plex Mono"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": INK},
                       "bar": {"color": rcol},
                       "bgcolor": BONE, "borderwidth": 1, "bordercolor": INK,
                       "steps": [{"range": [0, 40], "color": "rgba(14,124,102,0.18)"},
                                 {"range": [40, 60], "color": "rgba(183,121,31,0.18)"},
                                 {"range": [60, 100], "color": "rgba(179,54,31,0.18)"}],
                       "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.8,
                                     "value": prob * 100}}))
            fig.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20),
                              paper_bgcolor=BONE, font=dict(color=INK))
            chart(fig)
        with gb:
            st.markdown(f'<div class="blk fill" style="border-left:3px solid {rcol};">'
                        f'<div style="font-family:IBM Plex Mono,monospace; font-weight:600; '
                        f'color:{rcol}; letter-spacing:0.08em;">{label} · {prob:.0%}</div>'
                        f'<div style="margin-top:0.4rem; font-size:0.95rem;">{action}</div></div>',
                        unsafe_allow_html=True)
            signals = []
            if contract == "Month-to-month":
                signals.append("Month-to-month contract · highest-risk contract type")
            if tenure < 12:
                signals.append(f"Low tenure ({tenure} mo) · early-lifecycle churn window")
            if monthly > 70:
                signals.append(f"High monthly charges (${monthly:.0f})")
            if online_security in ["No", "No internet service"]:
                signals.append("No Online Security · protective service absent")
            if tech_support in ["No", "No internet service"]:
                signals.append("No Tech Support · protective service absent")
            if senior == 1:
                signals.append("Senior citizen · higher-churn cohort")
            if signals:
                st.markdown("<h4>Risk signals</h4>", unsafe_allow_html=True)
                for s in signals:
                    st.markdown(f'<span class="tag haz">{s}</span>', unsafe_allow_html=True)

        # SHAP waterfall
        if shap_data is not None and "XGBoost" in meta["model_name"]:
            @st.cache_resource
            def get_prep_and_booster():
                return model.named_steps["prep"], model.named_steps["clf"].get_booster()
            try:
                import xgboost as xgb_lib
                prep_sim, booster_sim = get_prep_and_booster()
                X_prep = prep_sim.transform(input_df).astype(np.float32)
                contribs = booster_sim.predict(xgb_lib.DMatrix(X_prep), pred_contribs=True)
                sv = contribs[0, :-1]; exp_val = float(contribs[0, -1])
                fnames = shap_data["feature_names"]
                ia = np.argsort(np.abs(sv))[-10:][::-1]
                order = ia[np.argsort(sv[ia])]
                svs = sv[order]; fts = [fnames[i] for i in order]
                bases, run = [], exp_val
                for v in svs:
                    bases.append(run); run += v
                cols = [HAZARD if v > 0 else PROTECT for v in svs]
                fig = go.Figure(go.Bar(
                    x=svs, y=fts, orientation="h", base=bases, marker_color=cols,
                    text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in svs],
                    textposition="outside",
                    textfont=dict(family="IBM Plex Mono", size=11)))
                fig.add_vline(x=exp_val, line_dash="dot", line_color=NEUTRAL,
                              annotation_text=f"base {exp_val:.2f}")
                fig.add_vline(x=exp_val + sv.sum(), line_dash="solid", line_color=rcol,
                              annotation_text=f"score {exp_val + sv.sum():.2f}")
                fig.update_layout(**layout(
                    title=f"Why this customer scores {prob:.0%}",
                    xaxis_title="Model output (log-odds)", yaxis=dict(automargin=True),
                    height=420, margin=dict(l=10, r=40, t=56, b=50)))
                chart(fig)
                callout("Magenta bars push toward churn, cyan away. Base value is the population "
                        f"average log-odds ({exp_val:.2f}).", "")
            except Exception as e:
                st.info(f"SHAP waterfall unavailable: {e}")

        # Survival profile
        rule("Survival profile", "Cox PH estimate")
        try:
            cox_input = pd.DataFrame([{
                "Contract": contract, "InternetService": internet, "MonthlyCharges": monthly,
                "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "OnlineSecurity": online_security, "TechSupport": tech_support,
                "PaymentMethod": payment, "MultipleLines": multiple_lines,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies}])
            sf = cox.predict_survival_function(cox_input)
            bl = cox.baseline_survival_
            r, g, b = (int(rcol[1:3], 16), int(rcol[3:5], 16), int(rcol[5:7], 16))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bl.index.tolist(), y=bl.iloc[:, 0].tolist(), mode="lines",
                          name="Population average", line=dict(color=NEUTRAL, width=1.5, dash="dash")))
            fig.add_trace(go.Scatter(x=sf.index.tolist(), y=sf.iloc[:, 0].tolist(), mode="lines",
                          name="This customer", line=dict(color=rcol, width=2.6),
                          fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.10)"))
            fig.add_vline(x=tenure, line_dash="dot", line_color=CAUTION,
                          annotation_text=f"tenure {tenure}mo")
            fig.add_hline(y=0.5, line_dash="dot", line_color=NEUTRAL, annotation_text="50%")
            fig.update_layout(**layout(title="Predicted survival function (Cox PH)",
                              xaxis_title="Months as customer", yaxis_title="Survival probability",
                              yaxis=dict(tickformat=".0%", range=[0, 1.04]), height=380))
            chart(fig)
        except Exception as e:
            st.info(f"Survival profile unavailable: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 05 · IMPACT
# ═════════════════════════════════════════════════════════════════════════════
elif section == "Impact":
    st.markdown("<h4>Case file 05 · The money</h4>", unsafe_allow_html=True)
    st.markdown("# Impact")
    st.markdown('<p class="lede">A ranking model is only worth the dollars it saves. How much '
                'churn a campaign catches, what it costs, and the targeting depth that maximises '
                'net return.</p>', unsafe_allow_html=True)

    lift_data = meta["lift_data"]
    n_customers = meta["n_samples"]
    churn_rate = meta["churn_rate"]
    avg_monthly = meta["avg_monthly_charges"]
    best_name = meta["model_name"]
    pct_t = np.array(lift_data["pct_targeted"])
    pct_c = np.array(lift_data["pct_churn_captured"])

    rule("Lift curve", "churners captured vs targeted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                  line=dict(color=NEUTRAL, dash="dash")))
    fig.add_trace(go.Scatter(x=pct_t.tolist(), y=pct_c.tolist(), mode="lines",
                  name=f"{best_name}", line=dict(color=HAZARD, width=2.6),
                  fill="tonexty", fillcolor="rgba(179,54,31,0.08)"))
    for pa in [0.10, 0.20, 0.30]:
        ca = float(np.interp(pa, pct_t, pct_c))
        fig.add_annotation(x=pa, y=ca, text=f"{pa:.0%}→{ca:.0%}", showarrow=True, arrowhead=2,
                           ax=55, ay=-28, font=dict(size=10, family="IBM Plex Mono"))
    fig.update_layout(**layout(title="Lift curve", xaxis_title="% customers targeted (by score)",
                      yaxis_title="% churners captured", xaxis=dict(tickformat=".0%"),
                      yaxis=dict(tickformat=".0%"), height=440))
    chart(fig)

    cols = st.columns(4)
    for col, pct in zip(cols, [0.10, 0.20, 0.30, 0.50]):
        cap = float(np.interp(pct, pct_t, pct_c))
        col.markdown(f'<div class="blk"><div style="font-family:IBM Plex Mono;font-weight:600;'
                     f'font-size:1.4rem;">{cap:.0%}</div><div style="font-family:IBM Plex Mono;'
                     f'font-size:0.64rem;text-transform:uppercase;letter-spacing:0.1em;color:{INK60};'
                     f'margin-top:0.3rem;">top {pct:.0%} → {cap/pct:.1f}× lift</div></div>',
                     unsafe_allow_html=True)

    rule("ROI calculator", "tune the campaign")
    cs, cm = st.columns(2)
    with cs:
        pct_target = st.slider("% customers to target", 5, 50, 20, step=5)
        offer_cost = st.slider("Retention offer per customer ($)", 5, 100, 25, step=5)
        save_rate = st.slider("Assumed save rate (% of captured retained)", 10, 60, 30, step=5)
        st.caption(f"Dataset avg monthly charge: ${avg_monthly:.2f}/customer")
    with cm:
        pct_cap = float(np.interp(pct_target / 100, pct_t, pct_c))
        total_churn = int(n_customers * churn_rate)
        captured = int(total_churn * pct_cap)
        targeted = int(n_customers * pct_target / 100)
        saved = int(captured * save_rate / 100)
        rev = saved * avg_monthly * 12
        cost = targeted * offer_cost
        net = rev - cost
        st.markdown(f"""<div class="kpis" style="grid-template-columns:repeat(2,1fr);">
          <div class="kpi"><div class="v">{targeted:,}</div><div class="k">Targeted</div></div>
          <div class="kpi"><div class="v">{captured:,}</div><div class="k">Churners captured</div>
            <div class="s">{pct_cap:.0%} of all churn</div></div>
          <div class="kpi"><div class="v pro">${rev:,.0f}</div><div class="k">Revenue saved / yr</div></div>
          <div class="kpi"><div class="v {'pro' if net>0 else 'haz'}">${net:,.0f}</div>
            <div class="k">Net ROI</div><div class="s">cost ${cost:,.0f}</div></div>
        </div>""", unsafe_allow_html=True)

    rule("Sensitivity", "optimal targeting depth")
    pct_range = np.arange(0.05, 0.55, 0.05)
    net_rois = []
    for pct in pct_range:
        cap = float(np.interp(pct, pct_t, pct_c))
        cc = int(total_churn * cap)
        ct = int(n_customers * pct)
        sv = int(cc * save_rate / 100)
        net_rois.append(sv * avg_monthly * 12 - ct * offer_cost)
    opt = int(np.argmax(net_rois))
    fig = go.Figure(go.Bar(x=[f"{int(p*100)}%" for p in pct_range], y=net_rois,
                    marker_color=[PROTECT if v > 0 else HAZARD for v in net_rois]))
    fig.add_annotation(x=f"{int(pct_range[opt]*100)}%", y=net_rois[opt],
                       text=f"optimal {int(pct_range[opt]*100)}% → ${net_rois[opt]:,.0f}",
                       showarrow=True, arrowhead=2, bgcolor=BONE, borderpad=4,
                       font=dict(family="IBM Plex Mono", size=11))
    fig.add_hline(y=0, line_dash="dash", line_color=NEUTRAL)
    fig.update_layout(**layout(title=f"Net ROI by targeting % (offer ${offer_cost}, save {save_rate}%)",
                      xaxis_title="% customers targeted", yaxis_title="Net ROI ($)", height=400))
    chart(fig)
    callout(f"Optimal strategy at these settings: target the top <b>{int(pct_range[opt]*100)}%</b> "
            f"by churn score for an estimated net ROI of <b>${net_rois[opt]:,.0f}</b>.", "pro")
