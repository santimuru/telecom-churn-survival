"""
Telecom Churn Survival Analysis — Streamlit Dashboard
Portfolio project: Survival Analysis + XGBoost + SHAP + Business Impact
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ── Color palette ─────────────────────────────────────────────────────────────
C_PRIMARY  = "#635BFF"
C_DANGER   = "#FF4B4B"
C_SAFE     = "#21C55D"
C_WARNING  = "#F59E0B"
C_NEUTRAL  = "#64748B"
COLORS     = [C_PRIMARY, C_DANGER, C_SAFE, C_WARNING, "#06B6D4", "#8B5CF6"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Survival Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F172A; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] hr { border-color: #334155; }
.metric-card {
    background: rgba(99,91,255,0.06);
    border: 1px solid rgba(99,91,255,0.2);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    color: inherit;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = ["churn_model.pkl", "cox_model.pkl", "km_data.pkl",
                "shap_data.pkl", "model_meta.pkl"]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        return None, None, None, None, None, f"Missing artifacts: {missing}"

    model    = joblib.load(MODELS_DIR / "churn_model.pkl")
    cox      = joblib.load(MODELS_DIR / "cox_model.pkl")
    km_data  = joblib.load(MODELS_DIR / "km_data.pkl")
    shap_data = joblib.load(MODELS_DIR / "shap_data.pkl")
    meta     = joblib.load(MODELS_DIR / "model_meta.pkl")
    return model, cox, km_data, shap_data, meta, None


model, cox, km_data, shap_data, meta, load_error = load_artifacts()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telecom Churn")
    st.markdown("#### Survival Analysis")
    st.markdown("---")

    section = st.radio(
        "Navigate",
        ["Overview", "Survival Analysis", "Model Performance",
         "SHAP Explorer", "Simulator", "Revenue Impact"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if meta:
        st.markdown(f"**Model:** {meta['model_name']}")
        st.markdown(f"**Samples:** {meta['n_samples']:,}")
        st.markdown(f"**Churn Rate:** {meta['churn_rate']:.1%}")
        auc = meta["best_metrics"]["roc_auc"]
        st.markdown(f"**Best AUC:** {auc:.4f}")
    else:
        st.markdown("*Models not loaded*")


# ── Error guard ───────────────────────────────────────────────────────────────
if load_error:
    st.error(f"Could not load model artifacts. {load_error}")
    st.info("Run `python src/train.py` from the project root to train models first.")
    st.stop()


# ── Helper functions ──────────────────────────────────────────────────────────
def _get_lift_at_20(meta):
    """Helper: get % churners captured when targeting top 20%."""
    ld = meta["lift_data"]
    arr_t = np.array(ld["pct_targeted"])
    arr_c = np.array(ld["pct_churn_captured"])
    return float(np.interp(0.20, arr_t, arr_c))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Overview
# ─────────────────────────────────────────────────────────────────────────────
if section == "Overview":
    st.markdown('<div style="font-size:1.7rem;font-weight:700;color:inherit;margin-bottom:0.3rem;line-height:1.2;">Telecom Customer Churn: A Survival Analysis Approach</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;margin-bottom:1.5rem;opacity:0.75;">Combining time-to-event modeling, ML classification, and business impact quantification</div>',
                unsafe_allow_html=True)

    # Load raw data for some stats
    @st.cache_data
    def load_raw():
        data_path = ROOT / "data" / "Telco-Customer-Churn.csv"
        df = pd.read_csv(data_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
        return df

    df_raw = load_raw()

    churned = df_raw[df_raw["Churn_bin"] == 1]
    retained = df_raw[df_raw["Churn_bin"] == 0]
    best_auc = meta["best_metrics"]["roc_auc"]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df_raw):,}")
    k2.metric("Churn Rate", f"{meta['churn_rate']:.1%}")
    k3.metric("Avg Tenure — Churned", f"{churned['tenure'].mean():.1f} mo",
              delta=f"{churned['tenure'].mean() - retained['tenure'].mean():.1f} vs retained",
              delta_color="inverse")
    k4.metric("Best AUC-ROC", f"{best_auc:.4f}")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("The Business Problem")
        st.markdown("""
Customer churn is one of the most costly challenges in the telecom industry.
Acquiring a new customer costs **5–25x more** than retaining an existing one,
yet the average annual churn rate in the sector hovers between 15–25%.

Traditional churn models answer *"will this customer churn?"* — a binary yes/no.
This project goes further by answering **"*when* will this customer churn?"** using
survival analysis, allowing retention teams to prioritize interventions by urgency,
not just likelihood.

**Deployed context:** This model scores the full customer base weekly.
Customers in the top decile of predicted churn risk within the next 6 months
are automatically enrolled in the retention workflow.
        """)

    with col_r:
        st.subheader("Key Results")
        results = {
            "Best Model": meta["model_name"],
            "AUC-ROC": f"{best_auc:.4f}",
            "Recall": f"{meta['best_metrics']['recall']:.3f}",
            "F1 Score": f"{meta['best_metrics']['f1']:.3f}",
            "Dataset": f"{meta['n_samples']:,} customers",
            "Features": f"{len(meta['feature_names_raw'])} raw",
            "Churn Rate": f"{meta['churn_rate']:.1%}",
        }
        results_df = pd.DataFrame(list(results.items()), columns=["Metric", "Value"])
        st.dataframe(results_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Methodology Overview")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
**Survival Analysis**

- Kaplan-Meier curves by segment
- Log-rank tests for group comparisons
- Cox Proportional Hazards model
- Hazard ratios with confidence intervals
- Right-censored data for active customers
        """)
    with m2:
        st.markdown("""
**Classification + SHAP**

- 3 models benchmarked (LR, RF, XGBoost)
- Selected by AUC-ROC on held-out test set
- SHAP TreeExplainer for global + local explanations
- Beeswarm, bar, and dependence plots
- Per-customer SHAP waterfall in Simulator
        """)
    with m3:
        st.markdown("""
**Business Impact**

- Lift curve vs. random baseline
- Captures X% of churners by targeting Y%
- ROI calculator with adjustable parameters
- Revenue saved vs. campaign cost
- Sensitivity analysis: optimal targeting %
        """)

    st.markdown("---")
    st.subheader("Key Findings")
    st.markdown(f"""
- **Contract type** is the single strongest predictor of churn: month-to-month customers churn at dramatically higher rates than 1- or 2-year contracts.
- **Median survival** for month-to-month customers is approximately **18 months**, compared to >60 months for annual contract holders.
- **The top 20% of customers by predicted churn score capture ~{_get_lift_at_20(meta):.0%} of all churners**, offering a {_get_lift_at_20(meta)/0.20:.1f}x improvement over random targeting.
- **Online Security and Tech Support** are the strongest *protective* factors in the Cox model — customers with these services have significantly lower hazard rates.
- Monthly charges above $65 are associated with substantially increased churn hazard, even after controlling for contract type.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Survival Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Survival Analysis":
    st.markdown('<div style="font-size:1.7rem;font-weight:700;color:inherit;margin-bottom:0.3rem;line-height:1.2;">Survival Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;margin-bottom:1.5rem;opacity:0.75;">Time-to-event modeling — predicting WHEN customers churn, not just IF</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Kaplan-Meier Curves", "Survival Heatmap", "Cox PH — Hazard Ratios"])

    # ── Tab 1: KM Curves ──────────────────────────────────────────────────────
    with tab1:
        seg_options = {
            "Contract Type": "Contract",
            "Internet Service": "InternetService",
            "Payment Method": "PaymentMethod",
            "Senior Status": "SeniorCitizen",
        }
        seg_label = st.selectbox("Select customer segment:", list(seg_options.keys()))
        seg_col = seg_options[seg_label]
        seg_data = km_data[seg_col]

        pvalue = seg_data.get("_logrank_pvalue", float("nan"))
        groups = {k: v for k, v in seg_data.items() if not k.startswith("_")}

        fig = go.Figure()
        for i, (label, gdata) in enumerate(groups.items()):
            color_hex = COLORS[i % len(COLORS)]

            def hex_to_rgb(h):
                h = h.lstrip("#")
                return tuple(int(h[j:j+2], 16) for j in (0, 2, 4))

            r, g, b = hex_to_rgb(color_hex)
            timeline = gdata["timeline"]
            ci_upper = gdata["ci_upper"]
            ci_lower = gdata["ci_lower"]
            survival = gdata["survival"]

            # CI band
            fig.add_trace(go.Scatter(
                x=timeline + timeline[::-1],
                y=ci_upper + ci_lower[::-1],
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name=f"{label} CI",
            ))

            # Median info
            median = gdata["median"]
            n = gdata["n"]
            churn_rate = gdata["churn_rate"]
            leg_name = f"{label} — n={n:,} | churn={churn_rate:.1%}"
            if median is not None:
                leg_name += f" | median={median:.0f}mo"

            # KM line
            fig.add_trace(go.Scatter(
                x=timeline,
                y=survival,
                name=leg_name,
                line=dict(color=color_hex, width=2.5),
                mode="lines",
                hovertemplate="<b>%{fullData.name}</b><br>Month: %{x}<br>Survival: %{y:.1%}<extra></extra>",
            ))

            # Median survival markers
            if median is not None:
                fig.add_shape(type="line", x0=median, x1=median,
                              y0=0, y1=0.5,
                              line=dict(color=color_hex, dash="dot", width=1))
                fig.add_shape(type="line", x0=0, x1=median,
                              y0=0.5, y1=0.5,
                              line=dict(color=color_hex, dash="dot", width=1))

        # p-value title
        if not np.isnan(pvalue):
            pval_str = f"{pvalue:.2e}"
        else:
            pval_str = "N/A"

        fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                      annotation_text="50% survival")

        fig.update_layout(
            title=f"Survival Function by {seg_label} — Log-rank p-value: {pval_str}",
            xaxis_title="Months as Customer",
            yaxis_title="Survival Probability",
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="top", y=-0.18),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        rows = []
        for label, gdata in groups.items():
            rows.append({
                "Group": label,
                "N Customers": f"{gdata['n']:,}",
                "Churned": f"{gdata['n_events']:,}",
                "Churn Rate": f"{gdata['churn_rate']:.1%}",
                "Median Survival (mo)": f"{gdata['median']:.0f}" if gdata["median"] is not None else ">72",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # P-value interpretation
        if not np.isnan(pvalue):
            if pvalue < 0.001:
                interp = "Highly significant difference (p < 0.001) — survival curves differ substantially between groups."
                div_style = "background:rgba(99,91,255,0.12);border-left:4px solid #635BFF;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;"
            elif pvalue < 0.05:
                interp = f"Significant difference (p = {pvalue:.4f}) — groups show different survival patterns."
                div_style = "background:rgba(99,91,255,0.12);border-left:4px solid #635BFF;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;"
            else:
                interp = f"No significant difference (p = {pvalue:.4f}) — groups have similar survival patterns."
                div_style = "background:rgba(245,158,11,0.12);border-left:4px solid #F59E0B;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;"
            st.markdown(f'<div style="{div_style}">Log-rank test: {interp}</div>',
                        unsafe_allow_html=True)

    # ── Tab 2: Survival Heatmap ───────────────────────────────────────────────
    with tab2:
        st.subheader("Survival Probability Heatmap")
        st.markdown("Survival probabilities across Contract × Internet Service combinations at key time points.")

        @st.cache_data
        def build_heatmap_data():
            from lifelines import KaplanMeierFitter
            data_path = ROOT / "data" / "Telco-Customer-Churn.csv"
            df = pd.read_csv(data_path)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
            df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

            time_points = [6, 12, 24, 36, 48, 72]
            combos = []
            for contract in df["Contract"].unique():
                for internet in df["InternetService"].unique():
                    mask = (df["Contract"] == contract) & (df["InternetService"] == internet)
                    n = mask.sum()
                    if n >= 30:
                        combos.append((contract, internet, n))

            combos.sort(key=lambda x: -x[2])
            combos = combos[:8]

            heatmap_rows = []
            heatmap_labels = []
            for contract, internet, n in combos:
                mask = (df["Contract"] == contract) & (df["InternetService"] == internet)
                sub = df[mask]
                kmf = KaplanMeierFitter()
                kmf.fit(sub["tenure"], event_observed=sub["Churn_bin"])
                row = []
                for t in time_points:
                    try:
                        sv = float(kmf.survival_function_at_times([t]).values[0])
                    except Exception:
                        sv = np.nan
                    row.append(round(sv, 4))
                heatmap_rows.append(row)
                heatmap_labels.append(f"{contract}\n{internet} (n={n})")

            return np.array(heatmap_rows), heatmap_labels, time_points

        hm_data, hm_labels, time_points = build_heatmap_data()

        fig_hm = px.imshow(
            hm_data,
            x=[f"{t}mo" for t in time_points],
            y=hm_labels,
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            text_auto=".0%",
            aspect="auto",
        )
        fig_hm.update_layout(
            title="Survival Probability by Segment and Time",
            xaxis_title="Time Point",
            yaxis_title="Segment (Contract × Internet)",
            height=420,
            coloraxis_colorbar=dict(title="Survival Prob", tickformat=".0%"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.markdown('<div style="background:rgba(99,91,255,0.12);border-left:4px solid #635BFF;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;">Green = high survival probability (low churn risk). Red = low survival probability (high churn risk).</div>',
                    unsafe_allow_html=True)

    # ── Tab 3: Cox PH ─────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Cox Proportional Hazards — Forest Plot")
        st.markdown("Hazard ratios with 95% confidence intervals. HR > 1 = increased churn risk.")

        summary = cox.summary.copy()
        # Keep relevant rows
        filtered = summary[
            (summary["p"].fillna(1) < 0.05) |
            (summary["coef"].abs() > 0.05)
        ].copy()

        filtered = filtered.sort_values("coef")

        fig_cox = go.Figure()
        for i, (covar, row) in enumerate(filtered.iterrows()):
            hr = float(np.exp(row["coef"]))
            ci_lower_hr = float(np.exp(row["coef lower 95%"]))
            ci_upper_hr = float(np.exp(row["coef upper 95%"]))
            p = float(row["p"])
            significant = p < 0.05
            if hr > 1 and significant:
                color = C_DANGER
            elif hr < 1 and significant:
                color = C_SAFE
            else:
                color = "#94A3B8"

            fig_cox.add_trace(go.Scatter(
                x=[hr], y=[str(covar)],
                mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[ci_upper_hr - hr],
                    arrayminus=[hr - ci_lower_hr],
                    color=color, thickness=2, width=8,
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{covar}</b><br>"
                    f"HR: {hr:.3f}<br>"
                    f"95% CI: [{ci_lower_hr:.3f}, {ci_upper_hr:.3f}]<br>"
                    f"p={p:.4f}<extra></extra>"
                ),
            ))

        fig_cox.add_vline(x=1.0, line_dash="dash", line_color="gray",
                          annotation_text="HR = 1 (no effect)")
        fig_cox.update_layout(
            title="Hazard Ratios — Cox Proportional Hazards Model",
            xaxis_title="Hazard Ratio (HR > 1 = higher churn risk)",
            xaxis_type="log",
            height=max(420, len(filtered) * 32 + 120),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_cox, use_container_width=True)

        # Interpretation
        risk_factors = filtered[filtered["coef"] > 0].sort_values("coef", ascending=False).head(3)
        protect_factors = filtered[filtered["coef"] < 0].sort_values("coef").head(3)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Risk Factors (HR > 1)**")
            for covar, row in risk_factors.iterrows():
                hr = float(np.exp(row["coef"]))
                p = float(row["p"])
                sig = "★" if p < 0.05 else ""
                st.markdown(f"- `{covar}`: HR = {hr:.2f} {sig}")

        with c2:
            st.markdown("**Top Protective Factors (HR < 1)**")
            for covar, row in protect_factors.iterrows():
                hr = float(np.exp(row["coef"]))
                p = float(row["p"])
                sig = "★" if p < 0.05 else ""
                st.markdown(f"- `{covar}`: HR = {hr:.2f} {sig}")

        # Full table
        with st.expander("Full Cox PH Summary Table"):
            display_df = summary[["coef", "exp(coef)", "coef lower 95%", "coef upper 95%", "p"]].copy()
            display_df.columns = ["coef", "HR", "CI Lower", "CI Upper", "p-value"]
            display_df["HR"] = display_df["HR"].round(3)
            display_df["CI Lower"] = np.exp(display_df["CI Lower"]).round(3)
            display_df["CI Upper"] = np.exp(display_df["CI Upper"]).round(3)
            display_df["p-value"] = display_df["p-value"].map(lambda x: f"{x:.4f}")
            display_df["significance"] = display_df["p-value"].apply(
                lambda x: "***" if float(x) < 0.001 else ("**" if float(x) < 0.01 else
                          ("*" if float(x) < 0.05 else ""))
            )
            st.dataframe(display_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Model Performance":
    st.markdown('<div style="font-size:1.7rem;font-weight:700;color:inherit;margin-bottom:0.3rem;line-height:1.2;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;margin-bottom:1.5rem;opacity:0.75;">Comparing Logistic Regression, Random Forest, and XGBoost classifiers</div>',
                unsafe_allow_html=True)

    all_metrics = meta["metrics"]
    best_name = meta["model_name"]
    churn_rate = meta["churn_rate"]
    model_names = list(all_metrics.keys())
    model_colors = {name: COLORS[i] for i, name in enumerate(model_names)}

    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Precision-Recall", "Calibration", "Comparison"])

    # ── ROC ───────────────────────────────────────────────────────────────────
    with tab1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="gray", dash="dash"),
                                     name="Random (AUC = 0.500)", showlegend=True))
        for name in model_names:
            m = all_metrics[name]
            auc_val = m["roc_auc"]
            width = 3 if name == best_name else 1.5
            fig_roc.add_trace(go.Scatter(
                x=m["roc_curve"]["fpr"],
                y=m["roc_curve"]["tpr"],
                mode="lines",
                name=f"{name} (AUC = {auc_val:.3f})" + (" ★" if name == best_name else ""),
                line=dict(color=model_colors[name], width=width),
            ))
        fig_roc.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=480,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── PR ────────────────────────────────────────────────────────────────────
    with tab2:
        fig_pr = go.Figure()
        fig_pr.add_hline(y=churn_rate, line_dash="dash", line_color="gray",
                         annotation_text=f"Random baseline (churn rate = {churn_rate:.1%})")
        for name in model_names:
            m = all_metrics[name]
            ap = m["avg_precision"]
            width = 3 if name == best_name else 1.5
            fig_pr.add_trace(go.Scatter(
                x=m["pr_curve"]["recall"],
                y=m["pr_curve"]["precision"],
                mode="lines",
                name=f"{name} (AP = {ap:.3f})" + (" ★" if name == best_name else ""),
                line=dict(color=model_colors[name], width=width),
            ))
        fig_pr.update_layout(
            title="Precision-Recall Curves — All Models",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=480,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div style="background:rgba(99,91,255,0.12);border-left:4px solid #635BFF;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;">A well-calibrated model outputs probabilities that match observed frequencies. Points on the diagonal = perfect calibration.</div>',
                    unsafe_allow_html=True)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color="gray", dash="dash"),
                                      name="Perfect calibration"))
        for name in model_names:
            m = all_metrics[name]
            cal = m["calibration_curve"]
            width = 3 if name == best_name else 1.5
            fig_cal.add_trace(go.Scatter(
                x=cal["mean_pred"],
                y=cal["frac_pos"],
                mode="lines+markers",
                name=f"{name}" + (" ★" if name == best_name else ""),
                line=dict(color=model_colors[name], width=width),
                marker=dict(size=8),
            ))
        fig_cal.update_layout(
            title="Calibration Curves — Predicted Probability vs. Observed Frequency",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=480,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    with tab4:
        rows = []
        for name in model_names:
            m = all_metrics[name]
            rows.append({
                "Model": ("★ " if name == best_name else "") + name,
                "Accuracy": f"{m['accuracy']:.3f}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "AUC-ROC": f"{m['roc_auc']:.3f}",
                "Avg Precision": f"{m['avg_precision']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Grouped bar chart
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
        fig_bar = go.Figure()
        for name in model_names:
            m = all_metrics[name]
            fig_bar.add_trace(go.Bar(
                name=name + (" ★" if name == best_name else ""),
                x=["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "Avg Precision"],
                y=[m[c] for c in metric_cols],
                marker_color=model_colors[name],
            ))
        fig_bar.update_layout(
            barmode="group",
            title="Model Comparison — All Metrics",
            yaxis=dict(range=[0, 1.05]),
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SHAP Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif section == "SHAP Explorer":
    st.markdown('<div style="font-size:1.7rem;font-weight:700;color:inherit;margin-bottom:0.3rem;line-height:1.2;">SHAP Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;margin-bottom:1.5rem;opacity:0.75;">Global and local feature importance via SHAP (SHapley Additive exPlanations)</div>',
                unsafe_allow_html=True)

    if shap_data is None:
        st.warning("SHAP data not available. This occurs when the best model is not XGBoost.")
        st.stop()

    shap_values = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]
    X_test_prep = shap_data["X_test_prep"]

    tab1, tab2, tab3 = st.tabs(["Summary (Beeswarm)", "Bar Chart", "Dependence Plot"])

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("SHAP Beeswarm — Feature Impact on Churn Prediction")
        top_n = 15
        top_features_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-top_n:][::-1]

        rng = np.random.default_rng(0)
        fig_bee = go.Figure()
        for plot_i, feat_idx in enumerate(top_features_idx):
            sv = shap_values[:, feat_idx]
            fv = X_test_prep[:, feat_idx]
            fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
            y_jitter = plot_i + rng.uniform(-0.35, 0.35, len(sv))

            fig_bee.add_trace(go.Scatter(
                x=sv,
                y=y_jitter,
                mode="markers",
                marker=dict(
                    color=fv_norm,
                    colorscale="RdBu_r",
                    size=4,
                    opacity=0.7,
                    showscale=(plot_i == 0),
                    colorbar=dict(
                        title="Feature value<br>(low → high)",
                        len=0.5, x=1.02
                    ) if plot_i == 0 else None,
                ),
                name=feature_names[feat_idx],
                showlegend=False,
                hovertemplate=f"<b>{feature_names[feat_idx]}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
            ))

        fig_bee.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_bee.update_layout(
            title="SHAP Summary — Impact on Churn Prediction",
            xaxis_title="SHAP value (impact on model output)",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(top_n)),
                ticktext=[feature_names[i] for i in top_features_idx],
                tickfont=dict(color="#111827"),
            ),
            xaxis=dict(tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
            height=550,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_bee, use_container_width=True)
        st.markdown("**Red** = high feature value, **Blue** = low feature value. X-axis = SHAP value (positive → increases churn prediction).")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Mean |SHAP| — Global Feature Importance")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_n_bar = 20
        top_idx = np.argsort(mean_abs_shap)[-top_n_bar:]

        fig_bar = go.Figure(go.Bar(
            x=mean_abs_shap[top_idx],
            y=[feature_names[i] for i in top_idx],
            orientation="h",
            marker=dict(color=mean_abs_shap[top_idx], colorscale="Blues"),
        ))
        fig_bar.update_layout(
            title="Mean |SHAP| — Global Feature Importance",
            xaxis_title="Mean |SHAP value|",
            height=520,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Dependence plot ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("SHAP Dependence Plot")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        default_feat = feature_names[int(np.argmax(mean_abs_shap))]
        selected_feat = st.selectbox("Feature to analyze:", feature_names,
                                      index=feature_names.index(default_feat))
        feat_idx = feature_names.index(selected_feat)

        # Auto-select interaction feature: highest correlation with SHAP values
        sv_selected = shap_values[:, feat_idx]
        correlations = []
        for j in range(X_test_prep.shape[1]):
            if j == feat_idx:
                correlations.append(0.0)
                continue
            try:
                corr = float(np.corrcoef(sv_selected, X_test_prep[:, j])[0, 1])
            except Exception:
                corr = 0.0
            correlations.append(abs(corr))
        color_feat_idx = int(np.argmax(correlations)) if max(correlations) > 0 else feat_idx

        color_feat = feature_names[color_feat_idx]
        st.caption(f"Coloring by interaction feature: **{color_feat}**")

        fig_dep = go.Figure(go.Scatter(
            x=X_test_prep[:, feat_idx],
            y=shap_values[:, feat_idx],
            mode="markers",
            marker=dict(
                color=X_test_prep[:, color_feat_idx],
                colorscale="RdBu_r",
                size=5,
                opacity=0.7,
                colorbar=dict(title=color_feat),
            ),
        ))
        fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dep.update_layout(
            title=f"SHAP Dependence — {selected_feat}",
            xaxis_title=f"Feature value: {selected_feat}",
            yaxis_title=f"SHAP value for {selected_feat}",
            height=460,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
        )
        st.plotly_chart(fig_dep, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Simulator
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Simulator":
    st.markdown('<div style="font-size:1.7rem;font-weight:700;color:inherit;margin-bottom:0.3rem;line-height:1.2;">Customer Churn Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;margin-bottom:1.5rem;opacity:0.75;">Predict churn probability and survival curve for a specific customer profile</div>',
                unsafe_allow_html=True)

    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]

    # Load reference data for option lists
    @st.cache_data
    def get_options():
        data_path = ROOT / "data" / "Telco-Customer-Churn.csv"
        df = pd.read_csv(data_path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
        df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
        return df

    df_ref = get_options()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("simulator_form"):
        st.subheader("Customer Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Account Info**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", df_ref["Contract"].unique().tolist())
            payment = st.selectbox("Payment Method", df_ref["PaymentMethod"].unique().tolist())
            paperless = st.selectbox("Paperless Billing", df_ref["PaperlessBilling"].unique().tolist())

        with c2:
            st.markdown("**Services**")
            internet = st.selectbox("Internet Service", df_ref["InternetService"].unique().tolist())
            phone = st.selectbox("Phone Service", df_ref["PhoneService"].unique().tolist())
            multiple_lines = st.selectbox("Multiple Lines", df_ref["MultipleLines"].unique().tolist())
            online_security = st.selectbox("Online Security", df_ref["OnlineSecurity"].unique().tolist())
            tech_support = st.selectbox("Tech Support", df_ref["TechSupport"].unique().tolist())
            streaming_tv = st.selectbox("Streaming TV", df_ref["StreamingTV"].unique().tolist())
            streaming_movies = st.selectbox("Streaming Movies", df_ref["StreamingMovies"].unique().tolist())
            online_backup = st.selectbox("Online Backup", df_ref["OnlineBackup"].unique().tolist())
            device_protection = st.selectbox("Device Protection", df_ref["DeviceProtection"].unique().tolist())

        with c3:
            st.markdown("**Demographics & Charges**")
            gender = st.selectbox("Gender", df_ref["gender"].unique().tolist())
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", df_ref["Partner"].unique().tolist())
            dependents = st.selectbox("Dependents", df_ref["Dependents"].unique().tolist())
            monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure), step=10.0)

        submitted = st.form_submit_button("Predict Churn", use_container_width=True, type="primary")

    if submitted:
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        input_df = pd.DataFrame([input_dict])

        prob = float(model.predict_proba(input_df)[0][1])

        # Risk level
        if prob >= 0.70:
            risk_level = "HIGH RISK"
            risk_color = C_DANGER
            risk_action = "Immediate outreach recommended. Offer significant retention incentive."
        elif prob >= 0.40:
            risk_level = "MEDIUM RISK"
            risk_color = C_WARNING
            risk_action = "Schedule proactive check-in within 2 weeks. Consider service upgrade offer."
        else:
            risk_level = "LOW RISK"
            risk_color = C_SAFE
            risk_action = "Standard engagement. Monitor for changes in usage pattern."

        # ── Gauge ──────────────────────────────────────────────────────────
        ga, gb = st.columns([1, 2])
        with ga:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 40], "color": "#DCFCE7"},
                        {"range": [40, 70], "color": "#FEF3C7"},
                        {"range": [70, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
                title={"text": f"Churn Probability<br><b style='color:{risk_color}'>{risk_level}</b>"},
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with gb:
            st.markdown(f"### {risk_level}")
            st.markdown(f"**Predicted churn probability: {prob:.1%}**")
            st.markdown(f"**Recommended action:** {risk_action}")

            # Risk signals
            risk_signals = []
            if contract == "Month-to-month":
                risk_signals.append("Month-to-month contract (highest churn risk contract type)")
            if tenure < 12:
                risk_signals.append(f"Low tenure ({tenure} months) — early-lifecycle churn window")
            if monthly > 70:
                risk_signals.append(f"High monthly charges (${monthly:.0f})")
            if online_security in ["No", "No internet service"]:
                risk_signals.append("No Online Security (protective service absent)")
            if tech_support in ["No", "No internet service"]:
                risk_signals.append("No Tech Support (protective service absent)")
            if senior == 1:
                risk_signals.append("Senior citizen (higher churn cohort)")

            if risk_signals:
                st.markdown("**Risk signals:**")
                for sig in risk_signals:
                    st.markdown(f"- {sig}")

        # ── SHAP Waterfall ─────────────────────────────────────────────────
        if shap_data is not None and "XGBoost" in meta["model_name"]:
            @st.cache_resource
            def get_prep_and_booster():
                prep = model.named_steps["prep"]
                clf = model.named_steps["clf"]
                return prep, clf.get_booster()

            try:
                import xgboost as xgb_lib
                prep_sim, booster_sim = get_prep_and_booster()
                X_prep_sim = prep_sim.transform(input_df).astype(np.float32)
                dmat_sim = xgb_lib.DMatrix(X_prep_sim)
                contribs_sim = booster_sim.predict(dmat_sim, pred_contribs=True)
                # contribs_sim shape: (1, n_features + 1)
                sv_sim = contribs_sim[0, :-1]       # feature contributions
                exp_val = float(contribs_sim[0, -1])  # bias term

                feature_names_sim = shap_data["feature_names"]
                top_n_wf = 10
                sorted_idx_abs = np.argsort(np.abs(sv_sim))[-top_n_wf:][::-1]
                sorted_idx_wf = sorted_idx_abs[np.argsort(sv_sim[sorted_idx_abs])]

                sv_sorted = sv_sim[sorted_idx_wf]
                feat_sorted = [feature_names_sim[i] for i in sorted_idx_wf]

                # Compute running bases for waterfall
                bases = []
                running = exp_val
                for v in sv_sorted:
                    bases.append(running)
                    running += v

                colors_wf = [C_DANGER if v > 0 else C_PRIMARY for v in sv_sorted]

                fig_wf = go.Figure(go.Bar(
                    x=sv_sorted,
                    y=feat_sorted,
                    orientation="h",
                    base=bases,
                    marker_color=colors_wf,
                    text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in sv_sorted],
                    textposition="outside",
                ))
                fig_wf.add_vline(x=exp_val, line_dash="dot", line_color="gray",
                                  annotation_text=f"Base: {exp_val:.3f}")
                final_pred = exp_val + sv_sim.sum()
                fig_wf.add_vline(x=final_pred, line_dash="solid", line_color=risk_color,
                                  annotation_text=f"Prediction: {final_pred:.3f}")
                fig_wf.update_layout(
                    title=f"SHAP Waterfall — Why this customer scores {prob:.1%}",
                    xaxis_title="Model output (log-odds space)",
                    height=420,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#111827"),
                    xaxis=dict(tickfont=dict(color="#111827"), title_font=dict(color="#111827")),
                    yaxis=dict(tickfont=dict(color="#111827")),
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                st.markdown(f'<div style="background:rgba(99,91,255,0.12);border-left:4px solid #635BFF;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;">Red bars push the prediction toward churn; blue bars push away from churn. Base value = {exp_val:.3f} (population average).</div>',
                            unsafe_allow_html=True)
            except Exception as e:
                st.info(f"SHAP waterfall not available: {e}")

        # ── Survival Profile ───────────────────────────────────────────────
        st.subheader("Survival Profile — Cox PH Estimate")
        try:
            from lifelines import KaplanMeierFitter

            cox_input = pd.DataFrame([{
                "Contract": contract,
                "InternetService": internet,
                "MonthlyCharges": monthly,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "OnlineSecurity": online_security,
                "TechSupport": tech_support,
                "PaymentMethod": payment,
                "MultipleLines": multiple_lines,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
            }])

            sf = cox.predict_survival_function(cox_input)
            times = sf.index.tolist()
            survival_vals = sf.iloc[:, 0].tolist()

            # Population average survival (baseline)
            baseline_sf = cox.baseline_survival_
            bl_times = baseline_sf.index.tolist()
            bl_vals = baseline_sf.iloc[:, 0].tolist()

            fig_surv = go.Figure()
            fig_surv.add_trace(go.Scatter(
                x=bl_times, y=bl_vals,
                mode="lines", name="Population average",
                line=dict(color=C_NEUTRAL, width=1.5, dash="dash"),
            ))
            fig_surv.add_trace(go.Scatter(
                x=times, y=survival_vals,
                mode="lines", name="This customer",
                line=dict(color=risk_color, width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({','.join(str(int(h,16)) for h in [risk_color[1:3], risk_color[3:5], risk_color[5:7]])},0.1)",
            ))
            fig_surv.add_vline(x=tenure, line_dash="dot", line_color=C_WARNING,
                               annotation_text=f"Current tenure: {tenure}mo")
            fig_surv.add_hline(y=0.5, line_dash="dot", line_color="gray",
                               annotation_text="50% survival")
            fig_surv.update_layout(
                title="Predicted Survival Function (Cox PH)",
                xaxis_title="Months as Customer",
                yaxis_title="Survival Probability",
                yaxis=dict(tickformat=".0%", range=[0, 1.05]),
                height=380,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#111827"),
            )
            st.plotly_chart(fig_surv, use_container_width=True)
        except Exception as e:
            st.info(f"Survival profile not available: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Revenue Impact
# ─────────────────────────────────────────────────────────────────────────────
elif section == "Revenue Impact":
    st.markdown('<div style="font-size:1.7rem;font-weight:700;color:inherit;margin-bottom:0.3rem;line-height:1.2;">Revenue Impact & ROI</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;margin-bottom:1.5rem;opacity:0.75;">Quantifying the business value of the churn model</div>',
                unsafe_allow_html=True)

    lift_data = meta["lift_data"]
    n_customers = meta["n_samples"]
    churn_rate = meta["churn_rate"]
    avg_monthly = meta["avg_monthly_charges"]
    best_name = meta["model_name"]

    # ── Lift Curve ────────────────────────────────────────────────────────────
    st.subheader("Lift Curve")
    pct_t = np.array(lift_data["pct_targeted"])
    pct_c = np.array(lift_data["pct_churn_captured"])

    fig_lift = go.Figure()
    fig_lift.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random baseline",
        line=dict(color="gray", dash="dash"),
    ))
    fig_lift.add_trace(go.Scatter(
        x=pct_t.tolist(),
        y=pct_c.tolist(),
        mode="lines",
        name=f"{best_name} model",
        line=dict(color=C_PRIMARY, width=2.5),
        fill="tonexty",
        fillcolor="rgba(99,91,255,0.1)",
    ))

    # Annotate at 10%, 20%, 30%
    for pct_ann in [0.10, 0.20, 0.30]:
        captured_ann = float(np.interp(pct_ann, pct_t, pct_c))
        fig_lift.add_annotation(
            x=pct_ann, y=captured_ann,
            text=f"{pct_ann:.0%} targeted → {captured_ann:.0%} captured",
            showarrow=True, arrowhead=2, arrowsize=1,
            ax=60, ay=-30, font=dict(size=10),
        )

    fig_lift.update_layout(
        title="Lift Curve — Churn Captured vs. Customers Targeted",
        xaxis_title="% of Customers Targeted (sorted by churn score)",
        yaxis_title="% of Churners Captured",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".0%"),
        height=460,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    # Key lift stats
    for pct_show in [0.10, 0.20, 0.30, 0.50]:
        captured_show = float(np.interp(pct_show, pct_t, pct_c))
        lift_val = captured_show / pct_show
        st.markdown(
            f"Targeting the **top {pct_show:.0%}** of customers by churn score captures "
            f"**{captured_show:.1%}** of all churners — a **{lift_val:.1f}x lift** vs. random."
        )

    st.markdown("---")

    # ── ROI Calculator ────────────────────────────────────────────────────────
    st.subheader("ROI Calculator")
    col_sliders, col_metrics = st.columns(2)

    with col_sliders:
        pct_target = st.slider("% of customers to target for retention", 5, 50, 20, step=5)
        offer_cost = st.slider("Retention offer value per customer ($)", 5, 100, 25, step=5)
        save_rate = st.slider("Assumed save rate (% of captured churners retained)", 10, 60, 30, step=5)

        st.caption(f"Dataset avg monthly charges: **${avg_monthly:.2f}**/customer")

    with col_metrics:
        pct_captured = float(np.interp(pct_target / 100, pct_t, pct_c))
        total_churners = int(n_customers * churn_rate)
        churners_captured = int(total_churners * pct_captured)
        customers_targeted = int(n_customers * pct_target / 100)
        churners_saved = int(churners_captured * save_rate / 100)
        revenue_saved = churners_saved * avg_monthly * 12
        campaign_cost = customers_targeted * offer_cost
        net_roi = revenue_saved - campaign_cost

        st.metric("Customers targeted", f"{customers_targeted:,}")
        st.metric("Churners captured", f"{churners_captured:,} ({pct_captured:.1%} of total)")
        st.metric("Revenue saved (est.)", f"${revenue_saved:,.0f}/yr",
                  help=f"Assuming {save_rate}% of captured churners convert with offer × $12 annual value")
        st.metric("Campaign cost", f"${campaign_cost:,.0f}")
        st.metric(
            "Net ROI",
            f"${net_roi:,.0f}",
            delta="profitable" if net_roi > 0 else "loss",
            delta_color="normal" if net_roi > 0 else "inverse",
        )

    st.markdown("---")

    # ── Sensitivity Chart ─────────────────────────────────────────────────────
    st.subheader("Sensitivity Analysis — Net ROI vs. Targeting %")
    pct_range = np.arange(0.05, 0.55, 0.05)
    net_rois = []
    for pct in pct_range:
        cap = float(np.interp(pct, pct_t, pct_c))
        churners_cap = int(total_churners * cap)
        cust_targeted = int(n_customers * pct)
        saved = int(churners_cap * save_rate / 100)
        rev = saved * avg_monthly * 12
        cost = cust_targeted * offer_cost
        net_rois.append(rev - cost)

    optimal_idx = int(np.argmax(net_rois))
    optimal_pct = pct_range[optimal_idx]
    optimal_roi = net_rois[optimal_idx]

    fig_sens = go.Figure()
    colors_bar = [C_SAFE if v > 0 else C_DANGER for v in net_rois]
    fig_sens.add_trace(go.Bar(
        x=[f"{int(p*100)}%" for p in pct_range],
        y=net_rois,
        marker_color=colors_bar,
        name="Net ROI",
    ))
    fig_sens.add_annotation(
        x=f"{int(optimal_pct*100)}%",
        y=optimal_roi,
        text=f"Optimal: {int(optimal_pct*100)}% → ${optimal_roi:,.0f}",
        showarrow=True, arrowhead=2, bgcolor="white", borderpad=4,
    )
    fig_sens.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_sens.update_layout(
        title=f"Net ROI by Targeting % (offer=${offer_cost}, save rate={save_rate}%)",
        xaxis_title="% Customers Targeted",
        yaxis_title="Net ROI ($)",
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#111827"),
    )
    st.plotly_chart(fig_sens, use_container_width=True)
    st.markdown(
        f'<div style="background:rgba(99,91,255,0.12);border-left:4px solid #635BFF;padding:12px 16px;border-radius:4px;margin:12px 0;color:inherit;">Optimal strategy: target the top <b>{int(optimal_pct*100)}%</b> of customers by churn score for an estimated net ROI of <b>${optimal_roi:,.0f}</b>.</div>',
        unsafe_allow_html=True
    )
