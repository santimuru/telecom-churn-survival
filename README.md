# Telecom Churn Survival Analysis

> **Live demo:** [telecom-churn-survival-santiagomuru.streamlit.app](https://telecom-churn-survival-santiagomuru.streamlit.app/)

Traditional churn models answer *will this customer leave?* — a binary yes/no. This project goes further: **when will they leave?** Survival analysis treats customer tenure as a time-to-event problem, quantifying urgency alongside likelihood and enabling prioritized retention interventions.

Applied a similar methodology in a production environment at a cable operator, scoring ~40,000 customers weekly for proactive outreach.

---

## Live Dashboard

| Section            | What you'll find                                                              |
| ------------------ | ----------------------------------------------------------------------------- |
| 📊 Overview        | Business context, KPIs, methodology summary, key findings                    |
| 📈 Survival        | Kaplan-Meier curves by segment, survival heatmap, Cox PH forest plot          |
| 🎯 Model Perf.     | ROC, Precision-Recall, calibration curves — LR vs. RF vs. XGBoost            |
| 🔍 SHAP Explorer   | Global beeswarm, mean SHAP bar chart, per-feature dependence plots            |
| 🔮 Simulator       | Enter any customer profile: churn probability + SHAP waterfall + Cox curve   |
| 💰 Revenue Impact  | Lift curve, ROI calculator, sensitivity analysis vs. targeting %             |

---

## Why Survival Analysis

Most portfolios classify churn as binary. Survival analysis adds the time dimension:

| Technique                     | What it models                                                         |
| ----------------------------- | ---------------------------------------------------------------------- |
| **Kaplan-Meier**              | Non-parametric survival curves by segment, with log-rank significance  |
| **Cox Proportional Hazards**  | Hazard ratios per feature — which variables accelerate or protect      |
| **XGBoost classifier**        | Overall churn probability optimized for AUC-ROC                        |
| **SHAP TreeExplainer**        | Local + global feature attribution for the classification model        |

Together they answer: who is at risk, how urgent is it, and why — in one unified dashboard.

---

## Key Results

- **Contract type** is the strongest predictor: month-to-month customers have a median survival of ~18 months vs. >60 months for 2-year contracts.
- **Targeting the top 20%** by churn score captures the majority of all churners — a 3-4x lift over random selection.
- **Online Security and Tech Support** are the strongest protective factors in the Cox model (HR < 1, p < 0.001).
- Monthly charges above $65 increase churn hazard independently of contract type.

---

## Dataset

IBM Telco Customer Churn — 7,043 customers, 20 features, 26.5% churn rate. Publicly available on Kaggle.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train models (downloads dataset automatically if not present):

```bash
python src/train.py
```

Run the dashboard:

```bash
streamlit run app/app.py
```

---

## Project Structure

```
telecom-churn-survival/
├── app/app.py          Streamlit dashboard (6 sections)
├── src/train.py        Training pipeline: KM + Cox + classification + SHAP
├── models/             Serialized artifacts (joblib)
├── data/               Dataset (Telco-Customer-Churn.csv)
└── requirements.txt
```

---

## Author

**Santiago Martinez** - Data Analyst

- Portfolio: https://santimuru.github.io
- LinkedIn: https://www.linkedin.com/in/santiago-martinez-pezzatti-4241a3165/
- GitHub: https://github.com/santimuru
