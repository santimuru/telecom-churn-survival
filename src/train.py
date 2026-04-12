"""
Telecom Churn Survival Analysis — Training Script
Trains XGBoost, Random Forest, and Logistic Regression classifiers,
fits a Cox PH model and Kaplan-Meier curves, computes SHAP values,
and saves all artifacts to models/.
"""

import os
import sys
import warnings
import requests
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"


# ── 1. Download data ──────────────────────────────────────────────────────────
def download_data():
    if DATA_PATH.exists():
        print(f"[data] Already exists: {DATA_PATH}")
    else:
        print(f"[data] Downloading from {DATA_URL} ...")
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        DATA_PATH.write_bytes(r.content)
        print(f"[data] Saved to {DATA_PATH}")


# ── 2. Load & clean ───────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    df.drop(columns=["customerID"], inplace=True)
    print(f"[data] Loaded {len(df):,} rows, churn rate = {df['Churn_bin'].mean():.2%}")
    return df


# ── 3. Kaplan-Meier curves ────────────────────────────────────────────────────
def build_km_data(df):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    segments = {
        "Contract": df["Contract"].unique().tolist(),
        "InternetService": df["InternetService"].unique().tolist(),
        "PaymentMethod": df["PaymentMethod"].unique().tolist(),
        "SeniorCitizen": [0, 1],
    }
    senior_labels = {0: "Non-senior", 1: "Senior (65+)"}

    km_data = {}
    for seg_col, groups in segments.items():
        km_data[seg_col] = {}
        group_labels = []
        durations_list = []
        events_list = []

        for grp in groups:
            if seg_col == "SeniorCitizen":
                mask = df[seg_col] == grp
                label = senior_labels[grp]
            else:
                mask = df[seg_col] == grp
                label = str(grp)

            sub = df[mask]
            if len(sub) < 10:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(sub["tenure"], event_observed=sub["Churn_bin"], label=label)

            timeline = kmf.timeline.tolist()
            survival = kmf.survival_function_[label].tolist()
            ci_lower = kmf.confidence_interval_[f"{label}_lower_0.95"].tolist()
            ci_upper = kmf.confidence_interval_[f"{label}_upper_0.95"].tolist()

            # Median survival
            try:
                median_val = float(kmf.median_survival_time_)
                if np.isnan(median_val) or np.isinf(median_val):
                    median_val = None
            except Exception:
                median_val = None

            n = int(len(sub))
            n_events = int(sub["Churn_bin"].sum())
            churn_rate = float(sub["Churn_bin"].mean())

            km_data[seg_col][label] = {
                "timeline": timeline,
                "survival": survival,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "median": median_val,
                "n": n,
                "n_events": n_events,
                "churn_rate": churn_rate,
            }

            group_labels.append(label)
            durations_list.append(sub["tenure"].values)
            events_list.append(sub["Churn_bin"].values)

        # Log-rank test
        try:
            all_durations = np.concatenate(durations_list)
            all_events = np.concatenate(events_list)
            all_groups = np.concatenate(
                [np.full(len(d), lbl) for d, lbl in zip(durations_list, group_labels)]
            )
            result = multivariate_logrank_test(
                all_durations, all_groups, all_events
            )
            km_data[seg_col]["_logrank_pvalue"] = float(result.p_value)
        except Exception as e:
            print(f"  [warn] log-rank test failed for {seg_col}: {e}")
            km_data[seg_col]["_logrank_pvalue"] = float("nan")

        print(f"[km] {seg_col}: {len(groups)} groups, p={km_data[seg_col]['_logrank_pvalue']:.4e}")

    return km_data


# ── 4. Cox PH model ───────────────────────────────────────────────────────────
def build_cox_model(df):
    from lifelines import CoxPHFitter

    cox_cols = [
        "tenure", "Churn_bin", "Contract", "InternetService", "MonthlyCharges",
        "SeniorCitizen", "Partner", "Dependents", "OnlineSecurity", "TechSupport",
        "PaymentMethod", "MultipleLines", "StreamingTV", "StreamingMovies",
    ]
    cox_df = df[cox_cols].copy()

    # Encode binary cat columns so CoxPH formula doesn't break
    for col in ["Partner", "Dependents", "OnlineSecurity", "TechSupport",
                "MultipleLines", "StreamingTV", "StreamingMovies"]:
        cox_df[col] = cox_df[col].astype(str)

    formula = (
        "Contract + InternetService + MonthlyCharges + SeniorCitizen + "
        "Partner + Dependents + OnlineSecurity + TechSupport + "
        "PaymentMethod + MultipleLines + StreamingTV + StreamingMovies"
    )

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_df, duration_col="tenure", event_col="Churn_bin", formula=formula)
    print(f"[cox] Fitted. Concordance = {cph.concordance_index_:.4f}")
    return cph


# ── 5. Classification models ──────────────────────────────────────────────────
def build_classifiers(df):
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve,
    )
    from sklearn.calibration import calibration_curve
    import xgboost as xgb

    # Feature columns
    drop_cols = ["Churn", "Churn_bin"]
    X = df.drop(columns=drop_cols)
    y = df["Churn_bin"]

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"[clf] Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pw = neg / pos

    models_dict = {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocessor),
            ("clf", xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=3, gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
                scale_pos_weight=scale_pw,
                eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
        ]),
    }

    all_metrics = {}
    for name, pipeline in models_dict.items():
        print(f"[clf] Training {name} ...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "avg_precision": float(average_precision_score(y_test, y_prob)),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": prec_vals.tolist(), "recall": rec_vals.tolist()},
            "calibration_curve": {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()},
            "y_prob": y_prob,
        }
        all_metrics[name] = metrics
        print(f"  AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"Recall={metrics['recall']:.4f}")

    # Best by AUC
    best_name = max(all_metrics, key=lambda n: all_metrics[n]["roc_auc"])
    best_pipeline = models_dict[best_name]
    best_metrics = all_metrics[best_name]
    print(f"[clf] Best model: {best_name} (AUC={best_metrics['roc_auc']:.4f})")

    # Feature importance
    prep = best_pipeline.named_steps["prep"]
    clf = best_pipeline.named_steps["clf"]

    ohe_cols = prep.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + ohe_cols

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Lift curve from best model
    y_prob_best = best_metrics["y_prob"]
    df_lift = pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob_best})
    df_lift = df_lift.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df_lift["pct_targeted"] = (df_lift.index + 1) / len(df_lift)
    df_lift["pct_churn_captured"] = df_lift["y_true"].cumsum() / df_lift["y_true"].sum()
    df_lift["lift"] = df_lift["pct_churn_captured"] / df_lift["pct_targeted"]
    step = max(1, len(df_lift) // 200)
    lift_data = {
        "pct_targeted": df_lift.iloc[::step]["pct_targeted"].tolist(),
        "pct_churn_captured": df_lift.iloc[::step]["pct_churn_captured"].tolist(),
        "lift": df_lift.iloc[::step]["lift"].tolist(),
    }

    return (
        best_name, best_pipeline, best_metrics, all_metrics,
        X_train, X_test, y_train, y_test,
        num_cols, cat_cols, feature_names, importance_df, lift_data, X, y,
    )


# ── 6. SHAP values ────────────────────────────────────────────────────────────
def build_shap_data(pipeline, X_train, X_test, feature_names, model_name):
    if "XGBoost" not in model_name:
        print(f"[shap] Skipping SHAP (best model is {model_name}, not XGBoost)")
        return None

    import shap

    print("[shap] Computing SHAP values ...")
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    X_test_prep = prep.transform(X_test)
    X_bg_prep = prep.transform(X_train.sample(100, random_state=42))

    n_shap = min(500, len(X_test_prep))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test_prep), n_shap, replace=False)

    # Use XGBoost native SHAP (pred_contribs) — avoids SHAP library compatibility issues
    import xgboost as xgb
    booster = clf.get_booster()
    X_sample = X_test_prep[idx].astype(np.float32)
    dmat = xgb.DMatrix(X_sample)
    contribs = booster.predict(dmat, pred_contribs=True)
    # contribs shape: (n, n_features + 1) — last col is bias/expected_value
    expected_value = float(contribs[:, -1].mean())
    shap_values = contribs[:, :-1]  # (n, n_features)

    shap_data = {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "X_test_prep": X_test_prep[idx],
        "X_test_raw": X_test.iloc[idx].reset_index(drop=True),
    }
    print(f"[shap] Done. shape={shap_values.shape}, expected_value={expected_value:.4f}")
    return shap_data


# ── 7. Save artifacts ─────────────────────────────────────────────────────────
def save_artifacts(pipeline, cox_model, km_data, shap_data, meta):
    joblib.dump(pipeline, MODELS_DIR / "churn_model.pkl")
    joblib.dump(cox_model, MODELS_DIR / "cox_model.pkl")
    joblib.dump(km_data, MODELS_DIR / "km_data.pkl")
    joblib.dump(shap_data, MODELS_DIR / "shap_data.pkl")
    joblib.dump(meta, MODELS_DIR / "model_meta.pkl")
    print("[save] All artifacts saved to models/")
    for p in (MODELS_DIR).iterdir():
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:30s} {size_kb:8.1f} KB")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Telecom Churn — Training Pipeline")
    print("=" * 60)

    download_data()
    df = load_data()

    print("\n[step 1/4] Building Kaplan-Meier curves ...")
    km_data = build_km_data(df)

    print("\n[step 2/4] Fitting Cox PH model ...")
    cox_model = build_cox_model(df)

    print("\n[step 3/4] Training classifiers ...")
    (
        best_name, best_pipeline, best_metrics, all_metrics,
        X_train, X_test, y_train, y_test,
        num_cols, cat_cols, feature_names, importance_df, lift_data, X, y,
    ) = build_classifiers(df)

    print("\n[step 4/4] Computing SHAP values ...")
    shap_data = build_shap_data(
        best_pipeline, X_train, X_test, feature_names, best_name
    )

    # Build meta
    meta = {
        "model_name": best_name,
        "metrics": {
            name: {k: v for k, v in m.items() if k != "y_prob"}
            for name, m in all_metrics.items()
        },
        "best_metrics": {k: v for k, v in best_metrics.items() if k != "y_prob"},
        "feature_importance": importance_df,
        "feature_names_raw": X.columns.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "churn_rate": round(float(y.mean()), 4),
        "n_samples": len(df),
        "lift_data": lift_data,
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "background_sample": X_train.sample(100, random_state=42).reset_index(drop=True),
    }

    save_artifacts(best_pipeline, cox_model, km_data, shap_data, meta)

    print("\n" + "=" * 60)
    print(f"DONE. Best model: {best_name}")
    print(f"  AUC-ROC   : {best_metrics['roc_auc']:.4f}")
    print(f"  F1 score  : {best_metrics['f1']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
