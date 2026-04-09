"""
=============================================================================
BRFSS 2023 — Mental Health Prediction Study
Script 03: Machine Learning Modeling
=============================================================================
Run AFTER brfss_01_preprocess.py

Models trained:
  1. Logistic Regression      (baseline, interpretable)
  2. Random Forest            (ensemble, handles nonlinearity)
  3. XGBoost                  (gradient boosting, primary model)
  4. LightGBM                 (gradient boosting, fast, primary model)

Evaluation:
  - 5-fold stratified cross-validation
  - Metrics: AUROC, AUPRC, F1, Precision, Recall, Brier Score
  - Calibration curves
  - ROC curves (all models overlaid)

Outputs (saved to models/ and figures/):
  - models/best_model.pkl             (best model by AUROC)
  - models/all_cv_results.csv         (full CV metrics table)
  - models/feature_list.pkl           (ordered feature names)
  - figures/fig06_roc_curves.png      (ROC all models)
  - figures/fig07_calibration.png     (calibration curves)
  - figures/fig08_cv_metrics.png      (CV metrics comparison bar chart)
  - models/modeling_report.txt        (summary report)
=============================================================================
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
FIG_DIR    = BASE_DIR / "figures"
MODEL_DIR  = BASE_DIR / "models"
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

PROC_PKL   = DATA_DIR / "brfss_2023_processed.pkl"

# ── Plot style ─────────────────────────────────────────────────────────────
C1, C2, C3, C4 = "#E63946", "#457B9D", "#2A9D8F", "#E9C46A"
MODEL_COLORS = {"Logistic Regression": C1, "Random Forest": C2, "LightGBM": C3}


# =============================================================================
# SECTION 1 — Feature Selection & Preparation
# =============================================================================

# Variables to EXCLUDE from features
# (outcome-related, identifiers, weights, or entirely missing)
EXCLUDE = [
    "MENTHLTH",     # raw outcome — never a feature
    "FMD",          # binary outcome
    "STATE_NAME",   # string label
    "_LLCPWT",      # survey weight (not a predictor)
    "SLEEP_CAT",    # categorical version of SLEPTIM1 (redundant)
    # Entirely missing optional modules (from QC report)
    "INDORTAN", "NUMBURN3", "SUNPRTCT", "WKDAYOUT", "WKENDOUT",
]

# Categorical variables that need one-hot encoding
OHE_COLS = [
    "_RACE1",    # 8 categories
    "MARITAL",   # 6 categories
    "EMPLOY1",   # 8 categories
    "RENTHOM1",  # 3 categories
]

# Ordinal/continuous — keep as-is
ORDINAL_COLS = [
    "_AGE_G", "EDUCA", "_INCOMG1", "_BMI5CAT", "GENHLTH",
    "SLEPTIM1", "PHYSHLTH", "_STATE",
]


def prepare_features(df: pd.DataFrame):
    """
    Build X (feature matrix) and y (outcome vector).
    - Drop excluded columns
    - One-hot encode categorical variables
    - Return X (DataFrame), y (Series), feature_names (list)
    """
    df = df.copy()

    # Drop excluded
    drop_cols = [c for c in EXCLUDE if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    # One-hot encode nominal categoricals
    ohe_present = [c for c in OHE_COLS if c in feature_df.columns]
    if ohe_present:
        feature_df = pd.get_dummies(feature_df, columns=ohe_present,
                                    drop_first=True, dtype=float)

    # Ensure all remaining columns are numeric
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    # Fill any remaining NaN (safety net)
    feature_df = feature_df.fillna(feature_df.median())

    y = df["FMD"].astype(int)
    X = feature_df

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Outcome balance: {y.mean()*100:.1f}% positive (FMD=1)")

    return X, y, list(X.columns)


# =============================================================================
# SECTION 2 — Model Definitions
# =============================================================================

def get_models(scale_pos_weight: float):
    """
    Return dict of model name → sklearn-compatible estimator.
    scale_pos_weight = n_negative / n_positive (handles class imbalance).
    """
    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                C=0.1,
                random_state=42,
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),

        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
    }
    return models


# =============================================================================
# SECTION 3 — Cross-Validation
# =============================================================================

def evaluate_fold(model, X_train, y_train, X_val, y_val):
    """Train model on one fold, return metrics dict."""
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auroc":     roc_auc_score(y_val, y_prob),
        "auprc":     average_precision_score(y_val, y_prob),
        "f1":        f1_score(y_val, y_pred, zero_division=0),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall":    recall_score(y_val, y_pred, zero_division=0),
        "brier":     brier_score_loss(y_val, y_prob),
    }


def run_cv(models: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Run stratified k-fold CV for all models.
    Returns:
      - cv_results: dict of model_name → list of fold metric dicts
      - oof_probs:  dict of model_name → out-of-fold predicted probabilities
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_arr = X.values
    y_arr = y.values

    cv_results = {name: [] for name in models}
    oof_probs  = {name: np.zeros(len(y)) for name in models}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        print(f"  Fold {fold}/{n_splits}...")
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        for name, model in models.items():
            import copy
            m = copy.deepcopy(model)
            metrics = evaluate_fold(m, X_train, y_train, X_val, y_val)
            cv_results[name].append(metrics)
            oof_probs[name][val_idx] = m.predict_proba(X_val)[:, 1]
            print(f"    {name:<22} AUROC={metrics['auroc']:.4f}  "
                  f"F1={metrics['f1']:.4f}  Brier={metrics['brier']:.4f}")

    return cv_results, oof_probs


def summarize_cv(cv_results: dict) -> pd.DataFrame:
    """Aggregate CV fold metrics into mean ± std DataFrame."""
    rows = []
    for name, folds in cv_results.items():
        fold_df = pd.DataFrame(folds)
        row = {"Model": name}
        for col in fold_df.columns:
            row[f"{col}_mean"] = fold_df[col].mean()
            row[f"{col}_std"]  = fold_df[col].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# =============================================================================
# SECTION 4 — Final Model Training (full dataset)
# =============================================================================

def train_final_model(model, X: pd.DataFrame, y: pd.Series, name: str):
    """Train model on full dataset and return fitted model."""
    import copy
    print(f"  Training final {name} on full dataset...")
    m = copy.deepcopy(model)
    m.fit(X.values, y.values)
    return m


# =============================================================================
# SECTION 5 — Figures
# =============================================================================

def fig06_roc_curves(oof_probs: dict, y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, probs in oof_probs.items():
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})",
                color=MODEL_COLORS[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Out-of-Fold Predictions\nBRFSS 2023 FMD Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig06_roc_curves.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig07_calibration(oof_probs: dict, y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    for name, probs in oof_probs.items():
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=15,
                                                 strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", label=name,
                color=MODEL_COLORS[name], linewidth=2, markersize=5)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves — Out-of-Fold Predictions\nBRFSS 2023 FMD Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig07_calibration.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig08_cv_metrics(summary_df: pd.DataFrame) -> None:
    metrics = ["auroc", "auprc", "f1", "precision", "recall", "brier"]
    titles  = ["AUROC", "AUPRC", "F1 Score", "Precision", "Recall", "Brier Score"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("5-Fold CV Performance Comparison — BRFSS 2023 FMD Prediction",
                 fontsize=13, fontweight="bold")
    models = summary_df.index.tolist()
    x = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        means = summary_df[f"{metric}_mean"].values
        stds  = summary_df[f"{metric}_std"].values
        bars = ax.bar(x, means, yerr=stds, color=colors,
                      capsize=5, edgecolor="white", linewidth=0.5, width=0.6)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[list(means).index(val)] + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        # Lower is better for Brier
        if metric == "brier":
            ax.set_title(f"{title} (↓ better)", fontweight="bold", fontsize=11)

    plt.tight_layout()
    out = FIG_DIR / "fig08_cv_metrics.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# SECTION 6 — Modeling Report
# =============================================================================

def generate_report(summary_df: pd.DataFrame, best_name: str,
                    feature_names: list, report_path: Path) -> None:
    lines = []
    lines.append("=" * 65)
    lines.append("BRFSS 2023 — MODELING REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 65)

    lines.append("\n── 5-Fold CV Results (Mean ± SD) ───────────────────────────")
    lines.append(f"\n  {'Model':<22} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} "
                 f"{'Precision':>10} {'Recall':>8} {'Brier':>8}")
    lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for model in summary_df.index:
        r = summary_df.loc[model]
        lines.append(
            f"  {model:<22} "
            f"{r['auroc_mean']:.4f}±{r['auroc_std']:.3f}  "
            f"{r['auprc_mean']:.4f}±{r['auprc_std']:.3f}  "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.3f}  "
            f"{r['precision_mean']:.4f}±{r['precision_std']:.3f}  "
            f"{r['recall_mean']:.4f}±{r['recall_std']:.3f}  "
            f"{r['brier_mean']:.4f}±{r['brier_std']:.3f}"
        )

    lines.append(f"\n── Best Model ───────────────────────────────────────────────")
    lines.append(f"  {best_name}  (highest mean AUROC)")
    best = summary_df.loc[best_name]
    lines.append(f"  AUROC:     {best['auroc_mean']:.4f} ± {best['auroc_std']:.4f}")
    lines.append(f"  AUPRC:     {best['auprc_mean']:.4f} ± {best['auprc_std']:.4f}")
    lines.append(f"  F1:        {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
    lines.append(f"  Brier:     {best['brier_mean']:.4f} ± {best['brier_std']:.4f}")

    lines.append(f"\n── Feature Count ────────────────────────────────────────────")
    lines.append(f"  Total features after encoding: {len(feature_names)}")

    lines.append("\n" + "=" * 65)
    text = "\n".join(lines)
    print(text)
    with open(report_path, "w") as f:
        f.write(text)
    print(f"\n✓ Report saved to {report_path.name}")


# =============================================================================
# SECTION 7 — Main Pipeline
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  BRFSS 2023 — Machine Learning Modeling")
    print("=" * 60 + "\n")

    # Load processed data
    print("[Step 1] Loading processed data...")
    if not PROC_PKL.exists():
        raise FileNotFoundError("Run brfss_01_preprocess.py first.")
    with open(PROC_PKL, "rb") as f:
        df = pickle.load(f)
    print(f"  Loaded {len(df):,} rows")

    # Prepare features
    print("\n[Step 2] Preparing feature matrix...")
    X, y, feature_names = prepare_features(df)

    # Class imbalance ratio
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    # Save feature list
    with open(MODEL_DIR / "feature_list.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    # Define models
    print("\n[Step 3] Defining models...")
    models = get_models(scale_pos_weight)
    for name in models:
        print(f"  • {name}")

    # Run cross-validation
    print(f"\n[Step 4] Running 5-fold stratified cross-validation...")
    print(f"  (This will take 5–15 minutes depending on your machine)\n")
    cv_results, oof_probs = run_cv(models, X, y, n_splits=5)

    # Summarize
    print("\n[Step 5] Summarizing CV results...")
    summary_df = summarize_cv(cv_results)
    summary_df.to_csv(MODEL_DIR / "all_cv_results.csv")
    print(summary_df[["auroc_mean", "auprc_mean", "f1_mean", "brier_mean"]].round(4))

    # Identify best model
    best_name = summary_df["auroc_mean"].idxmax()
    print(f"\n  ★ Best model by AUROC: {best_name}")

    # Train final model on full dataset
    print("\n[Step 6] Training final model on full dataset...")
    final_model = train_final_model(models[best_name], X, y, best_name)
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": final_model,
                     "features": feature_names}, f)
    print(f"  ✓ Saved to models/best_model.pkl")

    # Save all fitted models (needed for SHAP script)
    print("\n[Step 7] Training and saving all final models for SHAP...")
    all_final = {}
    for name, model in models.items():
        m = train_final_model(model, X, y, name)
        all_final[name] = m
    with open(MODEL_DIR / "all_final_models.pkl", "wb") as f:
        pickle.dump({"models": all_final, "features": feature_names}, f)
    print("  ✓ Saved to models/all_final_models.pkl")

    # Generate figures
    print("\n[Step 8] Generating figures...")
    fig06_roc_curves(oof_probs, y)
    fig07_calibration(oof_probs, y)
    fig08_cv_metrics(summary_df)

    # Generate report
    print("\n[Step 9] Generating modeling report...")
    generate_report(summary_df, best_name, feature_names,
                    MODEL_DIR / "modeling_report.txt")

    print("\n✅ Modeling complete. Ready for SHAP analysis (Script 04).\n")


if __name__ == "__main__":
    main()
