"""
=============================================================================
BRFSS 2023 — Mental Health Prediction Study
Script 04: SHAP Interpretability & State-Level Analysis
=============================================================================
Run AFTER brfss_03_modeling.py

This script produces the novel contribution of the paper:
  1. Global SHAP — which features drive FMD nationally
  2. SHAP dependence plots — how top features relate to FMD risk
  3. State-level SHAP — which features drive FMD *differently* by state
  4. Policy simulation — counterfactual SHAP scenarios

Outputs (saved to figures/ and results/):
  - figures/fig09_shap_beeswarm.png         (global feature importance)
  - figures/fig10_shap_bar.png              (mean |SHAP| bar chart)
  - figures/fig11_shap_dependence.png       (top 4 dependence plots)
  - figures/fig12_state_shap_heatmap.png    (state × feature SHAP heatmap)
  - figures/fig13_state_top_predictors.png  (top predictor per state map)
  - results/global_shap_importance.csv      (feature importance table)
  - results/state_shap_summary.csv          (state-level SHAP summary)
  - results/shap_report.txt                 (narrative summary)
=============================================================================
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
FIG_DIR     = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
SHAP_POS  = "#E63946"
SHAP_NEG  = "#457B9D"
SHAP_CMAP = "RdBu_r"

# ── Human-readable feature labels ─────────────────────────────────────────
FEATURE_LABELS = {
    "_AGE_G":       "Age Group",
    "FEMALE":       "Female Sex",
    "EDUCA":        "Education Level",
    "_INCOMG1":     "Income Category",
    "MARITAL":      "Marital Status",
    "EMPLOY1":      "Employment Status",
    "_STATE":       "State",
    "HLTHPLN2":     "Has Health Insurance",
    "MEDCOST1":     "Cannot Afford Doctor",
    "FOODSTMP":     "Receives SNAP/Food Stamps",
    "RENTHOM1":     "Housing Status",
    "EXERANY2":     "Physical Activity",
    "SMOKE100":     "Ever Smoked",
    "DRNKANY6":     "Alcohol Use",
    "HEAVY_DRINKER":"Heavy Drinker",
    "_BMI5CAT":     "BMI Category",
    "SLEPTIM1":     "Sleep Hours",
    "PHYSHLTH":     "Poor Physical Health Days",
    "GENHLTH":      "General Health Rating",
    "DIABETE4":     "Diabetes",
    "BPHIGH6":      "Hypertension",
    "ASTHMA3":      "Asthma",
    "CHCCOPD3":     "COPD",
    "CVDCRHD4":     "Coronary Heart Disease",
    "CVDSTRK3":     "Stroke History",
    "ADDEPEV3":     "Depression Diagnosis",
    "SEX":          "Sex",
    "SEXVAR":       "Sex",
}

STATE_FIPS = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "D.C.",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
    47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont",
    51: "Virginia", 53: "Washington", 54: "West Virginia", 55: "Wisconsin",
    56: "Wyoming",
}


# =============================================================================
# SECTION 1 — Load Model & Data
# =============================================================================

def load_artifacts():
    """Load best model, all models, processed data, and feature list."""
    print("  Loading best model...")
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        best = pickle.load(f)

    print("  Loading all final models...")
    with open(MODEL_DIR / "all_final_models.pkl", "rb") as f:
        all_models = pickle.load(f)

    print("  Loading processed data...")
    with open(DATA_DIR / "brfss_2023_processed.pkl", "rb") as f:
        df = pickle.load(f)

    feature_names = best["features"]
    model         = best["model"]
    model_name    = best["name"]

    return model, model_name, all_models, df, feature_names


def prepare_X(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reconstruct feature matrix matching training exactly."""
    from brfss_03_modeling import prepare_features
    X, y, _ = prepare_features(df)
    # Align columns to training feature list
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    return X, y


# =============================================================================
# SECTION 2 — Global SHAP
# =============================================================================

def compute_global_shap(model, X: pd.DataFrame, sample_n: int = 10000):
    """
    Compute SHAP values for LightGBM model.
    Uses TreeExplainer (exact, fast for tree models).
    Samples up to sample_n rows for speed.
    """
    print(f"  Computing SHAP values (n={min(sample_n, len(X)):,})...")
    if len(X) > sample_n:
        idx = np.random.RandomState(42).choice(len(X), sample_n, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary: shap_values may be list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"  SHAP matrix shape: {shap_values.shape}")
    return shap_values, X_sample, explainer


def get_feature_importance(shap_values: np.ndarray,
                           feature_names: list) -> pd.DataFrame:
    """Mean absolute SHAP values → feature importance DataFrame."""
    importance = np.abs(shap_values).mean(axis=0)
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    df = pd.DataFrame({
        "feature":     feature_names,
        "label":       labels,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# =============================================================================
# SECTION 3 — Figures
# =============================================================================

def fig09_shap_beeswarm(shap_values: np.ndarray,
                         X_sample: pd.DataFrame,
                         feature_names: list,
                         top_n: int = 20) -> None:
    """SHAP beeswarm summary plot — top N features."""
    importance = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:top_n]

    sv_top = shap_values[:, top_idx]
    X_top  = X_sample.iloc[:, top_idx]
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i])
              for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, 9))

    # Manual beeswarm-style plot
    for j, (col_sv, col_x, label) in enumerate(
            zip(sv_top.T, X_top.T.values, labels)):
        # Normalize feature values for color
        x_vals = col_x.astype(float)
        x_norm = (x_vals - np.nanmin(x_vals)) / (np.nanmax(x_vals) - np.nanmin(x_vals) + 1e-9)
        colors = plt.cm.RdBu_r(1 - x_norm)

        # Jitter y positions
        y_jitter = j + np.random.RandomState(j).uniform(-0.3, 0.3, len(col_sv))
        ax.scatter(col_sv, y_jitter, c=colors, s=4, alpha=0.4, linewidths=0)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1] if False else labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on FMD prediction)", fontsize=11)
    ax.set_title(f"Global SHAP Feature Importance — Top {top_n} Features\n"
                 f"LightGBM Model, BRFSS 2023 (n=10,000 sample)",
                 fontsize=12, fontweight="bold")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Feature value\n(low → high)", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig09_shap_beeswarm.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig10_shap_bar(importance_df: pd.DataFrame, top_n: int = 20) -> None:
    """Mean |SHAP| horizontal bar chart."""
    top = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [SHAP_POS if i < top_n // 2 else SHAP_NEG
              for i in range(len(top))][::-1]
    bars = ax.barh(top["label"], top["mean_abs_shap"],
                   color=colors, edgecolor="white", linewidth=0.4, height=0.7)
    ax.set_xlabel("Mean |SHAP Value| (average impact on model output)",
                  fontsize=10)
    ax.set_title(f"Feature Importance — Mean |SHAP| Values\n"
                 f"LightGBM Model, BRFSS 2023 FMD Prediction",
                 fontsize=12, fontweight="bold")
    for bar, val in zip(bars, top["mean_abs_shap"]):
        ax.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig10_shap_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig11_shap_dependence(shap_values: np.ndarray,
                           X_sample: pd.DataFrame,
                           importance_df: pd.DataFrame,
                           top_n: int = 4) -> None:
    """SHAP dependence plots for top N features."""
    top_features = importance_df.head(top_n)["feature"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SHAP Dependence Plots — Top 4 Predictors of FMD\nBRFSS 2023",
                 fontsize=13, fontweight="bold")

    for ax, feat in zip(axes.flat, top_features):
        if feat not in X_sample.columns:
            ax.set_visible(False)
            continue
        feat_idx  = list(X_sample.columns).index(feat)
        feat_vals = X_sample[feat].values.astype(float)
        sv        = shap_values[:, feat_idx]
        label     = FEATURE_LABELS.get(feat, feat)

        # Color by feature value
        norm   = plt.Normalize(np.nanpercentile(feat_vals, 2),
                               np.nanpercentile(feat_vals, 98))
        colors = plt.cm.RdBu_r(norm(feat_vals))
        ax.scatter(feat_vals, sv, c=colors, s=8, alpha=0.3, linewidths=0)

        # Smoothed trend line
        from scipy.stats import binned_statistic
        try:
            bin_means, bin_edges, _ = binned_statistic(
                feat_vals, sv, statistic="mean", bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            valid = ~np.isnan(bin_means)
            ax.plot(bin_centers[valid], bin_means[valid],
                    color="black", linewidth=2, zorder=5)
        except Exception:
            pass

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig11_shap_dependence.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig12_state_shap_heatmap(state_shap_df: pd.DataFrame) -> None:
    """
    Heatmap: states (rows) × top features (columns), colored by mean SHAP.
    This is the key novel figure for the paper.
    """
    # Pivot: states as rows, features as columns
    pivot = state_shap_df.pivot(index="state_name", columns="feature",
                                 values="mean_shap")

    # Keep top 12 features by overall variance across states
    feature_var = pivot.var(axis=0).sort_values(ascending=False)
    top_features = feature_var.head(12).index.tolist()
    pivot = pivot[top_features]

    # Rename columns
    pivot.columns = [FEATURE_LABELS.get(c, c) for c in pivot.columns]

    # Sort states by FMD prevalence (descending)
    state_mean = pivot.mean(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[state_mean.index]

    fig, ax = plt.subplots(figsize=(14, 16))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(pivot.values).max(),
                   vmax=np.abs(pivot.values).max())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03,
                 label="Mean SHAP Value (contribution to FMD risk)")

    ax.set_title("State-Level SHAP Feature Importance\n"
                 "Mean SHAP Values by State — BRFSS 2023 FMD Prediction",
                 fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    out = FIG_DIR / "fig12_state_shap_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def fig13_top_predictor_by_state(state_shap_df: pd.DataFrame) -> None:
    """
    Bar chart: for each state, show which feature has the highest mean |SHAP|.
    Groups states by their top predictor — key policy insight figure.
    """
    # For each state, find feature with highest mean |SHAP|
    idx = state_shap_df.groupby("state_name")["abs_mean_shap"].idxmax()
    top_per_state = state_shap_df.loc[idx][["state_name", "feature",
                                             "abs_mean_shap"]].copy()
    top_per_state["feature_label"] = top_per_state["feature"].map(
        lambda x: FEATURE_LABELS.get(x, x))

    # Sort by feature then by SHAP value within feature
    top_per_state = top_per_state.sort_values(
        ["feature_label", "abs_mean_shap"], ascending=[True, False])

    # Assign colors by top feature
    unique_features = top_per_state["feature_label"].unique()
    cmap = plt.cm.get_cmap("tab10", len(unique_features))
    feature_colors = {f: cmap(i) for i, f in enumerate(unique_features)}
    colors = [feature_colors[f] for f in top_per_state["feature_label"]]

    fig, ax = plt.subplots(figsize=(10, 14))
    bars = ax.barh(top_per_state["state_name"], top_per_state["abs_mean_shap"],
                   color=colors, edgecolor="white", linewidth=0.3, height=0.75)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=feature_colors[f])
               for f in unique_features]
    ax.legend(handles, unique_features, title="Top Predictor",
              loc="lower right", fontsize=8, title_fontsize=9)

    ax.set_xlabel("Mean |SHAP Value| of Top Predictor", fontsize=11)
    ax.set_title("Most Influential FMD Predictor by State\nBRFSS 2023 — LightGBM SHAP Analysis",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig13_top_predictor_by_state.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# SECTION 4 — State-Level SHAP Analysis
# =============================================================================

def compute_state_shap(model, df: pd.DataFrame, feature_names: list,
                        min_n: int = 500) -> pd.DataFrame:
    """
    For each state with ≥ min_n respondents:
      - Compute SHAP values
      - Calculate mean SHAP per feature
    Returns long-format DataFrame: state × feature × mean_shap
    """
    state_col = "_STATE"
    if state_col not in df.columns:
        print("  ⚠ _STATE column not found — skipping state-level SHAP")
        return pd.DataFrame()

    # Reconstruct feature matrix
    X, y = prepare_X(df, feature_names)

    states = df[state_col].dropna().unique()
    records = []

    explainer = shap.TreeExplainer(model)

    valid_states = []
    for s in sorted(states):
        mask = df[state_col] == s
        if mask.sum() >= min_n:
            valid_states.append(s)

    print(f"  Computing state SHAP for {len(valid_states)} states "
          f"(≥{min_n} respondents each)...")

    for i, state_code in enumerate(valid_states):
        mask  = (df[state_col] == state_code).values
        X_st  = X[mask]
        sv    = explainer.shap_values(X_st)
        if isinstance(sv, list):
            sv = sv[1]

        state_name = STATE_FIPS.get(int(state_code), f"State {int(state_code)}")

        for j, feat in enumerate(feature_names):
            records.append({
                "state_code":    int(state_code),
                "state_name":    state_name,
                "feature":       feat,
                "mean_shap":     sv[:, j].mean(),
                "abs_mean_shap": np.abs(sv[:, j]).mean(),
                "n":             mask.sum(),
            })

        if (i + 1) % 10 == 0:
            print(f"    ...{i+1}/{len(valid_states)} states done")

    return pd.DataFrame(records)


# =============================================================================
# SECTION 5 — SHAP Report
# =============================================================================

def generate_shap_report(importance_df: pd.DataFrame,
                          state_shap_df: pd.DataFrame,
                          report_path: Path) -> None:
    lines = []
    lines.append("=" * 65)
    lines.append("BRFSS 2023 — SHAP INTERPRETABILITY REPORT")
    lines.append("=" * 65)

    lines.append("\n── Top 15 Global Predictors (Mean |SHAP|) ──────────────────")
    lines.append(f"  {'Rank':<6} {'Feature':<35} {'Mean |SHAP|':>12}")
    lines.append(f"  {'-'*6} {'-'*35} {'-'*12}")
    for _, row in importance_df.head(15).iterrows():
        lines.append(f"  {int(row['rank']):<6} {row['label']:<35} {row['mean_abs_shap']:>12.5f}")

    if not state_shap_df.empty:
        lines.append("\n── Top Predictor by State (selected) ───────────────────────")
        idx = state_shap_df.groupby("state_name")["abs_mean_shap"].idxmax()
        top = state_shap_df.loc[idx].sort_values("abs_mean_shap", ascending=False)
        for _, row in top.head(20).iterrows():
            label = FEATURE_LABELS.get(row["feature"], row["feature"])
            lines.append(f"  {row['state_name']:<25} {label:<30} {row['abs_mean_shap']:.5f}")

        lines.append("\n── States Where SNAP Is Top Predictor ──────────────────────")
        snap_states = top[top["feature"] == "FOODSTMP"]["state_name"].tolist()
        if snap_states:
            lines.append("  " + ", ".join(snap_states))
        else:
            lines.append("  None in top predictor analysis")

        lines.append("\n── States Where Healthcare Access Is Top Predictor ─────────")
        access_states = top[top["feature"].isin(["MEDCOST1", "HLTHPLN2"])]["state_name"].tolist()
        if access_states:
            lines.append("  " + ", ".join(access_states))
        else:
            lines.append("  None in top predictor analysis")

    lines.append("\n" + "=" * 65)
    text = "\n".join(lines)
    print(text)
    with open(report_path, "w") as f:
        f.write(text)
    print(f"\n✓ SHAP report saved to {report_path.name}")


# =============================================================================
# SECTION 6 — Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  BRFSS 2023 — SHAP Interpretability Analysis")
    print("=" * 60 + "\n")

    # Load artifacts
    print("[Step 1] Loading model and data...")
    model, model_name, all_models, df, feature_names = load_artifacts()
    print(f"  Best model: {model_name}")
    print(f"  Features: {len(feature_names)}")

    # Prepare feature matrix
    print("\n[Step 2] Preparing feature matrix...")
    X, y = prepare_X(df, feature_names)
    print(f"  X shape: {X.shape}")

    # Compute global SHAP
    print("\n[Step 3] Computing global SHAP values...")
    shap_values, X_sample, explainer = compute_global_shap(
        model, X, sample_n=10000)

    # Feature importance
    importance_df = get_feature_importance(shap_values, feature_names)
    importance_df.to_csv(RESULTS_DIR / "global_shap_importance.csv", index=False)
    print(f"\n  Top 10 global predictors:")
    for _, row in importance_df.head(10).iterrows():
        print(f"    {int(row['rank']):>2}. {row['label']:<35} {row['mean_abs_shap']:.5f}")

    # Generate global figures
    print("\n[Step 4] Generating global SHAP figures...")
    fig09_shap_beeswarm(shap_values, X_sample, feature_names)
    fig10_shap_bar(importance_df)
    fig11_shap_dependence(shap_values, X_sample, importance_df)

    # State-level SHAP
    print("\n[Step 5] Computing state-level SHAP (this takes 10–20 min)...")
    state_shap_df = compute_state_shap(model, df, feature_names, min_n=500)

    if not state_shap_df.empty:
        state_shap_df.to_csv(RESULTS_DIR / "state_shap_summary.csv", index=False)
        print(f"  ✓ State SHAP saved ({len(state_shap_df):,} rows)")

        print("\n[Step 6] Generating state-level figures...")
        fig12_state_shap_heatmap(state_shap_df)
        fig13_top_predictor_by_state(state_shap_df)
    else:
        print("  ⚠ Skipping state figures — no state data")

    # SHAP report
    print("\n[Step 7] Generating SHAP report...")
    generate_shap_report(importance_df, state_shap_df,
                         RESULTS_DIR / "shap_report.txt")

    print("\n✅ SHAP analysis complete. Ready for manuscript writing.\n")
    print("Figures generated:")
    for f in sorted(FIG_DIR.glob("fig0[9-9]*.png")) + \
             sorted(FIG_DIR.glob("fig1[0-3]*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
