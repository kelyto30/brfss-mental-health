"""
=============================================================================
BRFSS 2023 — Figure Cleanup Script
Fixes:
  1. Raw encoded variable names → human-readable labels (fig09, fig10)
  2. Territory FIPS codes → proper names (fig12)
  3. Redesign fig13 → secondary predictor by state (more informative)
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

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
FIG_DIR     = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# ── Complete human-readable labels including one-hot dummies ───────────────
FEATURE_LABELS = {
    # Original variables
    "_AGE_G":           "Age Group",
    "FEMALE":           "Female Sex",
    "EDUCA":            "Education Level",
    "_INCOMG1":         "Income Category",
    "_STATE":           "State",
    "MEDCOST1":         "Cannot Afford Doctor",
    "FOODSTMP":         "Receives SNAP",
    "EXERANY2":         "Physical Activity",
    "SMOKE100":         "Ever Smoked",
    "DRNKANY6":         "Alcohol Use",
    "HEAVY_DRINKER":    "Heavy Drinker",
    "_BMI5CAT":         "BMI Category",
    "SLEPTIM1":         "Sleep Hours",
    "PHYSHLTH":         "Poor Physical Health Days",
    "GENHLTH":          "General Health Rating",
    "DIABETE4":         "Diabetes",
    "BPHIGH6":          "Hypertension",
    "ASTHMA3":          "Asthma",
    "CHCCOPD3":         "COPD",
    "CVDCRHD4":         "Coronary Heart Disease",
    "CVDSTRK3":         "Stroke History",
    "ADDEPEV3":         "Depression Diagnosis",
    "HLTHPLN2":         "Has Health Insurance",

    # One-hot encoded: EMPLOY1 (employment status)
    # 1=Employed, 2=Self-employed, 3=Unemployed, 4=Unable to work,
    # 5=Homemaker, 6=Student, 7=Retired, 8=Other
    "EMPLOY1_2.0":      "Self-Employed",
    "EMPLOY1_3.0":      "Unemployed",
    "EMPLOY1_4.0":      "Unable to Work",
    "EMPLOY1_5.0":      "Homemaker",
    "EMPLOY1_6.0":      "Student",
    "EMPLOY1_7.0":      "Retired",
    "EMPLOY1_8.0":      "Other Employment",

    # One-hot encoded: MARITAL (marital status)
    # 1=Married, 2=Divorced, 3=Widowed, 4=Separated,
    # 5=Never married, 6=Unmarried couple
    "MARITAL_2.0":      "Divorced",
    "MARITAL_3.0":      "Widowed",
    "MARITAL_4.0":      "Separated",
    "MARITAL_5.0":      "Never Married",
    "MARITAL_6.0":      "Unmarried Couple",

    # One-hot encoded: RENTHOM1 (housing)
    # 1=Own, 2=Rent, 3=Other
    "RENTHOM1_2.0":     "Renting Home",
    "RENTHOM1_3.0":     "Other Housing",
}

# ── State/territory FIPS lookup (including territories) ───────────────────
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
    56: "Wyoming", 66: "Guam", 72: "Puerto Rico", 78: "U.S. Virgin Islands",
}

# Colors
C_POS = "#E63946"
C_NEG = "#457B9D"


# =============================================================================
# Load artifacts
# =============================================================================

def load_all():
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        best = pickle.load(f)
    with open(DATA_DIR / "brfss_2023_processed.pkl", "rb") as f:
        df = pickle.load(f)
    shap_importance = pd.read_csv(RESULTS_DIR / "global_shap_importance.csv")
    state_shap      = pd.read_csv(RESULTS_DIR / "state_shap_summary.csv")

    # Apply full label mapping to importance df
    shap_importance["label"] = shap_importance["feature"].map(
        lambda x: FEATURE_LABELS.get(x, x))

    # Fix state names in state_shap
    state_shap["state_name"] = state_shap["state_code"].map(
        lambda x: STATE_FIPS.get(int(x), f"State {int(x)}"))

    # Apply labels to state_shap features
    state_shap["feature_label"] = state_shap["feature"].map(
        lambda x: FEATURE_LABELS.get(x, x))

    feature_names = best["features"]
    model         = best["model"]

    return model, df, shap_importance, state_shap, feature_names


# =============================================================================
# Recompute SHAP values (needed for beeswarm)
# =============================================================================

def recompute_shap(model, df, feature_names, sample_n=10000):
    import shap
    from brfss_03_modeling import prepare_features
    X, y, _ = prepare_features(df)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    if len(X) > sample_n:
        idx = np.random.RandomState(42).choice(len(X), sample_n, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_sample


# =============================================================================
# FIG 09 — Fixed Beeswarm
# =============================================================================

def fig09_fixed(shap_values, X_sample, feature_names, importance_df, top_n=20):
    # Get top N by mean |SHAP|
    top_features = importance_df.head(top_n)["feature"].tolist()
    top_labels   = importance_df.head(top_n)["label"].tolist()

    # Get indices in feature_names
    feat_idx = [list(X_sample.columns).index(f)
                for f in top_features if f in X_sample.columns]
    labels   = [FEATURE_LABELS.get(feature_names[i], feature_names[i])
                for i in feat_idx]

    sv_top = shap_values[:, feat_idx]
    X_top  = X_sample.iloc[:, feat_idx]

    fig, ax = plt.subplots(figsize=(12, 9))
    for j in range(len(feat_idx)):
        col_sv  = sv_top[:, j]
        col_x   = X_top.iloc[:, j].values.astype(float)
        x_norm  = (col_x - np.nanmin(col_x)) / (np.nanmax(col_x) - np.nanmin(col_x) + 1e-9)
        colors  = plt.cm.RdBu_r(1 - x_norm)
        y_jit   = j + np.random.RandomState(j).uniform(-0.3, 0.3, len(col_sv))
        ax.scatter(col_sv, y_jit, c=colors, s=4, alpha=0.4, linewidths=0)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value (impact on FMD prediction)", fontsize=11)
    ax.set_title(f"Global SHAP Feature Importance — Top {top_n} Features\n"
                 f"LightGBM Model, BRFSS 2023 (n=10,000 sample)",
                 fontsize=12, fontweight="bold")
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


# =============================================================================
# FIG 10 — Fixed Bar Chart
# =============================================================================

def fig10_fixed(importance_df, top_n=20):
    top = importance_df.head(top_n).iloc[::-1]
    colors = [C_POS if i >= top_n // 2 else C_NEG for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top["label"], top["mean_abs_shap"],
                   color=colors, edgecolor="white", linewidth=0.4, height=0.7)
    ax.set_xlabel("Mean |SHAP Value| (average impact on model output)", fontsize=10)
    ax.set_title("Feature Importance — Mean |SHAP| Values\n"
                 "LightGBM Model, BRFSS 2023 FMD Prediction",
                 fontsize=12, fontweight="bold")
    for bar, val in zip(bars, top["mean_abs_shap"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig10_shap_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# FIG 12 — Fixed Heatmap (territory labels corrected)
# =============================================================================

def fig12_fixed(state_shap_df):
    # Exclude territories for cleaner 50-state + DC figure
    us_states = [s for s in STATE_FIPS.values()
                 if s not in ["Guam", "Puerto Rico", "U.S. Virgin Islands"]]
    df_states = state_shap_df[state_shap_df["state_name"].isin(us_states)]

    pivot = df_states.pivot_table(
        index="state_name", columns="feature_label",
        values="mean_shap", aggfunc="mean")

    # Top 12 features by cross-state variance
    top_feats = pivot.var(axis=0).sort_values(ascending=False).head(12).index.tolist()
    pivot = pivot[top_feats]

    # Sort states by average SHAP (descending = highest risk states on top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    vmax = np.abs(pivot.values).max()
    fig, ax = plt.subplots(figsize=(15, 16))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03,
                 label="Mean SHAP Value (contribution to FMD risk)")
    ax.set_title("State-Level SHAP Feature Importance — 50 States + D.C.\n"
                 "Mean SHAP Values by State, BRFSS 2023 FMD Prediction",
                 fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    out = FIG_DIR / "fig12_state_shap_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# FIG 13 — REDESIGNED: Secondary Predictor by State
# =============================================================================

def fig13_redesigned(state_shap_df):
    """
    For each state, find the #2 most important predictor (by mean |SHAP|),
    excluding Depression Diagnosis which dominates everywhere.
    This reveals genuine geographic heterogeneity in secondary drivers.
    """
    # Exclude territories
    us_states = [s for s in STATE_FIPS.values()
                 if s not in ["Guam", "Puerto Rico", "U.S. Virgin Islands"]]
    df_states = state_shap_df[state_shap_df["state_name"].isin(us_states)].copy()

    # For each state, rank features by abs_mean_shap, pick #2 (excluding Depression Dx)
    exclude_top = {"Depression Diagnosis", "Age Group"}  # universal dominators

    records = []
    for state, grp in df_states.groupby("state_name"):
        grp_sorted = grp[~grp["feature_label"].isin(exclude_top)]\
                     .sort_values("abs_mean_shap", ascending=False)
        if len(grp_sorted) == 0:
            continue
        top_row = grp_sorted.iloc[0]
        records.append({
            "state":         state,
            "feature":       top_row["feature"],
            "feature_label": top_row["feature_label"],
            "abs_mean_shap": top_row["abs_mean_shap"],
            "mean_shap":     top_row["mean_shap"],
        })

    result_df = pd.DataFrame(records).sort_values("abs_mean_shap", ascending=True)

    # Assign colors by secondary predictor category
    unique_feats = result_df["feature_label"].unique()
    # Group into meaningful categories for color
    color_map = {
        "Poor Physical Health Days": "#E63946",
        "General Health Rating":     "#E76F51",
        "Cannot Afford Doctor":      "#2A9D8F",
        "Receives SNAP":             "#264653",
        "Retired":                   "#457B9D",
        "Ever Smoked":               "#E9C46A",
        "Physical Activity":         "#F4A261",
        "Alcohol Use":               "#6A4C93",
        "BMI Category":              "#1982C4",
        "Income Category":           "#8AC926",
        "Education Level":           "#FF595E",
        "Widowed":                   "#6A994E",
        "Divorced":                  "#BC4749",
        "Renting Home":              "#A7C957",
        "State":                     "#CCCCCC",
    }
    default_colors = plt.cm.get_cmap("tab20", len(unique_feats))
    for i, f in enumerate(unique_feats):
        if f not in color_map:
            color_map[f] = matplotlib.colors.to_hex(default_colors(i))

    colors = [color_map.get(f, "#AAAAAA") for f in result_df["feature_label"]]

    fig, ax = plt.subplots(figsize=(11, 16))
    bars = ax.barh(result_df["state"], result_df["abs_mean_shap"],
                   color=colors, edgecolor="white", linewidth=0.3, height=0.75)

    # Add feature label annotations on bars
    for bar, row in zip(bars, result_df.itertuples()):
        ax.text(row.abs_mean_shap + 0.001,
                bar.get_y() + bar.get_height() / 2,
                row.feature_label, va="center", fontsize=7.5,
                color="#333333")

    # Legend
    shown = result_df["feature_label"].value_counts()
    legend_feats = shown.index.tolist()
    handles = [plt.Rectangle((0, 0), 1, 1,
                color=color_map.get(f, "#AAAAAA"), linewidth=0)
               for f in legend_feats]
    ax.legend(handles, legend_feats,
              title="Secondary Predictor", loc="lower right",
              fontsize=8, title_fontsize=9, framealpha=0.9)

    ax.set_xlabel("Mean |SHAP Value| of Secondary Predictor", fontsize=11)
    ax.set_title("Secondary Driver of Frequent Mental Distress by State\n"
                 "(Excluding Depression Diagnosis & Age Group)\n"
                 "BRFSS 2023 — LightGBM SHAP Analysis",
                 fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig13_secondary_predictor_by_state.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")

    # Print summary
    print("\n  Secondary predictor distribution across states:")
    for feat, count in shown.items():
        states_list = result_df[result_df["feature_label"] == feat]["state"].tolist()
        print(f"    {feat:<35} {count:>2} states: {', '.join(states_list)}")

    return result_df


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  BRFSS 2023 — Figure Cleanup & Regeneration")
    print("=" * 60 + "\n")

    print("[Step 1] Loading artifacts...")
    model, df, importance_df, state_shap_df, feature_names = load_all()
    print(f"  Loaded {len(importance_df)} features, {len(state_shap_df):,} state-SHAP rows")

    print("\n[Step 2] Recomputing SHAP values for beeswarm (n=10,000)...")
    shap_values, X_sample = recompute_shap(model, df, feature_names)

    print("\n[Step 3] Regenerating fig09 — fixed beeswarm...")
    fig09_fixed(shap_values, X_sample, feature_names, importance_df)

    print("\n[Step 4] Regenerating fig10 — fixed bar chart...")
    fig10_fixed(importance_df)

    print("\n[Step 5] Regenerating fig12 — fixed heatmap (territories labeled)...")
    fig12_fixed(state_shap_df)

    print("\n[Step 6] Redesigning fig13 — secondary predictor by state...")
    result_df = fig13_redesigned(state_shap_df)
    result_df.to_csv(RESULTS_DIR / "secondary_predictor_by_state.csv", index=False)

    print("\n✅ All figures regenerated and saved to figures/")
    print("\nFigures updated:")
    print("  • fig09_shap_beeswarm.png     (encoded labels → readable)")
    print("  • fig10_shap_bar.png          (encoded labels → readable)")
    print("  • fig12_state_shap_heatmap.png (territories properly labeled, US-only)")
    print("  • fig13_secondary_predictor_by_state.png (redesigned — secondary drivers)")


if __name__ == "__main__":
    main()
