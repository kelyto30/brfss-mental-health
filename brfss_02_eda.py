"""
=============================================================================
BRFSS 2023 — Mental Health Prediction Study
Script 02: Exploratory Data Analysis (EDA)
=============================================================================
Run AFTER brfss_01_preprocess.py

Outputs (all saved to figures/):
  - fig01_fmd_prevalence_map.png     (US choropleth by state)
  - fig02_fmd_by_demographics.png    (bar charts by key demo groups)
  - fig03_fmd_by_sdoh.png            (FMD rate by SDOH variables)
  - fig04_correlation_heatmap.png    (feature correlation matrix)
  - fig05_outcome_distribution.png   (MENTHLTH raw distribution)
  - eda_summary.txt                  (descriptive statistics table)
=============================================================================
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
FIG_DIR   = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Color palette (consistent across all figures) ─────────────────────────
C_POS   = "#E63946"   # FMD positive
C_NEG   = "#457B9D"   # FMD negative / background
C_MAP   = "YlOrRd"
FONT    = "DejaVu Sans"

STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT",
    10: "DE", 11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID",
    17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA",
    23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS",
    29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ",
    35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK",
    41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV",
    55: "WI", 56: "WY",
}


def load_data() -> pd.DataFrame:
    pkl = DATA_DIR / "brfss_2023_processed.pkl"
    if not pkl.exists():
        raise FileNotFoundError("Run brfss_01_preprocess.py first.")
    with open(pkl, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Figure 1 — FMD Prevalence Bar Chart by State (horizontal, sorted)
# =============================================================================

def fig01_state_prevalence(df: pd.DataFrame) -> None:
    state_prev = (
        df.groupby("_STATE")["FMD"]
        .agg(["mean", "count"])
        .reset_index()
    )
    state_prev["abbr"] = state_prev["_STATE"].map(lambda x: STATE_FIPS.get(int(x), "??"))
    state_prev = state_prev[state_prev["abbr"] != "??"]   # drop territories
    state_prev = state_prev.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 14))
    colors = [C_POS if v >= state_prev["mean"].median() else C_NEG
              for v in state_prev["mean"]]
    bars = ax.barh(state_prev["abbr"], state_prev["mean"] * 100,
                   color=colors, edgecolor="white", linewidth=0.4, height=0.75)
    national_avg = df["FMD"].mean() * 100
    ax.axvline(national_avg, color="black", linestyle="--", linewidth=1.2,
               label=f"National avg: {national_avg:.1f}%")
    ax.set_xlabel("FMD Prevalence (%)", fontsize=11)
    ax.set_title("Frequent Mental Distress (≥14 days) Prevalence by State\nBRFSS 2023",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, state_prev["mean"].max() * 100 * 1.15)
    for bar, val in zip(bars, state_prev["mean"]):
        ax.text(val * 100 + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "fig01_fmd_prevalence_by_state.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Figure 2 — FMD by Sociodemographics
# =============================================================================

def fig02_demographics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("FMD Prevalence by Sociodemographic Characteristics\nBRFSS 2023",
                 fontsize=13, fontweight="bold", y=1.01)

    panels = [
        ("_AGE_G",   "Age Group",
         {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+"}),
        ("FEMALE",   "Sex",
         {0: "Male", 1: "Female"}),
        ("_RACE1",   "Race/Ethnicity",
         {1: "White NH", 2: "Black NH", 3: "AIAN", 4: "Asian NH",
          5: "NHPI", 6: "Other NH", 7: "Multiracial", 8: "Hispanic"}),
        ("EDUCA",    "Education",
         {1: "Never", 2: "Elem", 3: "Some HS", 4: "HS Grad", 5: "Some College", 6: "College+"}),
        ("_INCOMG1", "Income Category",
         {1: "<$15k", 2: "$15-25k", 3: "$25-35k", 4: "$35-50k",
          5: "$50-100k", 6: "$100-200k", 7: ">$200k"}),
        ("EMPLOY1",  "Employment Status",
         {1: "Employed", 2: "Self-Empl", 3: "Unemployed", 4: "Unable to Work",
          5: "Homemaker", 6: "Student", 7: "Retired", 8: "Other"}),
    ]

    for ax, (col, title, labels) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        tmp = df.groupby(col)["FMD"].mean() * 100
        tmp.index = [labels.get(k, str(k)) for k in tmp.index]
        colors = [C_POS if v >= tmp.mean() else C_NEG for v in tmp.values]
        tmp.plot(kind="bar", ax=ax, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
        ax.axhline(df["FMD"].mean() * 100, color="black", linestyle="--",
                   linewidth=1, alpha=0.7)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel("FMD Prevalence (%)", fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig02_fmd_by_demographics.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Figure 3 — FMD by SDOH & Health Behaviors
# =============================================================================

def fig03_sdoh_behaviors(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("FMD Prevalence by Social Determinants & Health Behaviors\nBRFSS 2023",
                 fontsize=13, fontweight="bold")

    panels = [
        ("HLTHPLN1",     "Has Health Insurance",     {0: "Uninsured", 1: "Insured"}),
        ("MEDCOST1",     "Couldn't Afford Doctor",   {0: "Could Afford", 1: "Couldn't Afford"}),
        ("FOODSTMP",     "Received SNAP/Food Stamps", {0: "No", 1: "Yes"}),
        ("RENTHOM1",     "Housing",                  {1: "Own", 2: "Rent", 3: "Other"}),
        ("EXERANY2",     "Physical Activity",        {0: "No Exercise", 1: "Exercised"}),
        ("SMOKE100",     "Ever Smoked 100 Cigs",     {0: "No", 1: "Yes"}),
        ("HEAVY_DRINKER","Heavy Drinker",             {0: "No", 1: "Yes"}),
        ("_BMI5CAT",     "BMI Category",
         {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}),
    ]

    for ax, (col, title, labels) in zip(axes.flat, panels):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        tmp = df.groupby(col)["FMD"].mean() * 100
        tmp.index = [labels.get(k, str(k)) for k in tmp.index]
        colors = [C_POS if v >= df["FMD"].mean() * 100 else C_NEG for v in tmp.values]
        tmp.plot(kind="bar", ax=ax, color=colors, edgecolor="white", linewidth=0.5, width=0.65)
        ax.axhline(df["FMD"].mean() * 100, color="black", linestyle="--",
                   linewidth=1, alpha=0.7)
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_ylabel("FMD (%)", fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "fig03_fmd_by_sdoh.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Figure 4 — Correlation Heatmap
# =============================================================================

def fig04_correlation(df: pd.DataFrame) -> None:
    num_cols = [
        "FMD", "PHYSHLTH", "GENHLTH", "SLEPTIM1", "_AGE_G",
        "FEMALE", "EDUCA", "_INCOMG1",
        "HLTHPLN1", "MEDCOST1", "FOODSTMP", "EXERANY2",
        "SMOKE100", "HEAVY_DRINKER", "_BMI5CAT",
        "DIABETE4", "BPHIGH6", "ASTHMA3", "CHCCOPD3",
        "CVDCRHD4", "CVDSTRK3", "ADDEPEV3",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    labels = {
        "FMD": "FMD", "PHYSHLTH": "Phys Hlth Days", "GENHLTH": "Gen Health",
        "SLEPTIM1": "Sleep Hours", "_AGE_G": "Age Group", "FEMALE": "Female",
        "EDUCA": "Education", "_INCOMG1": "Income",
        "HLTHPLN1": "Has Insurance", "MEDCOST1": "Can't Afford MD",
        "FOODSTMP": "SNAP", "EXERANY2": "Exercise",
        "SMOKE100": "Ever Smoked", "HEAVY_DRINKER": "Heavy Drinker",
        "_BMI5CAT": "BMI Cat", "DIABETE4": "Diabetes",
        "BPHIGH6": "Hypertension", "ASTHMA3": "Asthma",
        "CHCCOPD3": "COPD", "CVDCRHD4": "CHD", "CVDSTRK3": "Stroke",
        "ADDEPEV3": "Depress Dx",
    }
    display_labels = [labels.get(c, c) for c in num_cols]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(display_labels, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.values[i, j]
            if abs(val) > 0.15:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if abs(val) < 0.6 else "white")
    ax.set_title("Feature Correlation Matrix — BRFSS 2023 Mental Health Study",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = FIG_DIR / "fig04_correlation_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# Figure 5 — MENTHLTH Raw Distribution
# =============================================================================

def fig05_outcome_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of raw MENTHLTH days
    ax = axes[0]
    vals = df["MENTHLTH"].dropna()
    ax.hist(vals, bins=31, range=(0, 30), color=C_NEG, edgecolor="white",
            linewidth=0.5, density=False)
    ax.axvline(14, color=C_POS, linestyle="--", linewidth=2,
               label="≥14 days threshold (FMD)")
    ax.set_xlabel("Poor Mental Health Days (Past 30)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of MENTHLTH\n(0 = No poor days)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: FMD vs non-FMD pie chart
    ax2 = axes[1]
    counts = df["FMD"].value_counts()
    labels = [f"No FMD\n({counts.get(0.0, 0):,})", f"FMD\n({counts.get(1.0, 0):,})"]
    colors = [C_NEG, C_POS]
    wedges, texts, autotexts = ax2.pie(
        [counts.get(0.0, 0), counts.get(1.0, 0)],
        labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    ax2.set_title("FMD Outcome Distribution\n(≥14 poor mental health days)",
                  fontsize=11, fontweight="bold")

    plt.suptitle("BRFSS 2023 — Outcome Variable Summary", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig05_outcome_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


# =============================================================================
# EDA Summary Text
# =============================================================================

def eda_summary(df: pd.DataFrame) -> None:
    lines = []
    lines.append("=" * 65)
    lines.append("BRFSS 2023 — EDA SUMMARY")
    lines.append("=" * 65)
    lines.append(f"\nTotal analyzed respondents: {len(df):,}")
    lines.append(f"FMD prevalence (national):  {df['FMD'].mean()*100:.2f}%")
    lines.append(f"FMD cases:                  {int(df['FMD'].sum()):,}")

    lines.append("\n── Top 5 Highest FMD Prevalence States ──────────────────")
    sp = df.groupby("STATE_NAME")["FMD"].mean().sort_values(ascending=False)
    for s, v in sp.head(5).items():
        lines.append(f"  {s:<25} {v*100:.1f}%")

    lines.append("\n── Top 5 Lowest FMD Prevalence States ───────────────────")
    for s, v in sp.tail(5).items():
        lines.append(f"  {s:<25} {v*100:.1f}%")

    lines.append("\n── FMD by Key Risk Factors ───────────────────────────────")
    comparisons = [
        ("MEDCOST1",     "Cannot afford doctor (Yes vs No)"),
        ("FOODSTMP",     "Receives SNAP (Yes vs No)"),
        ("HLTHPLN1",     "Has insurance (Yes vs No)"),
        ("ADDEPEV3",     "Depression diagnosis (Yes vs No)"),
        ("EXERANY2",     "Exercises (Yes vs No)"),
    ]
    for col, label in comparisons:
        if col not in df.columns:
            continue
        v = df.groupby(col)["FMD"].mean() * 100
        if 1 in v.index and 0 in v.index:
            lines.append(f"  {label}")
            lines.append(f"    Yes: {v.get(1, 0):.1f}%   No: {v.get(0, 0):.1f}%  "
                         f"(diff: {v.get(1,0)-v.get(0,0):+.1f}pp)")

    out = BASE_DIR / "data" / "eda_summary.txt"
    text = "\n".join(lines)
    print(text)
    with open(out, "w") as f:
        f.write(text)
    print(f"\n✓ EDA summary saved to {out.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 55)
    print("  BRFSS 2023 — Exploratory Data Analysis")
    print("=" * 55 + "\n")

    df = load_data()
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns\n")

    print("Generating figures...")
    fig01_state_prevalence(df)
    fig02_demographics(df)
    fig03_sdoh_behaviors(df)
    fig04_correlation(df)
    fig05_outcome_distribution(df)

    print("\nGenerating EDA summary...")
    eda_summary(df)

    print(f"\n✅ EDA complete. All figures saved to: {FIG_DIR}\n")


if __name__ == "__main__":
    main()
