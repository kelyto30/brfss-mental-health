"""
=============================================================================
BRFSS 2023 — Mental Health Prediction Study
Script 01: Data Acquisition & Preprocessing
=============================================================================
Study: Machine Learning Prediction of Frequent Mental Distress Among U.S.
       Adults and State-Level Variation in Social Determinants
Data:  CDC BRFSS 2023 Annual Survey
       https://www.cdc.gov/brfss/annual_data/annual_2023.html

Outcome: Frequent Mental Distress (FMD) — defined as ≥14 poor mental health
         days in the past 30 days (MENTHLTH variable), binary 0/1

Output:
  - data/brfss_2023_raw.pkl         (raw subset, all selected columns)
  - data/brfss_2023_processed.csv   (analysis-ready, encoded, imputed)
  - data/brfss_2023_processed.pkl   (same, faster loading)
  - data/preprocessing_report.txt   (QC summary)

Usage:
  python brfss_01_preprocess.py

Note: The BRFSS XPT file is ~200MB. First run downloads it automatically.
      Subsequent runs load from local cache.
=============================================================================
"""

import os
import sys
import zipfile
import io
import pickle
import requests
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_XPT   = DATA_DIR / "LLCP2023.XPT"
RAW_ZIP   = DATA_DIR / "LLCP2023XPT.zip"
RAW_PKL   = DATA_DIR / "brfss_2023_raw.pkl"
PROC_CSV  = DATA_DIR / "brfss_2023_processed.csv"
PROC_PKL  = DATA_DIR / "brfss_2023_processed.pkl"
REPORT    = DATA_DIR / "preprocessing_report.txt"

# CDC BRFSS 2023 direct download URL
BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"


# =============================================================================
# SECTION 1 — Variable Definitions
# =============================================================================

# All BRFSS variable names are UPPERCASE as they appear in the XPT file.
# Descriptions follow the 2023 BRFSS Codebook.

OUTCOME_VAR = "MENTHLTH"   # Days of poor mental health in past 30 days (1–30, 88=none, 77/99=missing)

# Selected features — organized by domain
FEATURE_VARS = {

    # ── Sociodemographic ──────────────────────────────────────────────────
    "DEMO": [
        "_AGE_G",     # Age group (6 categories: 18-24 ... 65+)
        "SEX1",       # Biological sex (1=Male, 2=Female)
        "_RACE1",     # Race/ethnicity (8 categories)
        "EDUCA",      # Education level (1–6 scale)
        "MARITAL",    # Marital status (1–6)
        "EMPLOY1",    # Employment status (1–8)
        "_INCOMG1",   # Income category (1–7)
        "_STATE",     # State FIPS code (used for state-level analysis)
    ],

    # ── Social Determinants of Health (SDOH) ─────────────────────────────
    "SDOH": [
        "HLTHPLN1",   # Any health insurance coverage (1=Yes, 2=No)
        "MEDCOST1",   # Could not afford to see doctor in past 12 months (1=Yes, 2=No)
        "FOODSTMP",   # Received food stamps/SNAP in past 12 months (1=Yes, 2=No)
        "RENTHOM1",   # Own/rent home (1=Own, 2=Rent, 3=Other)
    ],

    # ── Health Behaviors ──────────────────────────────────────────────────
    "BEHAVIOR": [
        "EXERANY2",   # Any physical activity in past 30 days (1=Yes, 2=No)
        "SLEPTIM1",   # Average sleep hours per night (integer)
        "SMOKE100",   # Smoked ≥100 cigarettes in lifetime (1=Yes, 2=No)
        "DRNKANY6",   # Any alcohol consumption in past 30 days (1=Yes, 2=No)  
        "_BMI5CAT",   # BMI category (1=Underweight ... 4=Obese)
        "_RFDRHV7",   # Heavy drinker flag (1=No, 2=Yes)
    ],

    # ── Chronic Conditions ────────────────────────────────────────────────
    "CHRONIC": [
        "DIABETE4",   # Ever told had diabetes (1=Yes, 2=No/Pre, 3=Pre, 4=No)
        "BPHIGH6",    # Ever told had high blood pressure (1=Yes, 2=No, 3=Borderline, 4=Preg)
        "ASTHMA3",    # Ever told had asthma (1=Yes, 2=No)
        "CHCCOPD3",   # Ever told had COPD/emphysema (1=Yes, 2=No)
        "CVDCRHD4",   # Ever told had angina/coronary heart disease (1=Yes, 2=No)
        "CVDSTRK3",   # Ever told had stroke (1=Yes, 2=No)
        "ADDEPEV3",   # Ever told had depressive disorder (1=Yes, 2=No) — comorbidity, NOT outcome
    ],

    # ── Physical Health ───────────────────────────────────────────────────
    "PHYSHLTH": [
        "PHYSHLTH",   # Days of poor physical health past 30 days (1–30, 88=none)
        "GENHLTH",    # General health (1=Excellent ... 5=Poor)
    ],

    # ── Survey Weight (for weighted prevalence estimates) ─────────────────
    "WEIGHT": [
        "_LLCPWT",    # Final LLCP weight
    ],
}

# Flatten to single list for reading
ALL_VARS = [OUTCOME_VAR] + [v for group in FEATURE_VARS.values() for v in group]


# =============================================================================
# SECTION 2 — Download
# =============================================================================

def download_brfss(url: str, zip_path: Path, xpt_path: Path) -> None:
    """Download and extract BRFSS 2023 XPT file from CDC."""
    if xpt_path.exists():
        print(f"✓ XPT file already exists: {xpt_path}")
        return

    print(f"⬇  Downloading BRFSS 2023 (~200 MB) from CDC...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 64  # 64KB chunks

    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    print("📦 Extracting ZIP...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find the XPT file inside
        xpt_files = [n for n in z.namelist() if n.upper().endswith(".XPT")]
        if not xpt_files:
            raise FileNotFoundError("No XPT file found inside the ZIP archive.")
        z.extract(xpt_files[0], DATA_DIR)
        # Rename to standard name if needed
        extracted = DATA_DIR / xpt_files[0]
        if extracted != xpt_path:
            extracted.rename(xpt_path)

    print(f"✓ XPT saved to: {xpt_path}")


# =============================================================================
# SECTION 3 — Load Raw Data
# =============================================================================

def load_raw(xpt_path: Path, pkl_path: Path) -> pd.DataFrame:
    """Load XPT → DataFrame, cache as pickle for fast reloads."""
    if pkl_path.exists():
        print("✓ Loading from cached pickle...")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    print("📖 Reading XPT file (this takes 1–2 minutes)...")
    df, meta = pyreadstat.read_xport(str(xpt_path))

    # Keep only variables we need — gracefully skip any missing
    available = [v for v in ALL_VARS if v in df.columns]
    missing_vars = [v for v in ALL_VARS if v not in df.columns]
    if missing_vars:
        print(f"  ⚠ Variables not found in file (check codebook): {missing_vars}")

    df = df[available].copy()
    print(f"  → Loaded {len(df):,} rows × {len(df.columns)} columns")

    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)
    print(f"✓ Cached to {pkl_path}")
    return df


# =============================================================================
# SECTION 4 — Preprocessing & Recoding
# =============================================================================

def recode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    BRFSS uses specific codes for 'Don't know', 'Refused', and 'Not asked'.
    These must be converted to NaN before any analysis.

    Common patterns in BRFSS:
      7, 77, 777, 7777  → Don't know / Not sure
      9, 99, 999, 9999  → Refused
      BLANK             → Not asked / Missing
    """
    df = df.copy()

    # Variables where 7/9 or 77/99 mean missing
    standard_missing = {
        "MENTHLTH":  [77, 99],
        "PHYSHLTH":  [77, 99],
        "SLEPTIM1":  [77, 99],
        "SEX1":      [7, 9],
        "EDUCA":     [9],
        "MARITAL":   [9],
        "EMPLOY1":   [9],
        "HLTHPLN1":  [7, 9],
        "MEDCOST1":  [7, 9],
        "FOODSTMP":  [7, 9],
        "RENTHOM1":  [7, 9],
        "EXERANY2":  [7, 9],
        "SMOKE100":  [7, 9],
        "DRNKANY6":  [7, 9],
        "_RFDRHV7":  [9],
        "DIABETE4":  [7, 9],
        "BPHIGH6":   [7, 9],
        "ASTHMA3":   [7, 9],
        "CHCCOPD3":  [7, 9],
        "CVDCRHD4":  [7, 9],
        "CVDSTRK3":  [7, 9],
        "ADDEPEV3":  [7, 9],
        "GENHLTH":   [7, 9],
        "_RACE1":    [9],
        "_INCOMG1":  [9],
        "_BMI5CAT":  [9],
    }

    for col, codes in standard_missing.items():
        if col in df.columns:
            df[col] = df[col].replace(codes, np.nan)

    return df


def recode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary FMD (Frequent Mental Distress) outcome:
      MENTHLTH: 1–30 = number of days, 88 = none (0 days)
      FMD = 1 if MENTHLTH >= 14, else 0
    """
    df = df.copy()
    # Recode 88 ("None") → 0
    df["MENTHLTH"] = df["MENTHLTH"].replace(88, 0)
    # Create binary outcome
    df["FMD"] = (df["MENTHLTH"] >= 14).astype(float)
    df.loc[df["MENTHLTH"].isna(), "FMD"] = np.nan
    return df


def recode_physhlth(df: pd.DataFrame) -> pd.DataFrame:
    """Recode PHYSHLTH: 88 (none) → 0"""
    if "PHYSHLTH" in df.columns:
        df["PHYSHLTH"] = df["PHYSHLTH"].replace(88, 0)
    return df


def recode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode selected features into analysis-ready forms:
    - Binary features: 1=Yes/present, 0=No/absent
    - Ordinal features: kept as integer scale
    - Multi-category: left as integer codes (will be one-hot encoded later)
    """
    df = df.copy()

    # ── Binary yes/no variables (1=Yes → 1, 2=No → 0) ────────────────────
    binary_1yes_2no = [
        "HLTHPLN1", "MEDCOST1", "FOODSTMP", "EXERANY2",
        "SMOKE100", "DRNKANY6", "ASTHMA3", "CHCCOPD3",
        "CVDCRHD4", "CVDSTRK3", "ADDEPEV3",
    ]
    for col in binary_1yes_2no:
        if col in df.columns:
            df[col] = df[col].map({1: 1, 2: 0})

    # ── BPHIGH6: Yes(1) vs No/Borderline/Pregnancy-only (2,3,4) ──────────
    if "BPHIGH6" in df.columns:
        df["BPHIGH6"] = df["BPHIGH6"].map({1: 1, 2: 0, 3: 0, 4: 0})

    # ── DIABETE4: Diabetes (1=Yes) vs No/Pre (2,3,4) ─────────────────────
    if "DIABETE4" in df.columns:
        df["DIABETE4"] = df["DIABETE4"].map({1: 1, 2: 0, 3: 0, 4: 0})

    # ── SEX1: 1=Male, 2=Female → 0=Male, 1=Female ────────────────────────
    if "SEX1" in df.columns:
        df["SEX1"] = df["SEX1"].map({1: 0, 2: 1})
        df.rename(columns={"SEX1": "FEMALE"}, inplace=True)

    # ── _RFDRHV7: Heavy drinker: 1=No, 2=Yes → 0=No, 1=Yes ──────────────
    if "_RFDRHV7" in df.columns:
        df["_RFDRHV7"] = df["_RFDRHV7"].map({1: 0, 2: 1})
        df.rename(columns={"_RFDRHV7": "HEAVY_DRINKER"}, inplace=True)

    # ── RENTHOM1: 1=Own, 2=Rent, 3=Other (leave as ordinal) ──────────────
    # (Will be one-hot encoded in modeling script)

    # ── SLEPTIM1: Keep as continuous (valid range 1–24) ───────────────────
    if "SLEPTIM1" in df.columns:
        df["SLEPTIM1"] = df["SLEPTIM1"].clip(1, 24)

    return df


def recode_sleep_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical sleep variable alongside continuous (for EDA)."""
    if "SLEPTIM1" in df.columns:
        df["SLEEP_CAT"] = pd.cut(
            df["SLEPTIM1"],
            bins=[0, 5, 6, 8, 9, 24],
            labels=["Very short (<6h)", "Short (6h)", "Normal (7-8h)", "Long (9h)", "Very long (>9h)"],
            right=True
        )
    return df


def drop_missing_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing FMD outcome."""
    before = len(df)
    df = df.dropna(subset=["FMD"])
    after = len(df)
    print(f"  → Dropped {before - after:,} rows with missing outcome ({(before-after)/before*100:.1f}%)")
    return df


def impute_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Simple imputation strategy:
    - Continuous variables (SLEPTIM1, PHYSHLTH): median
    - Categorical/binary variables: mode
    
    Note: For the modeling script, we will use IterativeImputer (MICE) 
    as a sensitivity analysis. This function handles the simpler approach
    for the main analysis.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        if col in ["SLEPTIM1", "PHYSHLTH", "_LLCPWT"]:
            fill_val = df[col].median()
        else:
            mode_vals = df[col].mode()
            if len(mode_vals) == 0:
                continue
            fill_val = mode_vals.iloc[0]
        df[col] = df[col].fillna(fill_val)

    return df


# =============================================================================
# SECTION 5 — State Labels
# =============================================================================

STATE_FIPS = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
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
    56: "Wyoming", 66: "Guam", 72: "Puerto Rico", 78: "Virgin Islands",
}


# =============================================================================
# SECTION 6 — QC Report
# =============================================================================

def generate_report(df_raw: pd.DataFrame, df_proc: pd.DataFrame,
                    feature_cols: list, report_path: Path) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("BRFSS 2023 — PREPROCESSING QUALITY CONTROL REPORT")
    lines.append("=" * 70)
    lines.append(f"\nRaw dataset shape:       {df_raw.shape[0]:>10,} rows × {df_raw.shape[1]} columns")
    lines.append(f"Processed dataset shape: {df_proc.shape[0]:>10,} rows × {df_proc.shape[1]} columns")

    lines.append("\n── Outcome Variable (FMD) ──────────────────────────────────────────")
    fmd_counts = df_proc["FMD"].value_counts()
    fmd_pct    = df_proc["FMD"].value_counts(normalize=True) * 100
    lines.append(f"  FMD = 0 (< 14 days):  {fmd_counts.get(0.0, 0):>10,}  ({fmd_pct.get(0.0, 0):.1f}%)")
    lines.append(f"  FMD = 1 (≥ 14 days):  {fmd_counts.get(1.0, 0):>10,}  ({fmd_pct.get(1.0, 0):.1f}%)")

    lines.append("\n── Missing Data Summary (Processed) ───────────────────────────────")
    lines.append(f"  {'Variable':<20} {'N Missing':>12} {'% Missing':>10}")
    lines.append(f"  {'-'*20} {'-'*12} {'-'*10}")
    for col in [c for c in feature_cols if c in df_proc.columns]:
        nm = df_proc[col].isna().sum()
        pct = nm / len(df_proc) * 100
        if nm > 0:
            lines.append(f"  {col:<20} {nm:>12,} {pct:>10.1f}%")
    lines.append("  (Variables with 0 missing not shown)")

    lines.append("\n── FMD Prevalence by State (Top 10 Highest) ────────────────────────")
    state_prev = df_proc.groupby("_STATE")["FMD"].mean().sort_values(ascending=False)
    state_prev.index = state_prev.index.map(lambda x: STATE_FIPS.get(int(x), str(x)))
    for state, prev in state_prev.head(10).items():
        lines.append(f"  {state:<30} {prev*100:.1f}%")

    lines.append("\n── FMD Prevalence by State (Top 10 Lowest) ─────────────────────────")
    for state, prev in state_prev.tail(10).items():
        lines.append(f"  {state:<30} {prev*100:.1f}%")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n✓ Report saved to {report_path}")


# =============================================================================
# SECTION 7 — Main Pipeline
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  BRFSS 2023 — Data Acquisition & Preprocessing")
    print("=" * 60 + "\n")

    # Step 1: Download
    download_brfss(BRFSS_URL, RAW_ZIP, RAW_XPT)

    # Step 2: Load
    df = load_raw(RAW_XPT, RAW_PKL)
    df_raw_snapshot = df.copy()

    # Step 3: Recode missing values
    print("\n[Step 3] Recoding missing values...")
    df = recode_missing(df)

    # Step 4: Recode outcome
    print("[Step 4] Creating FMD outcome variable...")
    df = recode_outcome(df)

    # Step 5: Recode PHYSHLTH
    df = recode_physhlth(df)

    # Step 6: Recode features
    print("[Step 5] Recoding features...")
    df = recode_features(df)
    df = recode_sleep_category(df)

    # Step 7: Add state labels
    df["STATE_NAME"] = df["_STATE"].map(lambda x: STATE_FIPS.get(int(x), "Unknown") if pd.notna(x) else np.nan)

    # Step 8: Drop rows with missing outcome
    print("[Step 6] Dropping missing outcomes...")
    df = drop_missing_outcome(df)

    # Step 9: Define final feature columns
    feature_cols = [
        c for c in df.columns
        if c not in ["MENTHLTH", "FMD", "STATE_NAME", "_LLCPWT", "SLEEP_CAT"]
    ]

    # Step 10: Impute missing features
    print("[Step 7] Imputing missing features (median/mode)...")
    df = impute_features(df, feature_cols)

    # Step 11: Summary
    print(f"\n✓ Final processed dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  FMD prevalence: {df['FMD'].mean()*100:.1f}%")

    # Step 12: Save
    df.to_csv(PROC_CSV, index=False)
    with open(PROC_PKL, "wb") as f:
        pickle.dump(df, f)
    print(f"\n✓ Saved: {PROC_CSV}")
    print(f"✓ Saved: {PROC_PKL}")

    # Step 13: QC Report
    print("\n[Step 8] Generating QC report...")
    generate_report(df_raw_snapshot, df, feature_cols, REPORT)

    print("\n✅ Preprocessing complete. Ready for EDA (Script 02).\n")


if __name__ == "__main__":
    main()
