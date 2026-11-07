"""Clinical and financial feature engineering: ICD-10 hierarchy, HCC score, lab trajectories, medication features, catalog."""
import re
import numpy as np
import pandas as pd
from data.processing.synth import COND_KEYS, CONDITIONS

ICD_CHAPTERS = [("A00", "B99", "Infectious"), ("C00", "D49", "Neoplasms"), ("E00", "E89", "Endocrine"), ("F01", "F99", "Mental"),
                ("G00", "G99", "Nervous"), ("I00", "I99", "Circulatory"), ("J00", "J99", "Respiratory"), ("K00", "K95", "Digestive"),
                ("M00", "M99", "Musculoskeletal"), ("N00", "N99", "Genitourinary")]


def icd10_hierarchy(code):
    """'E11.65' -> chapter 'Endocrine', block 'E10-E14', category 'E11', full code. Invalid -> None."""
    if not re.fullmatch(r"[A-Z]\d{2}(\.\d{1,4})?", code or ""):
        return None
    cat = code[:3]
    chapter = next((n for a, b, n in ICD_CHAPTERS if a <= cat <= b), "Other")
    block = f"{cat[0]}{(int(cat[1:]) // 5) * 5:02d}-{cat[0]}{(int(cat[1:]) // 5) * 5 + 4:02d}"
    return {"chapter": chapter, "block": block, "category": cat, "code": code}


def hcc_risk_score(flags, age, sex):
    """Additive HCC-style score: demographic base + weights of present conditions (weights follow the brief's worked example)."""
    w = np.array([CONDITIONS[c][1] for c in COND_KEYS])
    return 0.25 + 0.01 * (age - 60) + 0.03 * sex + np.asarray(flags) @ w


def slope(values):
    """Least-squares slope per row of equally spaced measurements (NaN-safe)."""
    v = np.asarray(values, dtype=float)
    x = np.arange(v.shape[1], dtype=float)
    out = np.full(len(v), np.nan)
    for i, row in enumerate(v):
        m = ~np.isnan(row)
        if m.sum() >= 2:
            out[i] = np.polyfit(x[m], row[m], 1)[0]
    return out


def lab_variability(values):
    return np.nanstd(np.asarray(values, dtype=float), axis=1)


DEMO = ["age", "sex", "plan", "region", "tobacco"]
DX = ["hcc_score", "hcc_count", "charlson", "n_conditions"] + ["cond_" + c for c in COND_KEYS]
LABS = ["hba1c_last", "egfr_last", "hba1c_slope", "egfr_slope", "ldl", "sbp", "n_labs", "hba1c_var", "egfr_var"]
MEDS = ["med_count", "ddi_major", "ddi_moderate", "class_diversity", "polypharmacy_score", "adherence_mpr"]
UTIL = ["prior_hosp", "prior_ed", "prior_specialist", "prior_pcp", "log_prior_cost", "prior_cost_trend"]
FUSION = ["clinical_trajectory_score", "interaction_burden"]
STRUCT = DEMO + DX + LABS + MEDS + UTIL + FUSION


def add_features(P):
    X = P.copy()
    hb = X[[f"hba1c_{m}" for m in range(4)]].values
    eg = X[[f"egfr_{m}" for m in range(4)]].values
    X["n_labs"] = np.isfinite(hb).sum(axis=1) + np.isfinite(eg).sum(axis=1)
    X["hba1c_var"] = lab_variability(np.where(np.isfinite(hb).sum(axis=1)[:, None] >= 2, hb, np.nan))
    X["egfr_var"] = lab_variability(np.where(np.isfinite(eg).sum(axis=1)[:, None] >= 2, eg, np.nan))
    X["log_prior_cost"] = np.log1p(X["prior_cost"])
    z = lambda c: (X[c] - X[c].mean()) / (X[c].std() + 1e-9)
    X["clinical_trajectory_score"] = (0.5 * np.nan_to_num(z("hba1c_slope")) - 0.5 * np.nan_to_num(z("egfr_slope")) + 0.3 * z("prior_cost_trend")) / 1.3
    X["interaction_burden"] = 3 * X["ddi_major"] + 2 * X["ddi_moderate"]
    return X


_CAT = {
    "age": ("demographic", "years", "age drives baseline cost and mortality"), "sex": ("demographic", "0 male / 1 female", "rating factor"),
    "plan": ("demographic", "plan type code", "benefit design and risk adjustment"), "region": ("demographic", "region code", "price level"),
    "tobacco": ("demographic", "0/1", "classic rating factor"),
    "hcc_score": ("diagnosis", "0.25 + 0.01 x (age-60) + 0.03 x sex + sum of condition weights", "CMS-style risk adjustment"),
    "hcc_count": ("diagnosis", "number of conditions with non-zero HCC weight", "risk-adjusted condition burden"),
    "charlson": ("diagnosis", "sum of Charlson weights", "mortality comorbidity index"), "n_conditions": ("diagnosis", "count of chronic conditions", "comorbidity burden"),
    "hba1c_last": ("lab", "last HbA1c, %", "diabetes control"), "egfr_last": ("lab", "last eGFR, mL/min", "kidney function"),
    "hba1c_slope": ("lab", "OLS slope over 4 measurements", "worsening glucose control"), "egfr_slope": ("lab", "OLS slope over 4 measurements", "declining renal function"),
    "ldl": ("lab", "LDL mg/dL", "cardiovascular risk"), "sbp": ("lab", "systolic BP mmHg", "hypertension control"),
    "n_labs": ("lab", "number of measurements (missing not at random)", "sicker patients are tested more"),
    "hba1c_var": ("lab", "std of HbA1c", "unstable control"), "egfr_var": ("lab", "std of eGFR", "unstable renal function"),
    "med_count": ("medication", "number of drugs", "polypharmacy proxy"), "ddi_major": ("medication", "major interacting pairs among drugs", "adverse events"),
    "ddi_moderate": ("medication", "moderate interacting pairs", "adverse events"), "class_diversity": ("medication", "distinct ATC classes", "regimen complexity"),
    "polypharmacy_score": ("medication", "0-3 bucket of drug count", "polypharmacy"), "adherence_mpr": ("medication", "medication possession ratio", "adherence"),
    "prior_hosp": ("utilisation", "admissions in prior 12 months", "strongest utilisation signal"), "prior_ed": ("utilisation", "ED visits", "acute instability"),
    "prior_specialist": ("utilisation", "specialist visits", "care intensity"), "prior_pcp": ("utilisation", "primary-care visits", "access to care"),
    "log_prior_cost": ("utilisation", "ln(1 + prior-year cost)", "strongest single cost predictor"), "prior_cost_trend": ("utilisation", "year-over-year cost change", "trajectory"),
    "clinical_trajectory_score": ("fusion", "weighted z of HbA1c slope, -eGFR slope, cost trend", "single clinical deterioration number"),
    "interaction_burden": ("fusion", "3 x major + 2 x moderate DDI", "polypharmacy risk"),
}


def catalog():
    rows = [{"feature": c, "category": g, "formula": f, "financial_rationale": r, "validation": "range and missingness checked in ops.validate_patients"} for c, (g, f, r) in _CAT.items()]
    rows += [{"feature": "cond_" + c, "category": "diagnosis", "formula": f"1 if ICD-10 {CONDITIONS[c][0]} present", "financial_rationale": f"{c} drives utilisation and cost", "validation": "binary"} for c in COND_KEYS]
    return pd.DataFrame(rows)
