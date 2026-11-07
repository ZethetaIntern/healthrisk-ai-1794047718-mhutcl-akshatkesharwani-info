"""Six-stage data quality protocol, experiment log (MLflow stand-in) and config loader."""
import json
import os
import time
import numpy as np
from data.processing.synth import COND_KEYS, CONDITIONS
from data.features.clinical_features import icd10_hierarchy


def validate_patients(P):
    """1 completeness, 2 accuracy (code vs note), 3 consistency, 4 timeliness, 5 conformity (ICD-10 format), 6 deduplication."""
    from models.clinical_nlp.clinical_nlp import extract_entities
    comp = {c: round(float(P[c].notna().mean()), 3) for c in ["age", "sex", "plan", "prior_cost", "cost_12m", "hba1c_last", "egfr_last"]}
    agree = []
    for _, r in P.sample(1500, random_state=0).iterrows():
        found, _ = extract_entities(r["note"])
        have = {c for c in COND_KEYS if r["cond_" + c] == 1}
        agree.append(len(found & have) / max(len(found), 1))
    consistency = {"bph_in_females": int(((P.cond_bph == 1) & (P.sex == 1)).sum()), "negative_cost": int((P.cost_12m < 0).sum()),
                   "mortality_without_age": int(((P.mortality_1y == 1) & P.age.isna()).sum()), "hba1c_without_diabetes": int(((P.hba1c_0.notna()) & (P.cond_diabetes == 0)).sum())}
    conform = float(np.mean([icd10_hierarchy(CONDITIONS[c][0]) is not None for c in COND_KEYS]))
    return {"1_completeness": comp, "2_accuracy_code_vs_note_agreement": round(float(np.mean(agree)), 3), "3_consistency_violations": consistency,
            "4_timeliness": "synthetic data has no reporting lag", "5_conformity_icd10_share_valid": conform,
            "6_duplicate_member_ids": int(P.member_id.duplicated().sum()),
            "all_critical_checks_passed": bool(consistency["bph_in_females"] == 0 and consistency["negative_cost"] == 0 and conform == 1.0 and P.member_id.is_unique)}


def log_run(name, params, metrics, path="results/experiments.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": name, "params": params, "metrics": metrics}, default=float) + "\n")


def load_config(path="configs/config.json"):
    return json.load(open(path))
