"""Explainability: permutation importance, occlusion explanations (SHAP stand-in), partial dependence, counterfactuals, regulatory matrix."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence


def global_importance(model, X, y, scoring="roc_auc", n_repeats=3):
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=0)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def local_explanation(predict, row, reference, top=6):
    """Replace one feature at a time by the reference median; the change in the prediction is that feature's effect."""
    base = float(predict(row)[0])
    ref = reference.median()
    eff = {}
    for c in row.columns:
        alt = row.copy()
        alt[c] = ref[c]
        eff[c] = base - float(predict(alt)[0])
    return base, pd.Series(eff).sort_values(key=np.abs, ascending=False).head(top)


def premium_breakdown(predict_cost, row, reference, top=6):
    base_cost, eff = local_explanation(predict_cost, row, reference, top)
    return {"predicted_annual_cost": round(base_cost, 0), "largest_contributions": {k: round(float(v), 0) for k, v in eff.items()}}


def pdp(model, X, feature, grid=12):
    r = partial_dependence(model, X, [feature], grid_resolution=grid)
    return r["grid_values"][0], r["average"][0]


ACTIONS = {"hba1c_last": (-0.2, 5.5), "egfr_last": (+3.0, 90.0), "prior_hosp": (-1, 0), "prior_specialist": (+1, 8), "adherence_mpr": (+0.05, 1.0), "med_count": (-1, 0)}


def counterfactual(predict, row, target, max_steps=12):
    """Greedy search over clinically actionable features: smallest set of changes that brings the risk below `target`."""
    cur = row.copy()
    p = float(predict(cur)[0])
    path = []
    for _ in range(max_steps):
        if p <= target:
            break
        best = None
        for f, (step, limit) in ACTIONS.items():
            if f not in cur or pd.isna(cur[f].iloc[0]):
                continue
            v = float(cur[f].iloc[0])
            nv = max(v + step, limit) if step < 0 else min(v + step, limit)
            if nv == v:
                continue
            alt = cur.copy()
            alt[f] = nv
            pa = float(predict(alt)[0])
            if best is None or pa < best[0]:
                best = (pa, f, nv)
        if best is None or best[0] >= p - 1e-9:
            break
        p, f, nv = best
        cur[f] = nv
        path.append({"change": f, "new_value": round(nv, 2), "risk_after": round(p, 4)})
    return {"start": round(float(predict(row)[0]), 4), "target": target, "reached": p <= target, "final": round(p, 4), "steps": path}


REGULATORY_MATRIX = pd.DataFrame([
    ["Permutation importance (global)", "State rate-filing justification of rating factors", "Shows which factors move predictions and by how much"],
    ["Occlusion / premium breakdown (per member)", "Insurance rate filing, ECOA adverse-action reasons", "Gives point-in-time reasons for a price or decision"],
    ["Partial dependence plots", "CMS risk-adjustment transparency", "Shows the shape of the effect of each clinical factor"],
    ["Counterfactual search", "ECOA adverse action, consumer transparency", "States what would have to change for a better outcome"],
    ["Model cards", "EU AI Act conformity assessment (high risk), documentation duties", "Documents data, limits, performance, monitoring"],
    ["Feature selection record and catalog", "HIPAA minimum-necessary standard", "Justifies each clinical field used in financial models"],
    ["Subgroup predictive ratios", "Fairness and anti-discrimination review", "Checks pricing accuracy across plan, region and age groups"],
], columns=["Explainability feature", "Regulatory requirement addressed", "What it provides"])
