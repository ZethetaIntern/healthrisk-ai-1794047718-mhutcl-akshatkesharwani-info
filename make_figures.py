"""EDA charts, result charts and architecture diagram -> reports/figures/ (run after run_models.py and run_sim.py)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data.processing.synth import build_world, COND_KEYS
from data.features import clinical_features as CF

P_ = "reports/figures/"
w = build_world()
P = CF.add_features(w["patients"])
H, trials = w["hospitals"], w["trials"]
R = json.load(open("results/results.json"))
S = json.load(open("results/simulation.json"))
n = [0]


def save(title, name=None):
    n[0] += 1
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(P_ + (name or f"eda_{n[0]:02d}") + ".png", dpi=100)
    plt.close()


plt.figure(); plt.hist(P.age, bins=30); plt.xlabel("age"); save("Member age distribution")
plt.figure(); plt.hist(np.log1p(P.cost_12m[P.cost_12m > 0]), bins=40); plt.xlabel("ln(1 + cost)"); save("Next-12-month cost (positive costs, log scale)")
plt.figure(figsize=(8, 4)); P[["cond_" + c for c in COND_KEYS]].mean().rename(lambda s: s[5:]).sort_values().plot.barh(); save("Condition prevalence")
plt.figure(); P.groupby("n_conditions").cost_12m.mean().plot.bar(); plt.ylabel("mean cost"); save("Mean cost by number of conditions")
plt.figure(); P.groupby(pd.qcut(P.clinical_trajectory_score, 10, labels=False)).mortality_1y.mean().plot.bar(); save("1-year mortality by clinical trajectory decile")
plt.figure(); P.groupby("plan").cost_12m.mean().rename(dict(enumerate(["Medicare Adv.", "Commercial", "Medicaid"]))).plot.bar(); save("Mean cost by plan")
plt.figure(figsize=(8, 4)); P[P.cond_diabetes == 1][[f"hba1c_{m}" for m in range(4)]].mean().plot(marker="o"); save("Average HbA1c trajectory (diabetics)")
plt.figure(figsize=(8, 4)); P[P.cond_ckd == 1][[f"egfr_{m}" for m in range(4)]].mean().plot(marker="o"); save("Average eGFR trajectory (CKD)")
plt.figure(); P.groupby("med_count").readmit_30d.mean().plot.bar(); save("30-day readmission rate by medication count (all members)")
plt.figure(figsize=(8, 4)); H.groupby("q").readmit_rate.mean().plot(); plt.axvspan(20, 24, alpha=0.15); save("Hospital readmission rate over time (2020 shock shaded)")
plt.figure(figsize=(8, 4)); H.groupby("q").y_breach_4q.mean().plot(); save("Share of hospitals breaching a covenant within 4 quarters")
plt.figure(); trials.groupby("phase").success.mean().plot.bar(); plt.ylabel("success rate"); save("Clinical-trial success rate by phase")
obs, cum, ult, prem, years = w["triangle"]
plt.figure(figsize=(8, 4))
for i in range(0, 10, 2):
    plt.plot(obs[i, : 10 - i], marker="o", label=str(years[i]))
plt.legend(); plt.xlabel("development year"); save("Claims triangle: cumulative paid by accident year")
plt.figure(figsize=(8, 4)); P[CF.STRUCT].isna().mean().sort_values(ascending=False).head(8).plot.barh(); save("Missing data (labs are missing for people without the condition)")
c = P[["age", "n_conditions", "hcc_score", "med_count", "ddi_major", "prior_cost_trend", "adherence_mpr", "cost_12m", "mortality_1y"]].corr()
plt.figure(figsize=(7, 6)); plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(c)), c.columns, rotation=90, fontsize=7); plt.yticks(range(len(c)), c.columns, fontsize=7); save("Correlations")
names = ["LR_structured", "HGB_structured", "Graph_view", "NLP_view", "Survival_hazard", "Stacked_ensemble"]
for task, key, tgt, fname in [("1-year mortality", "mortality_1y", 0.78, "result_mortality_auroc"), ("30-day readmission", "readmission_30d", 0.72, "result_readmission_auroc")]:
    t = R["prediction_tasks"][key]
    plt.figure(figsize=(8, 4)); plt.barh(names, [t[k]["auroc"] for k in names]); plt.axvline(tgt, color="r", ls=":", label=f"brief target {tgt}"); plt.axvline(R["oracle_auroc_test"][key], color="g", ls="--", label="oracle")
    plt.xlim(0.5, 0.95); plt.legend(); save(f"{task}: AUROC by model", fname)
pr = R["actuarial"]["pricing"]
plt.figure(figsize=(8, 4)); plt.bar(["GLM", "Boost struct.", "Boost + notes/graph", "Ensemble"], [p["R2"] for p in pr]); plt.axhline(0.25, color="r", ls=":", label="brief target 0.25"); plt.legend(); plt.ylabel("R-squared (cost)"); save("Cost prediction R-squared", "result_cost_r2")
imp = R["explainability"]["mortality_permutation_importance"]
plt.figure(figsize=(8, 4)); plt.barh(list(imp)[::-1], list(imp.values())[::-1]); plt.xlabel("AUROC drop when shuffled"); save("Mortality model: most important features", "result_importance")
pdp = R["explainability"]["pdp_hba1c_mortality"]
plt.figure(figsize=(6, 4)); plt.plot(pdp["grid"], pdp["avg_risk"], marker="o"); plt.xlabel("HbA1c"); plt.ylabel("average predicted mortality"); save("Partial dependence: HbA1c", "result_pdp")
plt.figure(figsize=(7, 4)); plt.bar([f"M{i} {k}" for i in range(1, 5) for k in ("naive", "AI")], [S[f"mode{i}"][f"{k}_mean"] for i in range(1, 5) for k in ("naive", "ai")]); plt.xticks(rotation=45); plt.ylabel("mean score / 1000"); save("HealthRisk Lab scores", "result_lab_scores")
fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off")
boxes = [(0.01, 0.66, "Data layer\nclaims, labs, notes,\nhospital panel, trials"), (0.22, 0.66, "Feature layer\nICD-10, HCC, labs,\nmeds, utilisation"), (0.46, 0.84, "Clinical NLP"), (0.46, 0.68, "Graph network"),
         (0.46, 0.52, "Survival models"), (0.46, 0.36, "Tabular boosting"), (0.68, 0.60, "Stacking ensemble\n(ridge / logistic meta-learner)"), (0.20, 0.10, "Financial modules\ninsurance | hospital credit | pharma"), (0.60, 0.10, "Explainability +\nHealthRisk Lab (4 modes)")]
for x, y, t in boxes:
    ax.text(x, y, t, ha="left", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#e8f4ec", ec="#345"))
for a, b in [((0.16, 0.70), (0.22, 0.70)), ((0.40, 0.70), (0.46, 0.70)), ((0.62, 0.62), (0.68, 0.64)), ((0.78, 0.55), (0.55, 0.18)), ((0.55, 0.30), (0.48, 0.22))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->"))
save("HealthRisk AI architecture", "architecture")
print("figures", n[0])
