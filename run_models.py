"""Runs every model and writes results/results.json (about 1 minute on one CPU)."""
import json
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from data.processing.synth import build_world
from data.processing import ops
from data.features import clinical_features as CF
from models.clinical_nlp import clinical_nlp as N
from models.graph_network import graph_network as G
from models.survival import survival as S
from models.tabular import tabular as T
from models.ensemble import ensemble as E
from financial.insurance import insurance as INS
from financial.credit_risk import credit_risk as CR
from financial.pharma import pharma as PH
from explainability import explain as X

t0 = time.time()
tick = lambda m: print(m, round(time.time() - t0), flush=True)
w = build_world()
P = CF.add_features(w["patients"])
trm, tem = (P.split == "train").values, (P.split == "test").values
R = {"note": "ALL RESULTS COME FROM SYNTHETIC DATA (see docs/report.md)"}
R["data"] = {"members": len(P), "train": int(trm.sum()), "test": int(tem.sum()), "embargo_members": int((P.split == "embargo").sum()), "hospitals": int(w["hospitals"].hospital.nunique()),
             "hospital_quarters": len(w["hospitals"]), "trials": len(w["trials"]), "split": "train index years 2019-2020, embargo 2021, test 2022"}
R["data_quality"] = ops.validate_patients(P)
tt = w["truth"]
adm_te = tem & (P.admitted == 1).values
R["oracle_auroc_test"] = {"mortality_1y": round(float(roc_auc_score(P.mortality_1y[tem], tt.p_mortality_true[tem])), 3),
                          "readmission_30d": round(float(roc_auc_score(P.readmit_30d[adm_te], tt.p_readmit_true[adm_te])), 3),
                          "note": "AUROC of the hidden true probabilities: the best any model could do on this synthetic data"}
# ---- clinical NLP
tr, te = P[trm], P[tem]
R["clinical_nlp"], _ = N.diagnosis_classifier(tr, te)
R["clinical_nlp"]["ner"] = N.ner_f1(te)
cres, cfit = N.complexity_model(tr, te)
R["clinical_nlp"].update(cres)
P["nlp_complexity"] = N.complexity_scores(P, cfit, trm)
P["severity_cues"] = P["note"].map(N.severity_cue_count)
tick("nlp")
# ---- graph
GCOLS = [f"gnn_{i}" for i in range(16)]
P[GCOLS] = G.patient_embeddings(P, w["graph"], dim=16, train_mask=trm)
FULL = CF.STRUCT + ["nlp_complexity", "severity_cues"] + GCOLS
GVIEW = CF.DEMO + GCOLS + ["n_conditions"]
NVIEW = ["nlp_complexity", "severity_cues", "age"]
tick("graph")
# ---- survival
med = P[CF.STRUCT].median()
fill = lambda df, cols: df[cols].fillna(med).values
adm = P[P.admitted == 1]
a_tr, a_te = adm[adm.split == "train"], adm[adm.split == "test"]
A = fill(a_tr, CF.STRUCT), fill(a_te, CF.STRUCT)
cox = S.CoxPH().fit(A[0], a_tr.t_readmit_days, a_tr.e_readmit)
dh = S.DiscreteHazard(n_periods=12, period_len=30.0).fit(A[0], a_tr.t_readmit_days, a_tr.e_readmit)
R["survival"] = {"time_to_readmission": {"n_train": len(a_tr), "n_test": len(a_te), "events_test": int(a_te.e_readmit.sum()),
                                          "Cox_PH": S.evaluate(cox.risk(A[1]), a_te.t_readmit_days, a_te.e_readmit, [90, 180, 365]),
                                          "DiscreteHazard_boosting": S.evaluate(dh.risk(A[1]), a_te.t_readmit_days, a_te.e_readmit, [90, 180, 365])}}
dm = P[P.cond_diabetes == 1]
d_tr, d_te = dm[dm.split == "train"], dm[dm.split == "test"]
base = [c for c in CF.STRUCT if c not in ["hba1c_last", "egfr_last", "hba1c_slope", "egfr_slope", "hba1c_var", "egfr_var"]]
early = base + ["hba1c_0", "hba1c_1", "egfr_0", "egfr_1"]
upd = base + ["hba1c_3", "egfr_3", "hba1c_slope", "egfr_slope", "hba1c_var", "egfr_var"]
comp = {}
for name, cols in [("static_early_labs", early), ("updated_labs", upd)]:
    med_d = d_tr[cols].median()
    cx = S.CoxPH().fit(d_tr[cols].fillna(med_d).values, d_tr.t_complication_m, d_tr.e_complication)
    dd = S.DiscreteHazard(n_periods=20, period_len=3.0).fit(d_tr[cols].fillna(med_d).values, d_tr.t_complication_m, d_tr.e_complication)
    Xt = d_te[cols].fillna(med_d).values
    comp[name] = {"Cox_PH": S.evaluate(cx.risk(Xt), d_te.t_complication_m, d_te.e_complication, [3, 6, 12, 24]), "DiscreteHazard_boosting": S.evaluate(dd.risk(Xt), d_te.t_complication_m, d_te.e_complication, [3, 6, 12, 24])}
R["survival"]["time_to_complication_diabetics"] = {"n_train": len(d_tr), "n_test": len(d_te), "events_test": int(d_te.e_complication.sum()), **comp,
                                                     "note": "updated labs stand in for Dynamic-DeepHit: the hazard is re-estimated when newer measurements arrive"}
H = w["hospitals"]
Hc = CR.clinical_trend_features(H)
land = Hc[(Hc.q == 12)].copy()
land["time"] = np.where(land.breach_q >= 0, land.breach_q - 12 + 0.5, 27.0)
land["event"] = (land.breach_q >= 0).astype(int)
htr, hte = land[land.hospital % 3 != 0], land[land.hospital % 3 == 0]
hc = CR.FIN + CR.CLIN
hcx = S.CoxPH(l2=5.0).fit(htr[hc].values, htr.time, htr.event)
R["survival"]["time_to_covenant_breach_hospitals"] = {"n_train": len(htr), "n_test": len(hte), "events_test": int(hte.event.sum()), "Cox_PH_enhanced": S.evaluate(hcx.risk(hte[hc].values), hte.time, hte.event, [4, 8, 16]),
                                                      "note": "landmark at 2018Q1, hospitals split by id (independent units)"}
tick("survival")
Pall = fill(P, CF.STRUCT)
risk_all = cox.risk(Pall)
idx_tr = np.where((P.admitted == 1) & trm)[0]
for f in np.array_split(np.random.default_rng(0).permutation(idx_tr), 5):          # out-of-fold risk for training rows
    keep = np.setdiff1d(idx_tr, f)
    risk_all[f] = S.CoxPH().fit(Pall[keep], P.t_readmit_days.values[keep], P.e_readmit.values[keep]).risk(Pall[f])
P["cox_risk"] = risk_all


def run_task(target, rows_mask):
    d = P[rows_mask].sort_values(["index_year", "member_id"])
    dtr, dte = d[d.split == "train"], d[d.split == "test"]
    ytr, yte = dtr[target].values, dte[target].values
    views = {"LR_structured": CF.STRUCT, "HGB_structured": CF.STRUCT, "Graph_view": GVIEW, "NLP_view": NVIEW, "Survival_hazard": ["cox_risk"]}
    out = {"n_train": len(dtr), "n_test": len(dte), "events_test": int(yte.sum())}
    best_hgb, _ = T.tune_classifier(dtr[CF.STRUCT], ytr, n_iter=5)
    mk = {"LR_structured": T.make_lr, "HGB_structured": lambda b=best_hgb: HistGradientBoostingClassifier(**b.get_params()), "Graph_view": T.make_lr, "NLP_view": T.make_lr, "Survival_hazard": T.make_lr}
    tr_v = {n: dtr[c].values for n, c in views.items()}
    oof, ok = E.time_aware_oof(mk, tr_v, ytr, kind="clf")
    meta = E.fit_meta(oof, ytr, ok, "clf")
    preds = {}
    for n, c in views.items():
        preds[n] = mk[n]().fit(tr_v[n], ytr).predict_proba(dte[c].values)[:, 1]
        out[n] = {**T.class_metrics(yte, preds[n]), "auroc_ci95": T.bootstrap_ci(yte, preds[n])}
    pe = E.predict_meta(meta, np.column_stack([preds[n] for n in views]), "clf")
    out["Stacked_ensemble"] = {**T.class_metrics(yte, pe), "auroc_ci95": T.bootstrap_ci(yte, pe), "calibration_in_the_large": round(float(yte.sum() / pe.sum()), 3)}
    out["best_single_model"] = max(views, key=lambda n: out[n]["auroc"])
    out["ensemble_beats_best_single"] = bool(out["Stacked_ensemble"]["auroc"] > out[out["best_single_model"]]["auroc"])
    full_m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=60, l2_regularization=2.0, random_state=0).fit(dtr[FULL], ytr)
    out["HGB_all_features"] = T.class_metrics(yte, full_m.predict_proba(dte[FULL])[:, 1])
    return out, (full_m, dtr, dte, pe)


mort, mort_art = run_task("mortality_1y", P.index == P.index)
read, read_art = run_task("readmit_30d", (P.admitted == 1).values)
R["prediction_tasks"] = {"mortality_1y": mort, "readmission_30d": read}
tick("classification tasks")
# ---- cost / actuarial
d = P.sort_values(["index_year", "member_id"])
dtr, dte = d[d.split == "train"], d[d.split == "test"]
p_glm, _ = INS.glm_baseline(dtr, dte)
hgb_s, _ = T.tune_cost_model(dtr[CF.STRUCT], dtr["cost_12m"])
hgb_f = HistGradientBoostingRegressor(**hgb_s.get_params()).fit(dtr[FULL], dtr["cost_12m"])
p_s, p_f = hgb_s.predict(dte[CF.STRUCT]), hgb_f.predict(dte[FULL])
Xg_tr = INS.rating_matrix(dtr)
Xg_te = INS.rating_matrix(dte).reindex(columns=Xg_tr.columns, fill_value=0)
oofm = {"glm": lambda: Ridge(1.0), "hgb": lambda: HistGradientBoostingRegressor(loss="poisson", max_depth=3, max_iter=100, random_state=0)}
oof, ok = E.time_aware_oof(oofm, {"glm": Xg_tr.values, "hgb": dtr[FULL].values}, dtr["cost_12m"].values, kind="reg")
meta = E.fit_meta(oof, dtr["cost_12m"].values, ok, "reg", alpha=10.0)
p_ens = np.clip(E.predict_meta(meta, np.column_stack([Ridge(1.0).fit(Xg_tr.values, dtr["cost_12m"]).predict(Xg_te.values), p_f]), "reg"), 50, None)
R["actuarial"] = {"pricing": [INS.pricing_report(dte, p_glm, "GLM baseline (rating factors, Tweedie 1.5)"), INS.pricing_report(dte, p_s, "Boosting, structured clinical features"),
                              INS.pricing_report(dte, p_f, "Boosting, + notes and graph features (HealthRisk AI enhanced)"), INS.pricing_report(dte, p_ens, "Stacked ensemble (ridge meta-learner)")]}
strat, _ = INS.risk_stratification(dte, p_f)
strat_glm, _ = INS.risk_stratification(dte, p_glm)
R["actuarial"]["risk_stratification_enhanced"], R["actuarial"]["risk_stratification_glm"] = strat, strat_glm
R["actuarial"]["loss_ratio_spread_by_risk_decile"] = {"glm": INS.loss_ratio_by_decile(dte, p_glm), "enhanced": INS.loss_ratio_by_decile(dte, p_f)}
obs, cum, ult, prem, years = w["triangle"]
cl_u, lat = INS.chain_ladder(obs)
bf_u, _ = INS.bornhuetter_ferguson(obs, prem)
ea_u, _ = INS.emergence_adjusted(obs)
R["actuarial"]["ibnr"] = {"chain_ladder": INS.ibnr_accuracy(ult, cl_u, lat, years), "bornhuetter_ferguson": INS.ibnr_accuracy(ult, bf_u, lat, years), "emergence_adjusted_chain_ladder": INS.ibnr_accuracy(ult, ea_u, lat, years)}
tick("actuarial")
# ---- hospital credit
cr, Hc2, te_h, ENH = CR.run_pd_models(H)
R["hospital_credit"] = {"pd_models": cr, "early_warning": CR.early_warning_backtest(H)}
sp, _ = CR.bond_spread_model(te_h)
R["hospital_credit"]["bond_spread_model"] = sp
trh, teh = CR.split(Hc2)
R["hospital_credit"]["psi_train_vs_test"] = {c: round(CR.psi(trh[c].dropna().values, teh[c].dropna().values), 3) for c in ["op_margin", "dscr", "days_cash", "readmit_rate", "hcahps", "ed_boarding"]}
mt, me = CR.lr().fit(trh[CR.FIN], trh.y_breach_4q), CR.lr().fit(trh[ENH], trh.y_breach_4q)
cand = te_h[(te_h.op_margin.between(0.01, 0.03)) & (te_h.clinical_quality_score > 0.5)].head(1)
R["hospital_credit"]["worked_example"] = CR.worked_example(mt, me, cand, CR.FIN, ENH) if len(cand) else None
tick("credit")
# ---- pharma
pm_res, pm = PH.phase_model(w["trials"])
rets, sig, mkt = w["stocks"]
p1, b1 = PH.backtest(rets, sig, use_signal=True)
p0, b0 = PH.backtest(rets, sig, use_signal=False)
R["pharma"] = {"phase_success_model": pm_res, "pipeline_reports": PH.pipeline_report(pm), "pipeline_reports_generic_rates": PH.pipeline_report(None), "patent_cliff_example": PH.patent_cliff(2.5, 7),
               "portfolio_with_signal": PH.perf(p1, b1), "portfolio_without_signal": PH.perf(p0, b0),
               "trial_signal_example": PH.trial_signal(w["trials"].tail(1).assign(enrol_velocity=1.05, dropout=0.08, competitors=2), pm),
               "backtest_note": "signal alpha is built into the synthetic returns (IC about 0.06 assumed); the backtest checks the optimiser, not real alpha"}
tick("pharma")
# ---- explainability
full_m, dtr_m, dte_m, pe_m = mort_art
imp = X.global_importance(full_m, dte_m[FULL], dte_m["mortality_1y"])
R["explainability"] = {"mortality_permutation_importance": {k: round(float(v), 4) for k, v in imp.head(10).items()}}
pred_m = lambda df: full_m.predict_proba(df[FULL])[:, 1]
order = np.argsort(pred_m(dte_m))
R["explainability"]["local_explanations"] = {}
for nm, i in {"highest": order[-1], "median": order[len(order) // 2], "lowest": order[0]}.items():
    base_p, eff = X.local_explanation(pred_m, dte_m[FULL].iloc[[i]], dtr_m[FULL])
    R["explainability"]["local_explanations"][nm] = {"predicted_1y_mortality": round(base_p, 4), "drivers": {k: round(float(v), 4) for k, v in eff.items()}}
mem = dte[(dte.cond_diabetes == 1) & (dte.cond_ckd == 1)].head(1)
R["explainability"]["premium_breakdown_example"] = X.premium_breakdown(lambda df: hgb_f.predict(df[FULL]), mem[FULL], dtr[FULL])
full_r, rtr, rte, _ = read_art
pred_r = lambda df: full_r.predict_proba(df[FULL])[:, 1]
hi = rte[pred_r(rte) > np.quantile(pred_r(rte), 0.9)].head(3)
R["explainability"]["counterfactuals_readmission"] = [X.counterfactual(pred_r, hi[FULL].iloc[[i]], target=float(np.quantile(pred_r(rte), 0.5))) for i in range(len(hi))]
g_, a_ = X.pdp(full_m, dte_m[FULL].head(1500), "hba1c_last")
R["explainability"]["pdp_hba1c_mortality"] = {"grid": [round(float(v), 2) for v in g_], "avg_risk": [round(float(v), 4) for v in a_]}
R["feature_catalog_rows"] = len(CF.catalog()) + len(GCOLS) + 2
CF.catalog().to_csv("docs/feature_catalog.csv", index=False)
X.REGULATORY_MATRIX.to_csv("docs/regulatory_matrix.csv", index=False)
ops.log_run("all_models", {"n_features": len(FULL)}, {"mortality_auroc": mort["Stacked_ensemble"]["auroc"]})
json.dump(R, open("results/results.json", "w"), indent=1, default=float)
tick("ALL DONE")
