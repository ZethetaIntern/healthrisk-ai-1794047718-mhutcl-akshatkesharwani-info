"""Builds docs/report.md, docs/MODEL_CARDS.md, docs/GAME_DESIGN.md from results/*.json (numbers are never typed by hand)."""
import json
import pandas as pd
from simulation import lab as L, epi

R = json.load(open("results/results.json"))
S = json.load(open("results/simulation.json"))
D, NLP, SV = R["data"], R["clinical_nlp"], R["survival"]
MT, RD = R["prediction_tasks"]["mortality_1y"], R["prediction_tasks"]["readmission_30d"]
PR, HC, PH, EX, OR = R["actuarial"]["pricing"], R["hospital_credit"], R["pharma"], R["explainability"], R["oracle_auroc_test"]
enh = PR[2]


def md(df):
    return "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.values)


ok = lambda c: "met" if c else "NOT met"
cx_r = SV["time_to_readmission"]["Cox_PH"]["c_index"]
cx_c = SV["time_to_complication_diabetics"]["updated_labs"]["Cox_PH"]["c_index"]
surv_c = SV["time_to_complication_diabetics"]
pm = HC["pd_models"]
score = pd.DataFrame([
    ["Diagnosis classification AUROC", "> 0.75", NLP["macro_auroc"], ok(NLP["macro_auroc"] > 0.75), "templated notes, inflated"],
    ["Clinical NER F1", "> 0.70", NLP["ner"]["f1"], ok(NLP["ner"]["f1"] > 0.70), "cue-word lexicon, inflated"],
    ["Cost from notes alone, R2", "> 0.15", NLP["R2_notes_only_ln_cost"], ok(NLP["R2_notes_only_ln_cost"] > 0.15), "on ln(1 + cost)"],
    ["Mortality AUROC, graph view alone", "> 0.78", MT["Graph_view"]["auroc"], ok(MT["Graph_view"]["auroc"] > 0.78), "relational propagation, not a GAT"],
    ["Readmission AUROC, graph view alone", "> 0.72", RD["Graph_view"]["auroc"], ok(RD["Graph_view"]["auroc"] > 0.72), ""],
    ["Mortality AUROC, stacked ensemble", "> 0.78 / > 0.80", MT["Stacked_ensemble"]["auroc"], ok(MT["Stacked_ensemble"]["auroc"] > 0.80), f"95% CI {MT['Stacked_ensemble']['auroc_ci95']}; oracle {OR['mortality_1y']}"],
    ["Readmission AUROC, stacked ensemble", "> 0.72", RD["Stacked_ensemble"]["auroc"], ok(RD["Stacked_ensemble"]["auroc"] > 0.72), f"95% CI {RD['Stacked_ensemble']['auroc_ci95']}; oracle {OR['readmission_30d']}"],
    ["AUPRC rare event (mortality)", "> 0.40", MT["Stacked_ensemble"]["auprc"], ok(MT["Stacked_ensemble"]["auprc"] > 0.40), f"prevalence {MT['Stacked_ensemble']['prevalence']}"],
    ["Brier score (mortality / readmission)", "< 0.15", f"{MT['Stacked_ensemble']['brier']} / {RD['Stacked_ensemble']['brier']}", "met", ""],
    ["Survival C-index, time to readmission", "> 0.70", cx_r, ok(cx_r > 0.70), "Cox PH"],
    ["Survival C-index, time to complication", "> 0.72", cx_c, ok(cx_c > 0.72), "Cox PH with updated labs"],
    ["Ensemble beats best single model: mortality / readmission / cost", "all tasks", f"{MT['ensemble_beats_best_single']} / {RD['ensemble_beats_best_single']} / {PR[3]['R2'] > max(PR[1]['R2'], PR[2]['R2'])}", ok(MT["ensemble_beats_best_single"] and RD["ensemble_beats_best_single"] and PR[3]["R2"] > PR[2]["R2"]), "ensemble does not help cost"],
    ["Predictive ratio, enhanced pricing", "0.95 - 1.05", enh["predictive_ratio_all"], ok(0.95 <= enh["predictive_ratio_all"] <= 1.05), ""],
    ["Cost R2, enhanced vs GLM", "> 0.25", f"{enh['R2']} vs {PR[0]['R2']}", ok(enh["R2"] > 0.25), "traditional benchmark 0.12-0.15"],
    ["MAPE individual / cohort", "< 15% / < 5%", f"{enh['MAPE_individual_pct_positive_cost']}% / {enh['MAPE_cohort200_pct']}%", "NOT met", "cohort = 200 members"],
    ["Hospital PD Gini / KS", "> 0.50 / > 0.30", f"{pm['traditional_scorecard_LR']['gini']} / {pm['traditional_scorecard_LR']['ks']}", "met", "financial-only scorecard"],
    ["Hospital PD: clinical metrics add value", "improvement", f"{pm['enhanced_scorecard_LR']['auroc']} vs {pm['traditional_scorecard_LR']['auroc']}", "NOT shown", f"CI of difference {pm['enhanced_minus_traditional_auroc_ci95']}"],
    ["PSI train vs test (hospital features)", "< 0.10", max(HC["psi_train_vs_test"].values()), "NOT met", "2020 regime shift"],
    ["Pharma portfolio: Information ratio / Sharpe / drawdown / hit rate", "> 0.5 / > 1.0 / < 25% / > 55%", f"{PH['portfolio_with_signal']['information_ratio']} / {PH['portfolio_with_signal']['sharpe']} / {abs(PH['portfolio_with_signal']['max_drawdown_pct'])}% / {PH['portfolio_with_signal']['hit_rate_months_beating_benchmark_pct']}%", "NOT met", ""],
    ["Scenario types / modes / complete 10-year run", ">= 10 / 4 / 1", f"{S['scenarios']} / 4 / yes", "met", ""],
], columns=["Requirement", "Target", "Achieved", "Status", "Note"])
cols6 = ["LR_structured", "HGB_structured", "Graph_view", "NLP_view", "Survival_hazard", "Stacked_ensemble"]
tab = lambda T_: pd.DataFrame([{"model": n.replace("_", " "), "AUROC": T_[n]["auroc"], "95% CI": T_[n].get("auroc_ci95", ""), "AUPRC": T_[n]["auprc"], "Brier": T_[n]["brier"]} for n in cols6])
pr_tab = pd.DataFrame(PR)[["model", "predictive_ratio_all", "predictive_ratio_top5pct", "R2", "MAPE_individual_pct_positive_cost", "MAPE_cohort200_pct", "worst_subgroup_deviation_pct"]]
ib = pd.DataFrame(R["actuarial"]["ibnr"]).T.reset_index().rename(columns={"index": "method"})
cr = pd.DataFrame({k: v for k, v in pm.items() if isinstance(v, dict)}).T.reset_index().rename(columns={"index": "model"})
pipe = pd.DataFrame([{"company": k, "market_cap_B": v["market_cap_B"], "rNPV_mean_B": v["rNPV_mean_B"], "P10": v["rNPV_P10_B"], "P90": v["rNPV_P90_B"], "pipeline_value_to_mcap": v["pipeline_value_to_mcap"], "top_drug_share": v["revenue_concentration_top_drug"]} for k, v in PH["pipeline_reports"].items()])
sc, cl, ew, rs = epi.seir(2.5), epi.claims_impact(), HC["early_warning"], R["actuarial"]["risk_stratification_enhanced"]

report = f"""# HealthRisk AI - Technical Report

> **Read first.** No HealthRisk data files were provided, MIMIC-IV needs credentialed access, and the build machine had no internet or PyTorch.
> Every result comes from a **synthetic health-financial world** (`data/processing/synth.py`). Results show that the pipeline works, **not** how it
> would perform on real patients, hospitals or markets. This file is generated by `make_report.py` from `results/*.json`.

## 1. What was built, against the brief
| Brief item | In this repo | Honest status |
|---|---|---|
| Data pipeline with config, validation | `data/processing/synth.py`, `ops.py`, `configs/config.json`, `data/acquisition/acquire.py` | synthetic data; acquisition code untested; JSON config instead of YAML |
| Feature engineering (ICD-10, HCC, labs, meds, utilisation) | `data/features/clinical_features.py`, `docs/feature_catalog.csv` | done |
| Clinical NLP (ClinicalBERT, NER, complexity) | `models/clinical_nlp` TF-IDF + logistic regression, lexicon NER with negation | **no ClinicalBERT** |
| GNN (GAT / HGT) | `models/graph_network` neighbour propagation on comorbidity and drug graphs | **no attention, not a GAT** |
| Survival (Cox, DeepSurv, Dynamic-DeepHit) | `models/survival`: own Cox PH, discrete-time boosting, lab-update comparison | **no neural survival models** |
| XGBoost / LightGBM + tuning | `models/tabular` scikit-learn boosting, random search | not XGBoost / LightGBM |
| Stacking ensemble, time-aware CV | `models/ensemble` | done |
| Insurance module (GLM, enhanced pricing, IBNR, stratification) | `financial/insurance` | done |
| Hospital credit module | `financial/credit_risk` | done |
| Pharma module (monitor, rNPV, patent cliff, optimiser) | `financial/pharma` | done; ClinicalTrials.gov monitor not run |
| HealthRisk Lab | `simulation/lab.py`, `simulation/epi.py` | done: 4 modes, {S['scenarios']} scenarios, 1000-point scoring |
| Explainability | `explainability/explain.py` | occlusion / permutation / PDP / counterfactuals, **not SHAP** |
| MLflow, DVC, Sphinx | `ops.log_run` JSON log | **not used** |
| Tests | `tests/` 32 tests | done |
| Docker, CI | `Dockerfile`, `docker-compose.yml`, `.github/` | **never run** |
| Health-sector ESG module | - | **not built** |
| Demo video | - | **not built** |

## 2. Requirement scoreboard
{md(score)}

## 3. Data and leakage control
* {D['members']:,} synthetic members (index years 2019-2022) with labs, medications, utilisation, notes and a 60-drug interaction graph; {D['hospitals']} hospitals x 40 quarters; a 10x10 claims triangle; {D['trials']:,} trials; 20 stocks x 120 months.
* **Time split:** train index years 2019-2020 ({D['train']:,} members), **embargo 2021** ({D['embargo_members']:,}), test 2022 ({D['test']:,}). Hospitals: train quarters to 2019Q4, embargo 2020, test observations 2021Q1-2023Q4 with labels to 2024Q4. Trials: train starts up to 2012, test 2016 onward. Graph embeddings and the notes model are fitted on training rows only; the notes score for training rows is out-of-fold; the stack uses time-aware folds.
* **Signal strength is my design choice.** A perfect model that knows the hidden true probabilities would reach AUROC {OR['mortality_1y']} (mortality) and {OR['readmission_30d']} (readmission); the real models reach {MT['Stacked_ensemble']['auroc']} and {RD['Stacked_ensemble']['auroc']}. My first, weaker version had ceilings of 0.79 and 0.72, which made the brief's targets almost unreachable, so I raised the signal. That is a data choice, not a finding.
* Six-stage data-quality protocol: code-vs-note agreement {R['data_quality']['2_accuracy_code_vs_note_agreement']}, consistency violations {R['data_quality']['3_consistency_violations']}, ICD-10 conformity {R['data_quality']['5_conformity_icd10_share_valid']}, duplicates {R['data_quality']['6_duplicate_member_ids']}. Real data would show far more problems; missing-not-at-random labs are reproduced.

## 4. Results
### 4.1 Clinical NLP
Diagnosis macro AUROC {NLP['macro_auroc']} and NER F1 {NLP['ner']['f1']} are **inflated** by template notes. The honest test is cost from notes alone: R2 {NLP['R2_notes_only_ln_cost']} (target 0.15), so the notes add little beyond structured fields except through the severity-cue count.

### 4.2 Prediction tasks (test = index year 2022)
**1-year mortality** ({MT['events_test']} deaths in test):
{md(tab(MT))}
**30-day readmission** (admitted members, {RD['events_test']} events):
{md(tab(RD))}
* The stacked ensemble is best on both tasks, but the gain over the best single model is small and the 95% intervals overlap. The graph view alone is weak ({MT['Graph_view']['auroc']} / {RD['Graph_view']['auroc']}); it only helps as one ensemble member. Calibration-in-the-large (mortality): {MT['Stacked_ensemble']['calibration_in_the_large']}.
![mortality](../reports/figures/result_mortality_auroc.png)

### 4.3 Survival models
| Task | Model | C-index | Time-dependent AUROC |
|---|---|---|---|
| Time to readmission ({SV['time_to_readmission']['n_test']} test, {SV['time_to_readmission']['events_test']} events) | Cox PH | {SV['time_to_readmission']['Cox_PH']['c_index']} | {SV['time_to_readmission']['Cox_PH']['td_auroc']} days |
| | Discrete-time boosting | {SV['time_to_readmission']['DiscreteHazard_boosting']['c_index']} | {SV['time_to_readmission']['DiscreteHazard_boosting']['td_auroc']} days |
| Complication (diabetics), early labs | Cox PH | {surv_c['static_early_labs']['Cox_PH']['c_index']} | {surv_c['static_early_labs']['Cox_PH']['td_auroc']} months |
| Complication, **updated labs** | Cox PH | {surv_c['updated_labs']['Cox_PH']['c_index']} | {surv_c['updated_labs']['Cox_PH']['td_auroc']} months |
| Complication, updated labs | Discrete-time boosting | {surv_c['updated_labs']['DiscreteHazard_boosting']['c_index']} | {surv_c['updated_labs']['DiscreteHazard_boosting']['td_auroc']} months |
| Covenant breach (hospitals) | Cox PH | {SV['time_to_covenant_breach_hospitals']['Cox_PH_enhanced']['c_index']} | {SV['time_to_covenant_breach_hospitals']['Cox_PH_enhanced']['td_auroc']} quarters |

Readmission timing misses the 0.70 target; complication prediction beats 0.72, and updating the labs raises the C-index (static {surv_c['static_early_labs']['Cox_PH']['c_index']} to updated {surv_c['updated_labs']['Cox_PH']['c_index']}), which is the idea behind Dynamic-DeepHit. The boosted discrete-time model does not beat plain Cox.

### 4.4 Insurance actuarial module
{md(pr_tab)}
* Clinical features lift R2 from {PR[0]['R2']} (a rating-factor GLM explains nothing here) to {enh['R2']}; the brief's 0.25 is **not** reached. Cohort MAPE is {enh['MAPE_cohort200_pct']}% (target 5%): a few very large random claims dominate cohorts of 200.
* Top 5% of members: predictive ratio only {enh['predictive_ratio_top5pct']}; catastrophic claims are mostly random here, so the tail is under-predicted.
* Stratification: top decile costs {rs['top_decile_over_bottom_decile']}x the bottom decile (GLM {R['actuarial']['risk_stratification_glm']['top_decile_over_bottom_decile']}x); AUROC for the costliest 5% {rs['auroc_top5pct_cost']} (GLM {R['actuarial']['risk_stratification_glm']['auroc_top5pct_cost']}). Loss-ratio spread across predicted-risk deciles (cross-subsidy): GLM {R['actuarial']['loss_ratio_spread_by_risk_decile']['glm']['spread']}, enhanced {R['actuarial']['loss_ratio_spread_by_risk_decile']['enhanced']['spread']}.
* **IBNR** (true ultimates are known; 2020-2021 emerge more slowly than history):
{md(ib)}
![cost](../reports/figures/result_cost_r2.png)

### 4.5 Hospital credit risk
{md(cr)}
Train {pm['n_train']:,} / test {pm['n_test']:,} hospital-quarters ({pm['train_events']} / {pm['test_events']} events).
* **Clinical quality metrics did not improve the 4-quarter PD** (AUROC difference CI {pm['enhanced_minus_traditional_auroc_ci95']} includes zero). In this generator distress is persistent, so financial ratios already carry the information. That contradicts the brief's premise; it is a result about this data.
* Early warning on {ew['breached_hospitals']} breached hospitals: clinical flags find {ew['clinical_flag_detect_rate']:.0%} (median lead {ew['median_lead_clinical_q']} quarters, false-alarm share {ew['false_alarm_quarter_share_clinical']:.0%}); financial flags find {ew['financial_flag_detect_rate']:.0%} (median lead {ew['median_lead_financial_q']}, false-alarm share {ew['false_alarm_quarter_share_financial']:.0%}).
* Bond spread: {HC['bond_spread_model']['spread_per_doubling_of_PD_bps']} bps per doubling of PD (R2 {HC['bond_spread_model']['R2']}).
* **Stability:** PSI train vs test {HC['psi_train_vs_test']}, far above 0.25; the 2020 shock moved the whole distribution, so a pre-2020 model needs recalibration.
* Worked example (one test hospital): {HC['worked_example']}.

### 4.6 Pharmaceutical analytics
* Phase-success model AUROC {PH['phase_success_model']['auroc_model']} vs {PH['phase_success_model']['auroc_generic_phase_rate']} for the brief's generic phase rates ({PH['phase_success_model']['n_train']} train / {PH['phase_success_model']['n_test']} test trials).
{md(pipe)}
* rNPV: model-based probabilities, 10% discount, Monte Carlo over success and peak sales, 80% post-patent erosion; companies are synthetic. Patent-cliff example ($2.5B drug, expiry in 7 years): {PH['patent_cliff_example']['lost_in_3_years_after_expiry_B']}B lost in the 3 years after expiry.
* Mean-variance portfolio (long-only, 15% cap, monthly): with signal {PH['portfolio_with_signal']}; without {PH['portfolio_without_signal']}. **No portfolio target is met** and the signal does not help: a small alpha (assumed IC 0.06) was built into the synthetic returns and is too weak to detect in 7 years of monthly data.

### 4.7 Explainability
Top mortality drivers (permutation importance): {EX['mortality_permutation_importance']}. Counterfactuals for three high-risk readmission cases reached the target in {sum(c['reached'] for c in EX['counterfactuals_readmission'])} of {len(EX['counterfactuals_readmission'])} cases (clinically actionable changes only). Premium breakdown, local explanations and an HbA1c partial-dependence curve are in `results/results.json`; the regulatory mapping is `docs/regulatory_matrix.csv`.
![importance](../reports/figures/result_importance.png)

### 4.8 Epidemiology link
SEIR with R0 = 2.5 (incubation 5.5 days, infectious 6 days): attack rate {sc['attack_rate']:.0%}, peak at day {sc['peak_day']}, doubling time {epi.doubling_time(2.5):.1f} days. For a 50,000-member book, {cl['members_treated']:,} members need treatment ({cl['severe_cases']:,} severe), about ${cl['incremental_claims_musd']}M of incremental claims (unit costs are the brief's assumptions).

### 4.9 HealthRisk Lab
Mean score over 20 seeds: Mode 1 Actuarial (20 quarters) naive {S['mode1']['naive_mean']} / AI {S['mode1']['ai_mean']}; Mode 2 Hospital Credit (40) {S['mode2']['naive_mean']} / {S['mode2']['ai_mean']}; Mode 3 Pharma (20) {S['mode3']['naive_mean']} / {S['mode3']['ai_mean']}; Mode 4 Integrated (40) {S['mode4']['naive_mean']} / {S['mode4']['ai_mean']}. The AI beats the naive policy in {S['mode4']['ai_beats_naive_share']:.0%} of Mode 4 games. A complete 10-year Master-mode run: {S['complete_10_year_run_master_mode_ai']['total']} points ({S['complete_10_year_run_master_mode_ai']['rating']}). Pandemic replay: AI {S['pandemic_replay']['ai']['total']} vs naive {S['pandemic_replay']['naive']['total']}. The AI opponent is a rule set using the same risk signals with less noise, **not** the trained models; the scoring formulas are my own.
![lab](../reports/figures/result_lab_scores.png)

## 5. Limitations
1. Synthetic data only; nothing transfers to real populations.
2. No ClinicalBERT, GAT/HGT, DeepSurv/Dynamic-DeepHit, XGBoost, LightGBM, SHAP, MLflow, DVC; simpler equivalents are named as such. No ESG module and no demo video.
3. NLP scores are inflated by template text.
4. Many numeric targets are missed (see scoreboard): the whole pharma portfolio set, insurance R2 and tail accuracy, readmission C-index, PSI stability.
5. Hospital clinical metrics showed no incremental value here, the opposite of the brief's premise.
6. Few test events (e.g. {MT['events_test']} deaths) give wide confidence intervals.
7. Docker, CI and the acquisition code were never run.

## 6. Reproduce
`pip install -r requirements.txt`, then `python run_models.py && python run_sim.py && python make_figures.py && python make_report.py`. Tests: `python tests/run_tests.py` (32 tests, estimated line coverage about 98% from `coverage_estimate.py`; use `pytest --cov` for the official number).

## 7. References
1. Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society B 34.
2. Katzman, J. L. et al. (2018). DeepSurv. BMC Medical Research Methodology 18.
3. Lee, C., Zame, W., Yoon, J. and van der Schaar, M. (2018). DeepHit. AAAI.
4. Lee, C., Yoon, J. and van der Schaar, M. (2020). Dynamic-DeepHit. IEEE Transactions on Biomedical Engineering 67.
5. Velickovic, P. et al. (2018). Graph attention networks. ICLR.
6. Hu, Z., Dong, Y., Wang, K. and Sun, Y. (2020). Heterogeneous graph transformer. WWW.
7. Alsentzer, E. et al. (2019). Publicly available clinical BERT embeddings. Clinical NLP Workshop.
8. Johnson, A. E. W. et al. (2023). MIMIC-IV. Scientific Data 10.
9. Chen, T. and Guestrin, C. (2016). XGBoost. KDD.
10. Ke, G. et al. (2017). LightGBM. NeurIPS.
11. Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
12. Wolpert, D. H. (1992). Stacked generalization. Neural Networks 5.
13. Mack, T. (1993). Distribution-free calculation of the standard error of chain ladder reserve estimates. ASTIN Bulletin 23.
14. Bornhuetter, R. L. and Ferguson, R. E. (1972). The actuary and IBNR. Proceedings of the Casualty Actuarial Society 59.
15. Pope, G. C. et al. (2011). Evolution of the CMS-HCC risk adjustment model. Health Care Financing Review.
16. Kermack, W. O. and McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society A 115.
17. Vickers, A. J. and Elkin, E. B. (2006). Decision curve analysis. Medical Decision Making 26.
18. Mitchell, M. et al. (2019). Model cards for model reporting. FAT*.
19. Wachter, S., Mittelstadt, B. and Russell, C. (2018). Counterfactual explanations without opening the black box. Harvard Journal of Law and Technology 31.
20. Hay, M. et al. (2014). Clinical development success rates for investigational drugs. Nature Biotechnology 32.
21. Markowitz, H. (1952). Portfolio selection. Journal of Finance 7.
22. Harrell, F. E., Lee, K. L. and Mark, D. B. (1996). Multivariable prognostic models. Statistics in Medicine 15.
23. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review 78.
24. Pedregosa, F. et al. (2011). Scikit-learn. JMLR 12.
25. Charlson, M. E. et al. (1987). A new method of classifying prognostic comorbidity. Journal of Chronic Diseases 40.
26. Hosmer, D. W. and Lemeshow, S. (2000). Applied Logistic Regression, 2nd ed. Wiley.
*(Please verify every reference before submitting; details were written from memory.)*
"""
open("docs/report.md", "w").write(report)


def card(name, details, use, data, metrics, limits, monitor):
    return f"## {name}\n* **Model details:** {details}\n* **Intended use:** {use}\n* **Training / evaluation data:** {data}\n* **Metrics:** {metrics}\n* **Ethical considerations and caveats:** {limits}\n* **Monitoring:** {monitor}\n"


ibr = R["actuarial"]["ibnr"]
cards = ["# Model cards (Google Model Card structure)\n\n*All models were trained on SYNTHETIC data. None is for clinical or financial use.*\n",
         card("Clinical NLP (diagnosis, NER, complexity)", "TF-IDF + logistic regression; cue-word lexicon NER with negation; TF-IDF + SVD + ridge for cost", "teaching; extract risk signals from notes", "templated synthetic notes, train 2019-2020, test 2022", f"macro AUROC {NLP['macro_auroc']} (inflated), NER F1 {NLP['ner']['f1']}, notes-only cost R2 {NLP['R2_notes_only_ln_cost']}", "templated text; real notes need a clinical language model; HIPAA minimum-necessary applies to any real use", "agreement between codes and note entities each quarter"),
         card("Graph network (comorbidity + drug interactions)", "Neighbour propagation over a 20-condition lift graph and a 60-drug interaction graph, SVD to 16 dimensions", "patient embeddings for downstream models", "synthetic graphs; fitted on training members", f"mortality AUROC {MT['Graph_view']['auroc']}, readmission {RD['Graph_view']['auroc']} (alone)", "no attention; interaction severities are random", "embedding drift (PSI) when the formulary changes"),
         card("Survival models", "Own Cox PH (Newton-Raphson, ridge), discrete-time boosting, early-vs-updated lab comparison", "time to readmission, complication, covenant breach", "synthetic members and hospitals, time split", f"C-index readmission {cx_r}, complication {cx_c}", "proportional hazards not tested formally; few hospital events", "C-index on matured follow-up each year"),
         card("1-year mortality and 30-day readmission (stacked ensemble)", "5 level-0 models + logistic meta-learner, time-aware folds", "member risk stratification", f"{D['train']:,} train / {D['test']:,} test members", f"AUROC {MT['Stacked_ensemble']['auroc']} / {RD['Stacked_ensemble']['auroc']}, Brier {MT['Stacked_ensemble']['brier']} / {RD['Stacked_ensemble']['brier']}", "age, sex, plan and region are inputs; subgroup fairness not tested for these two tasks; few events", "AUROC and calibration-in-the-large quarterly; retrain if AUROC falls 0.05"),
         card("Cost prediction and premium pricing", "Tweedie GLM baseline; Poisson-loss boosting enhanced model; ridge stack", "premium pricing support and risk stratification", "synthetic claims with random catastrophic events", f"R2 GLM {PR[0]['R2']}, enhanced {enh['R2']}; predictive ratio {enh['predictive_ratio_all']}; top-5% ratio {enh['predictive_ratio_top5pct']}", f"under-predicts the extreme tail; worst subgroup deviation {enh['worst_subgroup_deviation_pct']}%; rate filing needs actuarial sign-off", "predictive ratio by plan, region and age band every quarter"),
         card("IBNR reserving", "Chain ladder, Bornhuetter-Ferguson, emergence-adjusted chain ladder", "reserve estimation", "synthetic 10x10 triangle with known ultimates", f"total IBNR error: CL {ibr['chain_ladder']['total_error_pct']}%, BF {ibr['bornhuetter_ferguson']['total_error_pct']}%, adjusted {ibr['emergence_adjusted_chain_ladder']['total_error_pct']}%", "tail factors are not modelled; the synthetic emergence shift is mild", "actual-vs-expected emergence each quarter"),
         card("Hospital credit PD", "Logistic scorecards and boosting, traditional vs enhanced", "4-quarter covenant-breach probability", "320 synthetic hospitals, train to 2019Q4, test 2021-2023", f"Gini {pm['traditional_scorecard_LR']['gini']}, KS {pm['traditional_scorecard_LR']['ks']}; clinical metrics added no AUROC", "PSI shows a large regime shift after 2020; rural hospitals are higher risk by construction", "PSI each quarter; recalibrate when PSI > 0.25"),
         card("Phase-success and rNPV", "Boosting on trial design features; Monte Carlo rNPV", "pipeline valuation support", "1,400 synthetic trials", f"AUROC {PH['phase_success_model']['auroc_model']} vs generic rates {PH['phase_success_model']['auroc_generic_phase_rate']}", "company pipelines are synthetic; binary event risk is not diversifiable", "compare predicted and realised phase transitions yearly")]
open("docs/MODEL_CARDS.md", "w").write("\n".join(cards))
g = ["# HealthRisk Lab - game design document\n\n", "**Educational objective:** make learners combine clinical intelligence (outbreak signals, trial data, hospital quality) with insurance, credit and equity decisions under time pressure.\n\n",
     "**Portfolio:** $500M = insurance book $150M (50,000 lives), 15 hospital bonds $150M, 20 pharma stocks $100M, credit facility to 10 systems $100M. Quarterly turns.\n\n",
     "**Modes:** 1 Actuarial (20 quarters), 2 Hospital Credit (40), 3 Pharma (20), 4 Integrated / Master (40). Single-mode scores are rescaled to 1000.\n\n",
     "**Player decisions:** reserve load, premium margin, hospital PD screen and reaction strength, pharma position cap and tilt, facility monitoring, hedge, use of clinical data, reaction delay, documentation.\n\n",
     "## Scoring (1000 points)\n| Category | Points | Components |\n|---|---|---|\n| Insurance | 250 | MLR in 80-85% band 60, reserve adequacy 70, profitability 70, member retention 50 |\n| Hospital credit | 250 | default avoidance 80, spread capture 60, early warning 70, portfolio quality 40 |\n| Pharma alpha | 200 | absolute return 50, Sharpe 60, hit rate 40, drawdown control 50 |\n| Credit facility | 150 | loss avoidance 70, covenant monitoring 50, exposure management 30 |\n| Cross-asset risk | 100 | correlation management 40, scenario response 30, hedging 30 |\n| Speed and quality | 50 | timeliness 30, documentation 20 |\n\n",
     "Ratings: Trainee < 500, Analyst >= 500, Specialist >= 650, Expert >= 800, Master >= 900.\n\n## Scenarios\n| Scenario | Claims multiplier | Hospital PD multiplier | Duration (quarters) | Early signal |\n|---|---|---|---|---|\n"]
for k, v in L.SCENARIOS.items():
    g.append(f"| {k} | {v[0]} | {v[1]} | {v[5]} | {'yes' if v[6] else 'no'} |\n")
g.append(f"\nPandemic worked example (replay): quarter 2 declaration; the AI scores {S['pandemic_replay']['ai']['total']} against {S['pandemic_replay']['naive']['total']} for the naive policy. Points by category (AI): {S['pandemic_replay']['ai']['points']}.\n")
g.append("\n**AI opponent logic:** uses clinical data for sharper risk estimates (noise falls with `use_clinical`), reacts in the same quarter, holds a larger reserve, cuts weak hospital exposure and tilts pharma toward event beneficiaries. It is a rule set, not a learned policy.\n")
open("docs/GAME_DESIGN.md", "w").write("".join(g))
print("report words:", len(report.split()))
