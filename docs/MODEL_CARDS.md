# Model cards (Google Model Card structure)

*All models were trained on SYNTHETIC data. None is for clinical or financial use.*

## Clinical NLP (diagnosis, NER, complexity)
* **Model details:** TF-IDF + logistic regression; cue-word lexicon NER with negation; TF-IDF + SVD + ridge for cost
* **Intended use:** teaching; extract risk signals from notes
* **Training / evaluation data:** templated synthetic notes, train 2019-2020, test 2022
* **Metrics:** macro AUROC 1.0 (inflated), NER F1 0.82, notes-only cost R2 0.03
* **Ethical considerations and caveats:** templated text; real notes need a clinical language model; HIPAA minimum-necessary applies to any real use
* **Monitoring:** agreement between codes and note entities each quarter

## Graph network (comorbidity + drug interactions)
* **Model details:** Neighbour propagation over a 20-condition lift graph and a 60-drug interaction graph, SVD to 16 dimensions
* **Intended use:** patient embeddings for downstream models
* **Training / evaluation data:** synthetic graphs; fitted on training members
* **Metrics:** mortality AUROC 0.668, readmission 0.613 (alone)
* **Ethical considerations and caveats:** no attention; interaction severities are random
* **Monitoring:** embedding drift (PSI) when the formulary changes

## Survival models
* **Model details:** Own Cox PH (Newton-Raphson, ridge), discrete-time boosting, early-vs-updated lab comparison
* **Intended use:** time to readmission, complication, covenant breach
* **Training / evaluation data:** synthetic members and hospitals, time split
* **Metrics:** C-index readmission 0.658, complication 0.817
* **Ethical considerations and caveats:** proportional hazards not tested formally; few hospital events
* **Monitoring:** C-index on matured follow-up each year

## 1-year mortality and 30-day readmission (stacked ensemble)
* **Model details:** 5 level-0 models + logistic meta-learner, time-aware folds
* **Intended use:** member risk stratification
* **Training / evaluation data:** 8,080 train / 4,013 test members
* **Metrics:** AUROC 0.822 / 0.746, Brier 0.0233 / 0.1497
* **Ethical considerations and caveats:** age, sex, plan and region are inputs; subgroup fairness not tested for these two tasks; few events
* **Monitoring:** AUROC and calibration-in-the-large quarterly; retrain if AUROC falls 0.05

## Cost prediction and premium pricing
* **Model details:** Tweedie GLM baseline; Poisson-loss boosting enhanced model; ridge stack
* **Intended use:** premium pricing support and risk stratification
* **Training / evaluation data:** synthetic claims with random catastrophic events
* **Metrics:** R2 GLM -0.002, enhanced 0.14; predictive ratio 0.948; top-5% ratio 0.248
* **Ethical considerations and caveats:** under-predicts the extreme tail; worst subgroup deviation 12.1%; rate filing needs actuarial sign-off
* **Monitoring:** predictive ratio by plan, region and age band every quarter

## IBNR reserving
* **Model details:** Chain ladder, Bornhuetter-Ferguson, emergence-adjusted chain ladder
* **Intended use:** reserve estimation
* **Training / evaluation data:** synthetic 10x10 triangle with known ultimates
* **Metrics:** total IBNR error: CL 3.06%, BF 2.12%, adjusted 1.98%
* **Ethical considerations and caveats:** tail factors are not modelled; the synthetic emergence shift is mild
* **Monitoring:** actual-vs-expected emergence each quarter

## Hospital credit PD
* **Model details:** Logistic scorecards and boosting, traditional vs enhanced
* **Intended use:** 4-quarter covenant-breach probability
* **Training / evaluation data:** 320 synthetic hospitals, train to 2019Q4, test 2021-2023
* **Metrics:** Gini 0.544, KS 0.448; clinical metrics added no AUROC
* **Ethical considerations and caveats:** PSI shows a large regime shift after 2020; rural hospitals are higher risk by construction
* **Monitoring:** PSI each quarter; recalibrate when PSI > 0.25

## Phase-success and rNPV
* **Model details:** Boosting on trial design features; Monte Carlo rNPV
* **Intended use:** pipeline valuation support
* **Training / evaluation data:** 1,400 synthetic trials
* **Metrics:** AUROC 0.674 vs generic rates 0.616
* **Ethical considerations and caveats:** company pipelines are synthetic; binary event risk is not diversifiable
* **Monitoring:** compare predicted and realised phase transitions yearly
