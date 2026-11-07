# Feature engineering
Every feature has rationale, formula and validation in `docs/feature_catalog.csv`; a unit test checks that all model inputs are in the catalog and that no outcome column is an input.
| Group | Examples | Implementation | Validation |
|---|---|---|---|
| Demographic | age, sex, plan, region, tobacco | raw | range checks |
| Diagnosis | 20 condition flags, hcc_score, charlson | ICD-10 root per condition, additive weights | unit test vs brief example (0.371) |
| Labs | last value, 4-point slope, variability, count | `slope`, `lab_variability` | slope test (+0.30) |
| Medications | drug count, ATC diversity, major/moderate interactions, adherence | drug sets + 60-drug interaction graph | bounded by pair counts |
| Utilisation | prior admissions, ED, specialist, PCP, ln cost, cost trend | prior-12-month fields | non-negative checks |
| Fusion | clinical trajectory score, interaction burden | weighted z-scores | monotone in inputs |
| NLP / graph | note complexity (out-of-fold), severity cues, 16-d embedding | TF-IDF + ridge; comorbidity and DDI graph propagation | fit on training rows only |
| Hospital | margin, DSCR, days cash, readmission, HCAHPS, boarding, 4-quarter changes | quarterly panel | PSI train vs test |
Leakage rules: features use the index year only; outcome columns are never inputs; embeddings and the notes model are fit on training rows; hospital labels look 4 quarters ahead and the train/test gap is wider than that.
