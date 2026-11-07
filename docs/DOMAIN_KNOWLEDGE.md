# Healthcare domain knowledge used in this project
**Clinical data** are longitudinal, sparse and coded: ICD-10 diagnoses (E11.65 = Endocrine, block E10-E14, category E11), CPT procedures, NDC/ATC drugs, LOINC labs. Codes are assigned by people and are sometimes wrong. Labs are ordered more for sicker patients, so missing values carry information (missing not at random).
**Risk adjustment:** CMS-HCC style score = demographic base + condition weights. Here: 0.25 + 0.01 x (age-60) + 0.03 x sex + sum of weights (diabetes 0.302 + CKD 0.069 = 0.371 in the brief's example, checked by a unit test). The Charlson index sums comorbidity weights and predicts mortality.
**Disease progression:** HbA1c rising and eGFR falling signal worsening control and kidney function; the 4-point slope is the trajectory feature. Updated labs raise the complication C-index.
**Epidemiology:** SEIR (S, E, I, R) with R0 = beta / gamma. In `simulation/epi.py`, R0 = 2.5 gives an attack rate near 89% and a doubling time near 7 days; R0 < 1 dies out.
**Drug development:** phase success about 63% / 31% / 58% and 85% at approval in the brief. Drug-drug interactions are a graph (minor, moderate, major); polypharmacy multiplies risk.
**Hospital quality:** readmission rate, infection ratio, patient-experience stars, case-mix index and ED boarding describe quality and capacity; the brief says they lead financial ratios. In this synthetic data they added no predictive value (see report).
These are textbook-level relationships, not clinical advice.
