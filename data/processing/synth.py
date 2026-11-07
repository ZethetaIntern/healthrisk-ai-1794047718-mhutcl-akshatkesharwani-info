"""Synthetic health-financial world (no HealthRisk data was provided and MIMIC-IV needs credentialed access).

Tables: patients (members), drug and comorbidity graphs, clinical notes, hospital panel, claims triangle, clinical trials,
stock returns.  Results on this data show the pipeline works, NOT how it performs on real patients or markets."""
import numpy as np
import pandas as pd

CONDITIONS = {  # key: (ICD-10 root, HCC weight, Charlson weight, base logit, age slope, severity slope, note cue words)
    "diabetes": ("E11", 0.302, 1, -1.2, 0.5, 0.4, "polyuria elevated glucose hba1c"),
    "hypertension": ("I10", 0.0, 0, -0.3, 0.7, 0.2, "elevated blood pressure"),
    "hyperlipidemia": ("E78", 0.0, 0, -0.6, 0.4, 0.1, "high ldl cholesterol"),
    "ckd": ("N18", 0.069, 2, -2.2, 0.8, 0.9, "reduced egfr creatinine"),
    "heart_failure": ("I50", 0.331, 1, -2.6, 0.9, 1.0, "dyspnea edema orthopnea"),
    "copd": ("J44", 0.335, 1, -2.4, 0.6, 0.8, "wheeze cough smoking"),
    "cad": ("I25", 0.140, 1, -1.8, 0.8, 0.6, "chest pain angina"),
    "cancer": ("C34", 1.024, 2, -3.3, 0.6, 1.0, "mass biopsy chemotherapy"),
    "depression": ("F32", 0.309, 0, -1.8, -0.2, 0.5, "low mood anhedonia"),
    "afib": ("I48", 0.268, 0, -2.5, 1.0, 0.5, "palpitations irregular rhythm"),
    "obesity": ("E66", 0.250, 0, -1.2, -0.2, 0.3, "elevated bmi"),
    "stroke": ("I63", 0.230, 1, -3.0, 0.9, 0.8, "weakness aphasia"),
    "dementia": ("G30", 0.346, 1, -3.2, 1.3, 0.8, "memory loss confusion"),
    "osteoporosis": ("M81", 0.0, 0, -2.0, 0.9, 0.2, "fracture bone density"),
    "gerd": ("K21", 0.0, 0, -1.0, 0.2, 0.0, "reflux heartburn"),
    "asthma": ("J45", 0.0, 0, -2.0, -0.4, 0.3, "wheeze shortness of breath"),
    "bph": ("N40", 0.0, 0, -2.0, 0.9, 0.0, "urinary frequency"),
    "hypothyroid": ("E03", 0.0, 0, -1.8, 0.4, 0.0, "fatigue cold intolerance"),
    "sleep_apnea": ("G47", 0.0, 0, -1.8, 0.2, 0.2, "snoring daytime sleepiness"),
    "back_pain": ("M54", 0.0, 0, -1.0, 0.0, 0.2, "lumbar pain"),
}
COND_KEYS = list(CONDITIONS)
PRIMARY_PRIORITY = ["cancer", "heart_failure", "stroke", "copd", "ckd", "diabetes", "cad", "hypertension"]
N_DRUG = 60
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Mid-Atlantic"]
PLANS = ["Medicare Advantage", "Commercial", "Medicaid"]
TRAIN_YEARS, EMBARGO_YEAR, TEST_YEAR = (2019, 2020), 2021, 2022


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_drug_graph(seed=1):
    rng = np.random.default_rng(seed)
    atc = rng.integers(0, 12, N_DRUG)                       # 12 therapeutic classes
    cyp = rng.integers(0, 5, N_DRUG)
    A = np.zeros((N_DRUG, N_DRUG))
    for i in range(N_DRUG):
        for j in range(i + 1, N_DRUG):
            p = 0.02 + 0.10 * (cyp[i] == cyp[j]) + 0.05 * (atc[i] == atc[j])
            if rng.random() < p:
                A[i, j] = A[j, i] = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])   # 1 minor, 2 moderate, 3 major
    drug_cond = rng.integers(0, len(COND_KEYS), N_DRUG)    # the condition each drug mainly treats
    return A, atc, cyp, drug_cond


def make_patients(n=16000, seed=2):
    rng = np.random.default_rng(seed)
    A, atc, cyp, drug_cond = make_drug_graph()
    year = rng.choice([2019, 2020, 2021, 2022], n)
    s = rng.normal(0, 1, n)                                 # hidden clinical severity
    f = rng.normal(0, 1, n)                                 # hidden frailty / social risk
    age = np.clip(rng.normal(64, 11, n), 40, 95)
    az = (age - 64) / 11
    sex = rng.integers(0, 2, n)
    region = rng.integers(0, len(REGIONS), n)
    plan = rng.choice(3, n, p=[0.5, 0.35, 0.15])
    tobacco = (rng.random(n) < 0.15 + 0.05 * (f > 0.5)).astype(int)
    P = pd.DataFrame({"member_id": np.arange(n), "index_year": year, "age": age, "sex": sex, "region": region, "plan": plan, "tobacco": tobacco})
    cond = np.zeros((n, len(COND_KEYS)), dtype=int)
    for k, name in enumerate(COND_KEYS):
        _, _, _, b, a_s, s_s, _ = CONDITIONS[name]
        cond[:, k] = rng.random(n) < sigmoid(b + 0.45 * a_s * az + 0.45 * s_s * s)
    cond[sex == 1, COND_KEYS.index("bph")] = 0                # BPH only occurs in males (a consistency rule)
    for k, name in enumerate(COND_KEYS):
        P["cond_" + name] = cond[:, k]
    hcc_w = np.array([CONDITIONS[c][1] for c in COND_KEYS])
    ch_w = np.array([CONDITIONS[c][2] for c in COND_KEYS])
    P["hcc_score"] = 0.25 + 0.01 * (age - 60) + 0.03 * sex + cond @ hcc_w
    P["hcc_count"] = (cond * (hcc_w > 0)).sum(axis=1)
    P["charlson"] = cond @ ch_w
    P["n_conditions"] = cond.sum(axis=1)
    dm, ckd = cond[:, 0] == 1, cond[:, 3] == 1               # labs: sick patients have more tests (missing not at random)
    for m in range(4):
        P[f"hba1c_{m}"] = np.where(dm & (rng.random(n) < 0.9), 6.8 + 0.5 * s + 0.12 * m * (0.5 + 0.5 * np.maximum(s, 0)) + rng.normal(0, 0.35, n), np.nan)
        P[f"egfr_{m}"] = np.where(ckd | (dm & (rng.random(n) < 0.5)), 78 - 9 * s - 1.2 * m * (0.5 + 0.5 * np.maximum(s, 0)) - 0.4 * (age - 64) + rng.normal(0, 4, n), np.nan)
    P["ldl"] = np.clip(100 + 12 * cond[:, 2] - 8 * s + rng.normal(0, 22, n), 40, 220)
    P["sbp"] = np.clip(122 + 8 * cond[:, 1] + 5 * s + 0.2 * (age - 64) + rng.normal(0, 11, n), 90, 200)
    P["hba1c_slope"] = (P["hba1c_3"] - P["hba1c_0"]) / 3
    P["egfr_slope"] = (P["egfr_3"] - P["egfr_0"]) / 3
    P["hba1c_last"], P["egfr_last"] = P["hba1c_3"], P["egfr_3"]
    n_med = np.clip(rng.poisson(0.8 + 0.9 * P["n_conditions"].values + 0.3 * np.maximum(f, 0)), 0, 14)
    drug_sets, ddi_major, ddi_mod = [], np.zeros(n), np.zeros(n)
    for i in range(n):
        own = [d for d in range(N_DRUG) if cond[i, drug_cond[d]] == 1]
        pool = own if len(own) >= 3 else list(range(N_DRUG))
        k = min(int(n_med[i]), len(pool))
        ds = rng.choice(pool, size=k, replace=False) if k else np.array([], int)
        drug_sets.append(ds)
        if k > 1:
            sub = A[np.ix_(ds, ds)]
            ddi_major[i], ddi_mod[i] = (sub == 3).sum() / 2, (sub == 2).sum() / 2
    P["med_count"], P["ddi_major"], P["ddi_moderate"] = n_med, ddi_major, ddi_mod
    P["class_diversity"] = [len(set(atc[d])) for d in drug_sets]
    P["polypharmacy_score"] = np.digitize(n_med, [3, 5, 8])
    P["adherence_mpr"] = np.clip(0.9 - 0.08 * f - 0.03 * (n_med > 6) + rng.normal(0, 0.1, n), 0.2, 1.0)
    base = 0.08 * np.exp(0.45 * s + 0.2 * P["n_conditions"].values * 0.4 + 0.2 * f)
    P["prior_hosp"] = rng.poisson(base)
    P["prior_ed"] = rng.poisson(0.2 * np.exp(0.3 * s + 0.3 * f))
    P["prior_specialist"] = rng.poisson(1 + 0.8 * P["n_conditions"].values * 0.5)
    P["prior_pcp"] = rng.poisson(3 + 0.5 * P["n_conditions"].values * 0.3)
    trend = 1.04 ** (year - 2019)
    P["prior_cost"] = np.exp(rng.normal(7.4 + 0.55 * s + 0.15 * P["n_conditions"].values * 0.5 + 0.5 * (plan == 0) + 0.2 * f, 0.9)) * trend
    P["prior_cost_trend"] = np.clip(0.04 + 0.12 * s + rng.normal(0, 0.1, n), -0.3, 0.8)
    comp_risk = sigmoid(-2.0 + 0.9 * s + 0.35 * zs(P["ddi_major"]) - 0.4 * zs(P["adherence_mpr"]) + 0.3 * (cond[:, 3] + cond[:, 4]))
    P["admitted"] = (rng.random(n) < sigmoid(-1.0 + 0.9 * s + 0.4 * f + 0.2 * P["prior_hosp"].values)).astype(int)
    p_re = sigmoid(-2.5 + 1.3 * s + 0.25 * zs(P["ddi_major"]) + 0.3 * cond[:, 3] + 0.3 * f - 0.3 * zs(P["adherence_mpr"]))
    p_mort = sigmoid(-5.0 + 1.6 * s + 0.045 * (age - 64) + 0.5 * cond[:, 7] + 0.4 * cond[:, 4] + 0.3 * f)
    P["readmit_30d"] = (rng.random(n) < p_re).astype(int) * P["admitted"]
    P["mortality_1y"] = (rng.random(n) < p_mort).astype(int)
    tail = rng.random(n) < 0.03 + 0.10 * comp_risk
    gam = np.exp(rng.normal(8.3 + 0.55 * s + 0.2 * f + 0.2 * P["n_conditions"].values * 0.4, 0.7)) * trend
    P["cost_12m"] = np.where(rng.random(n) < 0.93, gam, 0.0) + tail * np.exp(rng.normal(10.3, 0.7, n))
    rate_re = 1 / 700 * np.exp(0.7 * s + 0.2 * zs(P["ddi_major"]) + 0.25 * f)
    t_re = rng.exponential(1 / rate_re)
    c_re = rng.uniform(150, 365, n)
    P["t_readmit_days"], P["e_readmit"] = np.minimum(t_re, c_re), (t_re <= c_re).astype(int)
    rate_c = 1 / 80 * np.exp(0.9 * s + 0.4 * zs(np.nan_to_num(P["hba1c_slope"], nan=0)) - 0.3 * zs(P["adherence_mpr"]))
    t_c = rng.weibull(1.4, n) / rate_c
    c_c = np.minimum(60, rng.uniform(24, 90, n))
    P["t_complication_m"], P["e_complication"] = np.minimum(t_c, c_c), (t_c <= c_c).astype(int)
    mild = ["well controlled", "diet managed", "independent in daily activities", "good adherence"]
    sev = ["poorly controlled", "brittle with multiple complications", "limited functional status", "lives alone", "missed appointments", "frequent emergency visits"]
    notes, primary = [], []
    for i in range(n):
        have = [c for c in COND_KEYS if P.at[i, "cond_" + c] == 1]
        prim = next((c for c in PRIMARY_PRIORITY if c in have), have[0] if have else "back_pain")
        primary.append(prim)
        sev_prob = sigmoid(1.8 * s[i] + 0.5 * f[i])
        cues = [CONDITIONS[c][6].split() for c in have[:5]]
        words = [w for cue in cues for w in rng.choice(cue, size=min(2, len(cue)), replace=False)]
        phr = [rng.choice(sev) if rng.random() < sev_prob else rng.choice(mild) for _ in range(2)]
        neg = "no chest pain no fever" if rng.random() < 0.5 else "denies shortness of breath"
        notes.append(f"{int(age[i])} year old {'female' if sex[i] else 'male'} presents with {' '.join(words)} {prim.replace('_', ' ')} {' '.join(phr)} {neg}")
    P["note"], P["primary_dx"] = notes, primary
    P["split"] = np.where(np.isin(year, TRAIN_YEARS), "train", np.where(year == TEST_YEAR, "test", "embargo"))
    graph = {"ddi": A, "atc": atc, "cyp": cyp, "drug_cond": drug_cond, "drug_sets": drug_sets}
    return P, graph, pd.DataFrame({"member_id": np.arange(n), "s": s, "f": f, "comp_risk": comp_risk, "p_mortality_true": p_mort, "p_readmit_true": p_re})


def make_hospitals(n_hosp=320, quarters=40, seed=3):
    """Quarterly panel 2015Q1-2024Q4. Clinical quality deteriorates ~2 quarters before financial ratios."""
    rng = np.random.default_rng(seed)
    q = np.arange(quarters)
    shock = np.where((q >= 20) & (q <= 24), 1.0, 0.0)             # 2020 shock
    rural = rng.random(n_hosp) < 0.3
    medicaid = np.clip(rng.normal(0.2, 0.08, n_hosp) + 0.1 * rural, 0.03, 0.6)
    L = np.zeros((n_hosp, quarters))                                 # hidden distress, random walk
    L[:, 0] = rng.normal(0, 1, n_hosp) + 0.5 * rural
    for t in range(1, quarters):
        L[:, t] = 0.95 * L[:, t - 1] + rng.normal(0, 0.28, n_hosp) + 0.5 * shock[t] * (0.5 + rural)
    rows = []
    for h in range(n_hosp):
        lead = np.roll(L[h], -2)
        lead[-2:] = L[h, -2:]
        for t in range(quarters):
            clin, fin = lead[t], L[h, t]
            rows.append((h, t, 2015 + t // 4, rural[h], medicaid[h],
                         0.03 - 0.025 * fin + rng.normal(0, 0.012),
                         np.clip(2.4 - 0.5 * fin + rng.normal(0, 0.3), 0.3, 5),
                         np.clip(110 - 22 * fin + rng.normal(0, 12), 5, 300),
                         np.clip(0.40 + 0.04 * fin + rng.normal(0, 0.04), 0.1, 0.9),
                         np.clip(1.5 - 0.03 * fin + rng.normal(0, 0.05), 1.0, 2.2),
                         np.clip(0.155 + 0.012 * clin + rng.normal(0, 0.006), 0.08, 0.28),
                         np.clip(1.0 + 0.15 * clin + rng.normal(0, 0.08), 0.3, 2.5),
                         np.clip(3.6 - 0.35 * clin + rng.normal(0, 0.3), 1, 5),
                         np.clip(5 + 1.4 * clin + rng.normal(0, 0.8), 0.5, 14),
                         0.0 - 0.02 * clin + rng.normal(0, 0.012)))
    H = pd.DataFrame(rows, columns=["hospital", "q", "year", "rural", "medicaid_share", "op_margin", "dscr", "days_cash", "debt_cap", "cmi",
                                    "readmit_rate", "hai_sir", "hcahps", "ed_boarding", "volume_growth"])
    H["rural"] = H["rural"].astype(int)
    H["distress"] = np.ravel(L)
    hz = sigmoid(-5.6 + 1.25 * H["distress"].values + 1.2 * H["medicaid_share"].values + 0.4 * H["rural"].values)
    H["breach_this_q"] = (rng.random(len(H)) < hz).astype(int)
    first = H[H.breach_this_q == 1].groupby("hospital").q.min()
    H["breach_q"] = H["hospital"].map(first).fillna(-1)
    H = H[(H.breach_q == -1) | (H.q <= H.breach_q)].copy()                  # hospital leaves after its first breach
    H["y_breach_4q"] = ((H.breach_q > H.q) & (H.breach_q <= H.q + 4)).astype(int)
    H["rating_downgrade_q"] = H["breach_q"] - rng.integers(0, 3, len(H))
    H["spread_bps"] = 80 + 900 * sigmoid(-5.6 + 1.25 * H["distress"] + 1.2 * H["medicaid_share"] + 0.4 * H["rural"]) * 6 + rng.normal(0, 25, len(H))
    return H.reset_index(drop=True)


def make_claims_triangle(seed=4):
    """Annual accident years 2015-2024, 10 development years. Known true ultimate allows IBNR accuracy tests."""
    rng = np.random.default_rng(seed)
    years = np.arange(2015, 2025)
    pattern = np.array([0.52, 0.22, 0.10, 0.06, 0.04, 0.02, 0.015, 0.01, 0.01, 0.005])
    ult = 400 * (1.05 ** (years - 2015)) * np.exp(rng.normal(0, 0.03, 10))
    ult[years == 2020] *= 0.97
    inc = np.zeros((10, 10))
    for i in range(10):
        pat = pattern.copy()
        if years[i] in (2020, 2021):
            pat = np.array([0.46, 0.27, 0.11, 0.06, 0.04, 0.02, 0.015, 0.01, 0.01, 0.005])   # slower emergence than history
        inc[i] = ult[i] * pat * np.exp(rng.normal(0, 0.02, 10))
    cum = inc.cumsum(axis=1)
    obs = np.full((10, 10), np.nan)
    for i in range(10):
        obs[i, : 10 - i] = cum[i, : 10 - i]
    return obs, cum, ult, ult / 0.84, years


def make_trials(n=1400, seed=5):
    rng = np.random.default_rng(seed)
    phase = rng.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25])
    area = rng.integers(0, 8, n)
    start = rng.integers(2005, 2021, n)
    enr = np.clip(rng.normal(1.0, 0.25, n), 0.3, 1.8)
    drop = np.clip(rng.normal(0.12, 0.05, n), 0.01, 0.4)
    sites = rng.integers(5, 150, n)
    sponsor = rng.normal(0, 1, n)
    comp = rng.integers(0, 6, n)
    novel = rng.random(n) < 0.5
    bio = rng.random(n) < 0.35
    effect = rng.normal(0, 1, n)
    base = np.array([0.63, 0.31, 0.58])[phase - 1]
    lg = np.log(base / (1 - base)) + 0.9 * (enr - 1) * 3 - 5 * (drop - 0.12) + 0.35 * sponsor - 0.15 * comp + 0.5 * bio - 0.3 * novel + 0.8 * effect - 0.25 * (area == 0)
    y = rng.random(n) < sigmoid(lg)
    return pd.DataFrame({"trial": np.arange(n), "phase": phase, "area": area, "start_year": start, "enrol_velocity": enr, "dropout": drop,
                         "n_sites": sites, "sponsor_exp": sponsor, "competitors": comp, "novel_mechanism": novel.astype(int),
                         "biomarker": bio.astype(int), "success": y.astype(int)})


def make_stocks(n_stocks=20, months=120, seed=6):
    """Monthly returns. A small genuine alpha is tied to a 'clinical signal' (assumption: IC about 0.06)."""
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.007, 0.04, months)
    beta = rng.uniform(0.6, 1.4, n_stocks)
    signal = rng.normal(0, 1, (months, n_stocks))
    ret = beta * mkt[:, None] + 0.004 * signal + rng.normal(0, 0.06, (months, n_stocks))
    jumps = rng.random((months, n_stocks)) < 0.02
    ret += jumps * rng.choice([-0.35, 0.45], (months, n_stocks), p=[0.6, 0.4])
    dates = pd.date_range("2015-01-31", periods=months, freq="ME")
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(ret, index=dates, columns=cols), pd.DataFrame(signal, index=dates, columns=cols), pd.Series(mkt, index=dates)


def build_world():
    P, graph, truth = make_patients()
    return {"patients": P, "graph": graph, "truth": truth, "hospitals": make_hospitals(), "triangle": make_claims_triangle(),
            "trials": make_trials(), "stocks": make_stocks()}
