import json
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data.processing import synth, ops
from data.features import clinical_features as CF
from data.acquisition import acquire
from models.clinical_nlp import clinical_nlp as N
from models.graph_network import graph_network as G
from models.survival import survival as S
from models.tabular import tabular as T
from models.ensemble import ensemble as E
from financial.insurance import insurance as INS
from financial.credit_risk import credit_risk as CR
from financial.pharma import pharma as PH
from explainability import explain as X
from simulation import lab as L, epi

_c = {}


def world():
    if not _c:
        w = synth.build_world()
        _c["w"], _c["P"] = w, CF.add_features(w["patients"])
    return _c["w"], _c["P"]


def test_world_sizes_and_split_embargo():
    w, P = world()
    assert len(P) >= 15000 and w["hospitals"].hospital.nunique() >= 300 and len(w["trials"]) >= 1000
    assert set(P[P.split == "train"].index_year) == {2019, 2020} and set(P[P.split == "test"].index_year) == {2022}
    assert set(P[P.split == "embargo"].index_year) == {2021}


def test_consistency_rule_bph_only_in_males():
    _, P = world()
    assert ((P.cond_bph == 1) & (P.sex == 1)).sum() == 0 and P.member_id.is_unique


def test_icd10_hierarchy():
    h = CF.icd10_hierarchy("E11.65")
    assert h["chapter"] == "Endocrine" and h["category"] == "E11" and h["block"] == "E10-E14"
    assert CF.icd10_hierarchy("XYZ") is None


def test_hcc_score_matches_worked_example():
    flags = np.zeros(len(synth.COND_KEYS))
    flags[synth.COND_KEYS.index("diabetes")] = 1
    flags[synth.COND_KEYS.index("ckd")] = 1
    assert abs((CF.hcc_risk_score(flags, 60, 0) - CF.hcc_risk_score(np.zeros(len(flags)), 60, 0)) - 0.371) < 1e-9


def test_slope_and_nan_handling():
    s = CF.slope([[7.2, 7.5, 7.8, 8.1], [np.nan, 5, np.nan, np.nan]])
    assert abs(s[0] - 0.30) < 1e-9 and np.isnan(s[1])


def test_catalog_covers_features_and_no_label_columns():
    assert set(CF.STRUCT) <= set(CF.catalog().feature)
    for bad in ["cost_12m", "mortality_1y", "readmit_30d", "e_readmit", "t_readmit_days"]:
        assert bad not in CF.STRUCT


def test_data_quality_protocol_runs_all_six_stages():
    r = ops.validate_patients(world()[1])
    assert r["all_critical_checks_passed"] and len(r) >= 7 and r["6_duplicate_member_ids"] == 0


def test_experiment_log_and_config():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.jsonl")
        ops.log_run("t", {"a": 1}, {"m": 2}, path=p)
        assert json.loads(open(p).readline())["metrics"]["m"] == 2
    assert "split" in ops.load_config()


def test_acquire_functions_fail_safe_without_internet():
    out = acquire.fetch_who_gho("NOT_A_REAL_INDICATOR")
    assert out is None or isinstance(out, pd.DataFrame)


def test_negation_and_entity_extraction():
    found, neg = N.extract_entities("62 year old female with reduced egfr creatinine no chest pain denies shortness of breath")
    assert "ckd" in found and "cad" not in found and "cad" in neg and "asthma" not in found


def test_diagnosis_classifier_and_complexity_model():
    _, P = world()
    tr, te = P[P.split == "train"].head(3000), P[P.split == "test"].head(1500)
    assert N.diagnosis_classifier(tr, te)[0]["macro_auroc"] > 0.7
    c, fit = N.complexity_model(tr, te)
    assert -0.5 < c["R2_notes_only_ln_cost"] < 1
    h = P.head(2000)
    sc = N.complexity_scores(h, fit, (h.split == "train").values)
    assert len(sc) == 2000 and np.isfinite(sc).all()
    assert 0 < N.ner_f1(te, n=200)["f1"] <= 1 and N.severity_cue_count("poorly controlled and lives alone") == 2


def test_graph_embeddings_and_polypharmacy():
    w, P = world()
    h = P.head(3000)
    emb = G.patient_embeddings(h, w["graph"], dim=8, train_mask=(h.split == "train").values)
    assert emb.shape == (3000, 8)
    lift = G.comorbidity_graph(h)
    assert np.allclose(lift, lift.T) and np.allclose(np.diag(lift), 0)
    assert G.polypharmacy_risk(w["graph"], np.array([0])) == 0.0 and G.polypharmacy_risk(w["graph"], np.arange(8)) >= 0


def test_cox_recovers_sign_of_effect():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 2))
    t = rng.exponential(1 / np.exp(0.8 * x[:, 0]))
    cens = rng.exponential(2.0, 1500)
    time, ev = np.minimum(t, cens), (t <= cens).astype(int)
    m = S.CoxPH().fit(x, time, ev)
    assert m.beta[0] > 0.4 and abs(m.beta[1]) < 0.2 and S.concordance_index(time, ev, m.risk(x)) > 0.6


def test_concordance_and_time_dependent_auroc():
    t, e = [1, 2, 3, 4], [1, 1, 1, 0]
    assert S.concordance_index(t, e, [4, 3, 2, 1]) == 1.0 and S.concordance_index(t, e, [1, 2, 3, 4]) == 0.0
    time = np.random.default_rng(1).uniform(0, 10, 300)
    assert all(v > 0.99 for v in S.time_dependent_auroc(time, np.ones(300, int), -time, [3, 6]).values())


def test_discrete_hazard_survival_is_monotone():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 3))
    t = rng.exponential(1 / np.exp(0.7 * X[:, 0])) * 4
    ev = (t < 10).astype(int)
    m = S.DiscreteHazard(n_periods=6, period_len=2.0).fit(X, np.minimum(t, 10), ev)
    s = m.survival(X[:50])
    assert (np.diff(s, axis=1) <= 1e-9).all() and ((s >= 0) & (s <= 1)).all()
    assert S.evaluate(m.risk(X), np.minimum(t, 10), ev, [2, 4])["c_index"] > 0.55


def test_tabular_helpers():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    y = (X.a + rng.normal(size=500) > 0).astype(int)
    best, _ = T.tune_classifier(X, y, n_iter=2)
    p = best.predict_proba(X)[:, 1]
    assert T.class_metrics(y, p)["auroc"] > 0.7
    lo, hi = T.bootstrap_ci(y, p, n=50)
    assert lo < hi
    cost, _ = T.tune_cost_model(X, np.abs(X.a) * 100, n_iter=2)
    assert len(cost.predict(X)) == 500


def test_time_aware_stacking_runs():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(1200, 3))
    y = (X[:, 0] + rng.normal(size=1200) > 0).astype(int)
    oof, ok = E.time_aware_oof({"lr": lambda: LogisticRegression(), "lr2": lambda: LogisticRegression(C=0.1)}, {"lr": X, "lr2": X[:, :2]}, y)
    assert oof.shape == (1200, 2) and ok.sum() > 800 and not ok[:100].any()
    p = E.predict_meta(E.fit_meta(oof, y, ok), np.column_stack([LogisticRegression().fit(X, y).predict_proba(X)[:, 1]] * 2))
    assert ((p >= 0) & (p <= 1)).all()


def test_chain_ladder_exact_on_constant_pattern():
    pattern = np.array([0.5, 0.25, 0.15, 0.1])
    ult = np.array([100.0, 110.0, 120.0, 130.0])
    cum = np.cumsum(ult[:, None] * pattern, axis=1)
    obs = np.full((4, 4), np.nan)
    for i in range(4):
        obs[i, :4 - i] = cum[i, :4 - i]
    u, lat = INS.chain_ladder(obs)
    assert np.allclose(u, ult, rtol=1e-9)
    assert np.allclose(INS.bornhuetter_ferguson(obs, ult / 0.84)[0], ult, rtol=1e-6)
    assert abs(INS.ibnr_accuracy(ult, u, lat, np.arange(4), last_n=3)["total_error_pct"]) < 1e-6


def test_emergence_adjusted_handles_real_triangle():
    obs, cum, ult, prem, years = synth.make_claims_triangle()
    for fn in (INS.chain_ladder, INS.emergence_adjusted):
        u, lat = fn(obs)
        assert np.all(u >= lat) and INS.ibnr_accuracy(ult, u, lat, years)["ultimate_MAPE_pct"] < 10


def test_glm_baseline_and_pricing_report():
    _, P = world()
    tr, te = P[P.split == "train"].head(4000), P[P.split == "test"].head(2000)
    pred, _ = INS.glm_baseline(tr, te)
    r = INS.pricing_report(te.reset_index(drop=True), pred, "glm")
    assert 0.5 < r["predictive_ratio_all"] < 1.5 and "subgroup_ratios" in r
    strat, tab = INS.risk_stratification(te, pred)
    assert 0 <= strat["auroc_top5pct_cost"] <= 1 and len(tab) == 10 and INS.loss_ratio_by_decile(te, pred)["spread"] >= 0


def test_rating_bands_and_split_embargo():
    assert CR.implied_rating(0.002) == "A" and CR.implied_rating(0.03) == "BBB-" and CR.implied_rating(0.5) == "CCC"
    tr, te = CR.split(world()[0]["hospitals"])
    assert tr.q.max() + 4 < te.q.min()


def test_hospital_models_early_warning_and_spread():
    H = world()[0]["hospitals"]
    res, Hc, te, enh = CR.run_pd_models(H)
    assert 0.6 < res["traditional_scorecard_LR"]["auroc"] < 1 and len(res["enhanced_minus_traditional_auroc_ci95"]) == 2
    assert CR.early_warning_backtest(H)["breached_hospitals"] > 50 and 0 <= res["traditional_scorecard_LR"]["ks"] <= 1
    x = np.random.default_rng(0).normal(size=2000)
    assert CR.psi(x, x) < 1e-6 and CR.psi(x, x + 1.5) > 0.25
    assert CR.bond_spread_model(te)[0]["spread_per_doubling_of_PD_bps"] > 0
    trh, _ = CR.split(Hc)
    ex = CR.worked_example(CR.lr().fit(trh[CR.FIN], trh.y_breach_4q), CR.lr().fit(trh[enh], trh.y_breach_4q), te.head(1), CR.FIN, enh)
    assert 0 < ex["PD_traditional"] < 1


def test_phase_model_beats_generic_rate():
    res, _ = PH.phase_model(world()[0]["trials"])
    assert res["auroc_model"] > res["auroc_generic_phase_rate"]


def test_rnpv_and_patent_cliff():
    v, det = PH.rnpv_pipeline([("X", 3, 2.0, 2, 10)], None, n_sims=2000)
    assert v.mean() > 0 and det["X"]["prob_of_approval"] == round(0.58 * 0.85, 3)
    assert PH.rnpv_pipeline([("X", 1, 2.0, 2, 10)], None, n_sims=2000)[0].mean() < v.mean()
    pc = PH.patent_cliff(2.0, 5)
    assert pc["revenue_path_B"][4] == 2.0 and pc["revenue_path_B"][9] < 1.0 and pc["lost_in_3_years_after_expiry_B"] > 0
    assert len(PH.pipeline_report(None)) == 5


def test_portfolio_optimiser_and_backtest():
    cov = np.cov(np.random.default_rng(0).normal(size=(60, 10)).T)
    w = PH.mean_variance_weights(np.zeros(10), cov)
    assert abs(w.sum() - 1) < 1e-6 and w.max() <= 0.15 + 1e-6 and w.min() >= -1e-9
    rets, sig, _ = world()[0]["stocks"]
    p, b = PH.backtest(rets.iloc[:60], sig.iloc[:60], window=24)
    perf = PH.perf(p, b)
    assert {"sharpe", "information_ratio", "max_drawdown_pct", "hit_rate_months_beating_benchmark_pct"} <= set(perf) and perf["max_drawdown_pct"] <= 0
    res, m = PH.phase_model(world()[0]["trials"])
    assert PH.trial_signal(world()[0]["trials"].tail(1), m)["enrolment_signal"] in ("POSITIVE", "NEGATIVE")


def test_explanations_pdp_and_counterfactual():
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = world()[1].head(3000)
    X_, y = d[CF.STRUCT], d["mortality_1y"]
    m = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(X_, y)
    assert len(X.global_importance(m, X_.head(1000), y.head(1000), n_repeats=1)) == len(CF.STRUCT)
    pred = lambda df: m.predict_proba(df)[:, 1]
    assert len(X.local_explanation(pred, X_.iloc[[0]], X_)[1]) == 6
    g, a = X.pdp(m, X_.head(500), "age")
    assert len(g) == len(a)
    cf = X.counterfactual(pred, X_[X_.hba1c_last.notna()].iloc[[0]], target=0.0)
    assert cf["final"] <= cf["start"] + 1e-9 and "largest_contributions" in X.premium_breakdown(pred, X_.iloc[[0]], X_) and len(X.REGULATORY_MATRIX) >= 6


def test_twelve_scenarios_and_modes():
    assert len(L.SCENARIOS) >= 10 and set(L.MODE_QUARTERS) == {1, 2, 3, 4} and L.MODE_QUARTERS[4] == 40 and L.MODE_QUARTERS[1] == 20


def test_scores_in_range_and_ai_beats_naive_on_average():
    for m in (1, 2, 4):
        a = [L.run(m, L.NAIVE, s)["total"] for s in range(6)]
        b = [L.ai_opponent(m, s)["total"] for s in range(6)]
        assert 0 <= min(a + b) and max(a + b) <= 1000 and np.mean(b) > np.mean(a)


def test_category_points_respect_maximums_and_master_mode_has_all_six():
    r = L.run(4, L.AI, 3)
    assert set(r["category_points"]) == set(L.MAX_PTS)
    for c, v in r["category_points"].items():
        assert 0 <= v <= L.MAX_PTS[c] + 1e-9
    assert set(L.run(1, L.AI, 3)["category_points"]) == {"insurance", "speed"}


def test_full_10_year_run_replay_and_json():
    r = L.to_plain(L.run(4, L.AI, 7))
    assert r["quarters"] == 40 and len(r["log"]) == 40
    json.dumps(r)
    assert L.replay_pandemic(L.AI)["total"] > L.replay_pandemic(L.NAIVE)["total"]
    assert len(L.compare(4, L.NAIVE, 1)["biggest_gaps"]) == 2


def test_early_signal_events_exist():
    assert sum(v[6] for v in L.SCENARIOS.values()) >= 2
    assert all(1 <= e["q"] < 40 for e in L.schedule(40, np.random.default_rng(0)))


def test_seir_threshold_behaviour_and_claims():
    hi, lo = epi.seir(r0=2.5), epi.seir(r0=0.8)
    assert hi["attack_rate"] > 0.7 and lo["attack_rate"] < 0.05 and hi["peak_infected"] > lo["peak_infected"]
    assert epi.doubling_time(0.8) == float("inf") and 2 < epi.doubling_time(2.5) < 15
    c = epi.claims_impact()
    assert 0.02 * 50000 <= c["members_treated"] <= 0.05 * 50000 and c["incremental_claims_musd"] > 0
