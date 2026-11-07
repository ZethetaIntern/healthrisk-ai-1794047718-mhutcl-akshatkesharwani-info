"""Pharmaceutical analytics: phase-success model, rNPV with Monte Carlo, patent cliff, trial signals, mean-variance portfolio with signal alpha."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATS = ["phase", "area", "enrol_velocity", "dropout", "n_sites", "sponsor_exp", "competitors", "novel_mechanism", "biomarker"]
GENERIC_POS = {1: 0.63, 2: 0.31, 3: 0.58}                       # phase transition rates quoted in the brief


def phase_model(trials):
    tr, te = trials[trials.start_year <= 2012], trials[trials.start_year >= 2016]
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=30, random_state=0).fit(tr[FEATS], tr["success"])
    return {"auroc_model": round(float(roc_auc_score(te.success, m.predict_proba(te[FEATS])[:, 1])), 3),
            "auroc_generic_phase_rate": round(float(roc_auc_score(te.success, te["phase"].map(GENERIC_POS).values)), 3), "n_train": len(tr), "n_test": len(te)}, m


COMPANIES = {  # 5 representative synthetic companies: market cap ($B), cash ($B), candidates (name, phase, peak sales $B, years to launch, patent years after launch)
    "Alpha Oncology": (5.2, 0.9, [("A1", 3, 2.5, 2, 11), ("A2", 2, 1.2, 4, 12), ("A3", 1, 0.8, 6, 12)]),
    "Beta Immunology": (4.5, 0.8, [("B1", 3, 2.0, 2, 9), ("B2", 2, 0.6, 4, 12), ("B3", 1, 0.3, 6, 13)]),
    "Gamma Rare Disease": (2.0, 0.4, [("C1", 2, 1.5, 3, 12), ("C2", 1, 0.5, 5, 13)]),
    "Delta CNS": (7.5, 1.6, [("D1", 3, 3.0, 1, 7), ("D2", 3, 1.0, 2, 10), ("D3", 2, 0.9, 4, 11), ("D4", 1, 0.5, 6, 12)]),
    "Epsilon Vaccines": (3.0, 1.2, [("E1", 2, 1.8, 3, 14), ("E2", 3, 0.7, 2, 12)]),
}
TRIAL_PROFILE = dict(area=0, enrol_velocity=1.05, dropout=0.10, n_sites=60, sponsor_exp=0.3, competitors=2, novel_mechanism=0, biomarker=1)


def rnpv_pipeline(cands, model=None, r=0.10, n_sims=5000, seed=0, margin=0.35, erosion=0.80):
    """Risk-adjusted NPV = sum over candidates of P(success through launch) x PV(cash flows); Monte Carlo over success draws and peak sales."""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_sims)
    detail = {}
    for name, phase, peak, yrs, pat in cands:
        pos = 1.0
        for ph in range(phase, 4):
            pos *= float(model.predict_proba(pd.DataFrame([{**TRIAL_PROFILE, "phase": ph}])[FEATS])[:, 1][0]) if model is not None else GENERIC_POS[ph]
        pos *= 0.85                                              # regulatory review success
        success = rng.random(n_sims) < pos
        sales = peak * np.exp(rng.normal(-0.5 * 0.3 ** 2, 0.3, n_sims))
        t = np.arange(0, 25)
        shape = np.clip((t - yrs) / 4, 0, 1) * np.where(t > yrs + pat, 1 - erosion, 1.0)
        pv = (shape * margin / (1 + r) ** t).sum()
        total += success * sales * pv
        detail[name] = {"phase": phase, "prob_of_approval": round(pos, 3), "expected_rnpv_B": round(float(pv * peak * pos), 3)}
    return total, detail


def pipeline_report(model=None):
    out = {}
    for co, (mcap, cash, cands) in COMPANIES.items():
        v, det = rnpv_pipeline(cands, model)
        out[co] = {"market_cap_B": mcap, "cash_B": cash, "rNPV_mean_B": round(float(v.mean()), 2), "rNPV_P10_B": round(float(np.percentile(v, 10)), 2),
                   "rNPV_P90_B": round(float(np.percentile(v, 90)), 2), "pipeline_value_to_mcap": round(float(v.mean() / mcap), 2),
                   "revenue_concentration_top_drug": round(max(c[2] for c in cands) / sum(c[2] for c in cands), 2), "candidates": det}
    return out


def patent_cliff(peak_sales_B, years_to_expiry, erosion=0.8, horizon=12):
    t = np.arange(horizon)
    rev = peak_sales_B * np.where(t <= years_to_expiry, 1.0, 1 - erosion * np.clip((t - years_to_expiry) / 2, 0, 1))
    return {"revenue_path_B": np.round(rev, 2).tolist(), "lost_in_3_years_after_expiry_B": round(float(peak_sales_B * 3 - rev[years_to_expiry + 1:years_to_expiry + 4].sum()), 2)}


def mean_variance_weights(mu, cov, risk_aversion=8.0, max_w=0.15):
    n = len(mu)
    res = minimize(lambda w: -(w @ mu) + 0.5 * risk_aversion * (w @ cov @ w), np.full(n, 1 / n), bounds=[(0, max_w)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, method="SLSQP")
    return res.x if res.success else np.full(n, 1 / n)


def backtest(returns, signal, window=36, ic=0.06, use_signal=True):
    """Monthly rebalance. Expected return = flat historical mean + signal alpha (IC x vol x z-score)."""
    dates = returns.index
    port, bench = [], []
    for t in range(window, len(dates) - 1):
        hist = returns.iloc[t - window:t]
        c = hist.cov().values
        cov = c * 0.7 + np.diag(np.diag(c)) * 0.3
        mu = np.full(returns.shape[1], hist.values.mean())
        if use_signal:
            mu = mu + ic * hist.std().values * signal.iloc[t].values
        w = mean_variance_weights(mu, cov)
        port.append(float(w @ returns.iloc[t + 1].values))
        bench.append(float(returns.iloc[t + 1].mean()))
    return pd.Series(port, index=dates[window + 1:]), pd.Series(bench, index=dates[window + 1:])


def perf(port, bench):
    ex = port - bench
    eq = (1 + port).cumprod()
    return {"sharpe": round(float(port.mean() / port.std() * np.sqrt(12)), 2), "information_ratio": round(float(ex.mean() / ex.std() * np.sqrt(12)), 2),
            "max_drawdown_pct": round(float(((eq - eq.cummax()) / eq.cummax()).min()) * 100, 1), "hit_rate_months_beating_benchmark_pct": round(float((ex > 0).mean() * 100), 1),
            "annual_return_pct": round(float(port.mean() * 12 * 100), 1)}


def trial_signal(trial_row, model):
    p = float(model.predict_proba(trial_row[FEATS])[:, 1][0])
    return {"enrolment_signal": "POSITIVE" if float(trial_row["enrol_velocity"].iloc[0]) >= 1.0 else "NEGATIVE", "dropout_signal": "POSITIVE" if float(trial_row["dropout"].iloc[0]) < 0.12 else "NEGATIVE",
            "competition_signal": "MODERATE RISK" if int(trial_row["competitors"].iloc[0]) >= 2 else "LOW RISK", "model_probability_of_success": round(p, 3)}
