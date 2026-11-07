"""Health insurance actuarial module: GLM premium baseline, enhanced pricing, risk stratification, IBNR (chain ladder / Bornhuetter-Ferguson)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def rating_matrix(P):
    """Traditional rating factors only: age band, sex, region, plan, tobacco."""
    X = pd.get_dummies(pd.cut(P["age"], [0, 50, 55, 60, 65, 70, 75, 120]).astype(str), prefix="age").astype(float)
    for c in ["region", "plan"]:
        X = pd.concat([X, pd.get_dummies(P[c], prefix=c).astype(float)], axis=1)
    X["sex"], X["tobacco"] = P["sex"].values, P["tobacco"].values
    return X


def glm_baseline(train, test):
    Xtr = rating_matrix(train)
    Xte = rating_matrix(test).reindex(columns=Xtr.columns, fill_value=0)
    m = make_pipeline(StandardScaler(), TweedieRegressor(power=1.5, alpha=1e-3, link="log", max_iter=500))
    m.fit(Xtr, train["cost_12m"])
    return m.predict(Xte), m


def predictive_ratio(actual, pred):
    return float(np.sum(pred) / np.sum(actual))


def pricing_report(test, pred, label, cohort_size=200, seed=0):
    y = test["cost_12m"].values
    top = y >= np.quantile(y, 0.95)
    idx = np.random.default_rng(seed).permutation(len(y))
    cohorts = np.array_split(idx, len(y) // cohort_size)
    mape_c = np.mean([abs(pred[c].sum() - y[c].sum()) / y[c].sum() for c in cohorts]) * 100
    pos = y > 0
    sub = {}
    for col in ["plan", "region"]:
        for k, g in test.groupby(col).groups.items():
            ii = test.index.get_indexer(g)
            sub[f"{col}={k}"] = round(predictive_ratio(y[ii], pred[ii]), 3)
    age_band = pd.cut(test["age"], [0, 60, 70, 120], labels=["<60", "60-70", "70+"]).astype(str)
    for k in ["<60", "60-70", "70+"]:
        ii = np.where(age_band.values == k)[0]
        sub[f"age={k}"] = round(predictive_ratio(y[ii], pred[ii]), 3)
    return {"model": label, "predictive_ratio_all": round(predictive_ratio(y, pred), 3), "predictive_ratio_top5pct": round(predictive_ratio(y[top], pred[top]), 3),
            "R2": round(float(r2_score(y, pred)), 3), "MAPE_individual_pct_positive_cost": round(float(np.mean(np.abs(y[pos] - pred[pos]) / y[pos]) * 100), 1),
            "MAPE_cohort200_pct": round(float(mape_c), 2), "worst_subgroup_deviation_pct": round(max(abs(v - 1) for v in sub.values()) * 100, 1), "subgroup_ratios": sub}


def risk_stratification(test, pred):
    y = test["cost_12m"].values
    d = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    tab = pd.DataFrame({"decile": d, "actual": y, "pred": pred}).groupby("decile").mean().round(0)
    top = y >= np.quantile(y, 0.95)
    return {"auroc_top5pct_cost": round(float(roc_auc_score(top, pred)), 3), "top_decile_over_bottom_decile": round(float(tab.actual.iloc[-1] / tab.actual.iloc[0]), 2),
            "share_of_cost_in_top_decile": round(float(y[d == 9].sum() / y.sum()), 3)}, tab


def loss_ratio_by_decile(test, pred, loading=0.16):
    """Loss ratio by predicted-risk decile when premium = predicted cost x (1 + loading). Flat ratios = no cross-subsidy."""
    y = test["cost_12m"].values
    d = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    lr = pd.DataFrame({"d": d, "claims": y, "prem": pred * (1 + loading)}).groupby("d").sum()
    r = lr.claims / lr.prem
    return {"min": round(float(r.min()), 3), "max": round(float(r.max()), 3), "spread": round(float(r.max() - r.min()), 3)}


def development_factors(obs):
    f = []
    for j in range(obs.shape[1] - 1):
        m = ~np.isnan(obs[:, j + 1])
        f.append(obs[m, j + 1].sum() / obs[m, j].sum())
    return np.array(f)


def chain_ladder(obs):
    f, n = development_factors(obs), obs.shape[0]
    ult, latest = np.zeros(n), np.zeros(n)
    for i in range(n):
        j = n - 1 - i
        latest[i] = obs[i, j]
        ult[i] = latest[i] * np.prod(f[j:])
    return ult, latest


def bornhuetter_ferguson(obs, premium, a_priori_lr=0.84):
    f, n = development_factors(obs), obs.shape[0]
    ult, latest = np.zeros(n), np.zeros(n)
    for i in range(n):
        j = n - 1 - i
        latest[i] = obs[i, j]
        ult[i] = latest[i] + premium[i] * a_priori_lr * (1 - 1 / np.prod(f[j:]))
    return ult, latest


def emergence_adjusted(obs):
    """Chain ladder where each accident year's early emergence speed (dev-1 / dev-0 versus earlier years) rescales its future factors."""
    f, n = development_factors(obs), obs.shape[0]
    base_ratio = np.nanmean(obs[:5, 1] / obs[:5, 0])
    ult, latest = np.zeros(n), np.zeros(n)
    for i in range(n):
        j = n - 1 - i
        latest[i] = obs[i, j]
        speed = (obs[i, 1] / obs[i, 0]) / base_ratio if not np.isnan(obs[i, 1]) else 1.0
        ult[i] = latest[i] * np.prod(1 + (f[j:] - 1) / np.clip(speed, 0.5, 2.0))
    return ult, latest


def ibnr_accuracy(true_ult, ult, latest, years, last_n=6):
    true_ibnr, est_ibnr = true_ult - latest, ult - latest
    sel = slice(len(years) - last_n, len(years) - 1)
    dev = np.abs(est_ibnr[sel] / true_ibnr[sel] - 1)
    return {"total_ibnr_est": round(float(est_ibnr.sum()), 1), "total_ibnr_true": round(float(true_ibnr.sum()), 1),
            "total_error_pct": round(float((est_ibnr.sum() / true_ibnr.sum() - 1) * 100), 2),
            "mean_abs_dev_recent_years_pct": round(float(dev.mean() * 100), 1), "ultimate_MAPE_pct": round(float(np.mean(np.abs(ult / true_ult - 1)) * 100), 2)}
