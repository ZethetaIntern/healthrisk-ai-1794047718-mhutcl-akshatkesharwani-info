"""Hospital credit risk: traditional vs enhanced scorecards, PD models, early-warning backtest, bond spread model, rating mapping."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

FIN = ["op_margin", "dscr", "days_cash", "debt_cap", "cmi", "medicaid_share", "rural", "volume_growth"]
CLIN = ["readmit_rate", "hai_sir", "hcahps", "ed_boarding"]
TRAIN_LAST_Q, TEST_FIRST_Q, TEST_LAST_Q = 19, 24, 35         # 2019Q4, embargo 2020, test obs 2021Q1-2023Q4 (labels run to 2024Q4)


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def split(H):
    return H[H.q <= TRAIN_LAST_Q], H[(H.q >= TEST_FIRST_Q) & (H.q <= TEST_LAST_Q)]


def clinical_trend_features(H):
    H = H.sort_values(["hospital", "q"]).copy()
    g = H.groupby("hospital")
    for c in ["readmit_rate", "hcahps", "ed_boarding", "cmi"]:
        H[c + "_chg4"] = H[c] - g[c].shift(4)
    H["clinical_quality_score"] = (zs(H["readmit_rate"]) + zs(H["hai_sir"]) - zs(H["hcahps"]) + zs(H["ed_boarding"])) / 4
    return H


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.2))


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=60, l2_regularization=2.0, random_state=0)


def ks_stat(y, p):
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(p)
    return float(np.max(np.abs(np.cumsum(y[o]) / max(y.sum(), 1) - np.cumsum(1 - y[o]) / max((1 - y).sum(), 1))))


def psi(train_col, test_col, bins=10):
    """Population stability index: below 0.10 stable, 0.10-0.25 monitor, above 0.25 recalibrate."""
    edges = np.unique(np.quantile(train_col, np.linspace(0, 1, bins + 1)))
    a = np.histogram(np.clip(train_col, edges[0], edges[-1]), edges)[0] / len(train_col)
    b = np.histogram(np.clip(test_col, edges[0], edges[-1]), edges)[0] / len(test_col)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def score_block(y, p):
    return {"auroc": round(float(roc_auc_score(y, p)), 3), "gini": round(float(2 * roc_auc_score(y, p) - 1), 3), "ks": round(ks_stat(y, p), 3), "brier": round(float(brier_score_loss(y, p)), 4)}


def run_pd_models(H):
    H = clinical_trend_features(H)
    ENH = FIN + CLIN + ["readmit_rate_chg4", "hcahps_chg4", "ed_boarding_chg4", "cmi_chg4", "clinical_quality_score"]
    tr, te = split(H)
    res, preds = {}, {}
    for name, cols, mk in [("traditional_scorecard_LR", FIN, lr), ("enhanced_scorecard_LR", ENH, lr), ("traditional_HGB", FIN, hgb), ("enhanced_HGB", ENH, hgb)]:
        preds[name] = mk().fit(tr[cols], tr["y_breach_4q"]).predict_proba(te[cols])[:, 1]
        res[name] = score_block(te["y_breach_4q"], preds[name])
    res["n_train"], res["n_test"] = len(tr), len(te)
    res["train_events"], res["test_events"] = int(tr.y_breach_4q.sum()), int(te.y_breach_4q.sum())
    rng = np.random.default_rng(0)
    y, a, b = te["y_breach_4q"].values, preds["enhanced_scorecard_LR"], preds["traditional_scorecard_LR"]
    d = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    res["enhanced_minus_traditional_auroc_ci95"] = [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]
    return res, H, te.assign(pd_enh=a, pd_trad=b), ENH


def early_warning_backtest(H, clin_thr=0.6):
    """For every hospital that breached, when did each system first raise a flag? Lead time = quarters before the breach."""
    H = clinical_trend_features(H)
    fin_flag = lambda g: (g.dscr < 1.6) | (g.op_margin < 0.01) | (g.days_cash < 70)
    rows = []
    for h, g in H[H.breach_q >= 0].groupby("hospital"):
        bq = int(g.breach_q.iloc[0])
        g = g[g.q >= 8]
        clin, fin = g[g.clinical_quality_score > clin_thr], g[fin_flag(g)]
        rows.append({"hospital": h, "clin_lead": bq - clin.q.min() if len(clin) else np.nan, "fin_lead": bq - fin.q.min() if len(fin) else np.nan})
    d = pd.DataFrame(rows)
    ok = H[H.breach_q == -1]
    return {"breached_hospitals": len(d), "clinical_flag_detect_rate": round(float((d.clin_lead >= 1).mean()), 3), "financial_flag_detect_rate": round(float((d.fin_lead >= 1).mean()), 3),
            "median_lead_clinical_q": float(np.nanmedian(d.clin_lead)), "median_lead_financial_q": float(np.nanmedian(d.fin_lead)),
            "false_alarm_quarter_share_clinical": round(float((ok.clinical_quality_score > clin_thr).mean()), 3), "false_alarm_quarter_share_financial": round(float(fin_flag(ok).mean()), 3)}


def bond_spread_model(te):
    """Spread (bps) regressed on ln(PD) of the enhanced model."""
    X = np.log(te[["pd_enh"]].clip(lower=1e-4))
    m = LinearRegression().fit(X, te["spread_bps"])
    return {"spread_per_doubling_of_PD_bps": round(float(m.coef_[0] * np.log(2)), 1), "R2": round(float(m.score(X, te["spread_bps"])), 3)}, m


RATING_BANDS = [(0.003, "A"), (0.008, "A-"), (0.015, "BBB+"), (0.025, "BBB"), (0.04, "BBB-"), (0.07, "BB"), (0.12, "B"), (1.0, "CCC")]


def implied_rating(pd_1y):
    return next(r for t, r in RATING_BANDS if pd_1y <= t)


def worked_example(model_traditional, model_enhanced, row, cols_fin, cols_enh):
    """Mirrors the brief's hospital bond example: PD from financial ratios only versus financial + clinical metrics (4-quarter-ahead PD)."""
    p_t = float(model_traditional.predict_proba(row[cols_fin])[:, 1][0])
    p_e = float(model_enhanced.predict_proba(row[cols_enh])[:, 1][0])
    return {"PD_traditional": round(p_t, 4), "PD_enhanced": round(p_e, 4), "pd_change_pct": round((p_e / p_t - 1) * 100, 1), "rating_traditional": implied_rating(p_t), "rating_enhanced": implied_rating(p_e)}
