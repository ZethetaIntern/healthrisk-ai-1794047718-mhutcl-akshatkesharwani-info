"""Survival analysis: Cox PH (own Newton-Raphson), discrete-time hazard boosting (DeepSurv stand-in), C-index and time-dependent AUROC."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


class CoxPH:
    """Cox proportional hazards by Newton-Raphson on the Breslow partial likelihood, with a small ridge penalty."""
    def __init__(self, l2=1.0, iters=25):
        self.l2, self.iters = l2, iters

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        o = np.argsort(-np.asarray(time))                 # descending time: cumulative sums = risk sets
        Z, ev = Z[o], np.asarray(event)[o]
        n, p = Z.shape
        beta = np.zeros(p)
        for _ in range(self.iters):
            w = np.exp(np.clip(Z @ beta, -30, 30))
            s0 = np.cumsum(w)
            s1 = np.cumsum(w[:, None] * Z, axis=0)
            s2 = np.cumsum(w[:, None, None] * Z[:, :, None] * Z[:, None, :], axis=0)
            e = ev == 1
            grad = (Z[e] - s1[e] / s0[e, None]).sum(axis=0) - self.l2 * beta
            hess = (s2[e] / s0[e, None, None] - (s1[e, :, None] * s1[e, None, :]) / s0[e, None, None] ** 2).sum(axis=0) + self.l2 * np.eye(p)
            step = np.linalg.solve(hess, grad)
            beta += step
            if np.abs(step).max() < 1e-6:
                break
        self.beta = beta
        return self

    def risk(self, X):
        return ((np.asarray(X, dtype=float) - self.mu) / self.sd) @ self.beta


def concordance_index(time, event, risk):
    time, event, risk = map(np.asarray, (time, event, risk))
    num = den = 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(num / den) if den else float("nan")


def time_dependent_auroc(time, event, risk, horizons):
    """AUROC at horizon t: cases = event by t, controls = still event-free after t (those censored before t are dropped)."""
    time, event, risk = map(np.asarray, (time, event, risk))
    out = {}
    for h in horizons:
        case, ctrl = (time <= h) & (event == 1), time > h
        if case.sum() > 5 and ctrl.sum() > 5:
            y = np.r_[np.ones(case.sum()), np.zeros(ctrl.sum())]
            out[h] = round(float(roc_auc_score(y, np.r_[risk[case], risk[ctrl]])), 3)
    return out


class DiscreteHazard:
    """Person-period expansion; boosting predicts the hazard in each period.  Stand-in for DeepSurv / DeepHit."""
    def __init__(self, n_periods=12, period_len=1.0):
        self.n_periods, self.period_len = n_periods, period_len
        self.model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150, min_samples_leaf=60, random_state=0)

    def _expand(self, X, time, event):
        k = np.minimum(np.ceil(np.asarray(time) / self.period_len).astype(int), self.n_periods)
        rows = np.repeat(np.arange(len(X)), k)
        t = np.concatenate([np.arange(1, kk + 1) for kk in k])
        last = t == np.repeat(k, k)
        y = (last & (np.repeat(np.asarray(event), k) == 1) & (np.repeat(np.asarray(time), k) <= self.n_periods * self.period_len)).astype(int)
        return np.column_stack([np.asarray(X)[rows], t]), y

    def fit(self, X, time, event):
        Xe, y = self._expand(X, time, event)
        self.model.fit(Xe, y)
        return self

    def survival(self, X):
        X = np.asarray(X)
        s = np.ones(len(X))
        out = np.zeros((len(X), self.n_periods))
        for t in range(1, self.n_periods + 1):
            s = s * (1 - self.model.predict_proba(np.column_stack([X, np.full(len(X), t)]))[:, 1])
            out[:, t - 1] = s
        return out

    def risk(self, X, horizon=None):
        return 1 - self.survival(X)[:, (horizon or self.n_periods) - 1]


def evaluate(model_risk, time, event, horizons):
    return {"c_index": round(concordance_index(time, event, model_risk), 3), "td_auroc": time_dependent_auroc(time, event, model_risk, horizons)}
