"""Stacking ensemble: level-0 models on different views, level-1 regularised linear meta-learner, time-aware cross-validation."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def time_aware_oof(models, views_train, y, n_splits=5, kind="clf"):
    """models: {name: factory}; views_train: {name: array}. Rows must be ordered by time. Returns OOF matrix and the mask of predicted rows."""
    names = list(models)
    oof = np.full((len(y), len(names)), np.nan)
    for tr, va in TimeSeriesSplit(n_splits).split(np.arange(len(y))):
        for j, n in enumerate(names):
            m = models[n]()
            X = views_train[n]
            m.fit(X[tr], y[tr])
            oof[va, j] = m.predict_proba(X[va])[:, 1] if kind == "clf" else m.predict(X[va])
    return oof, ~np.isnan(oof).any(axis=1)


def fit_meta(oof, y, mask, kind="clf", alpha=1.0):
    Z = logit(oof[mask]) if kind == "clf" else oof[mask]
    return LogisticRegression(C=1.0, max_iter=1000).fit(Z, y[mask]) if kind == "clf" else Ridge(alpha=alpha).fit(Z, y[mask])


def predict_meta(meta, preds, kind="clf"):
    Z = logit(preds) if kind == "clf" else preds
    return meta.predict_proba(Z)[:, 1] if kind == "clf" else meta.predict(Z)
