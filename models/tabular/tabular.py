"""Tabular baselines (gradient boosting; XGBoost / LightGBM were not available) with random-search tuning."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SPACE = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.08], "max_iter": [100, 200, 300], "min_samples_leaf": [30, 60, 100], "l2_regularization": [0.0, 1.0, 5.0]}


def make_lr(C=0.1):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, C=C))


def tune_classifier(X, y, n_iter=8, seed=0):
    s = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=seed), SPACE, n_iter=n_iter, scoring="roc_auc", cv=StratifiedKFold(3, shuffle=True, random_state=seed), random_state=seed)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


def tune_cost_model(X, y, n_iter=6, seed=0):
    """Poisson-loss boosting handles zero-inflated, right-skewed cost (Tweedie-like)."""
    s = RandomizedSearchCV(HistGradientBoostingRegressor(loss="poisson", random_state=seed), SPACE, n_iter=n_iter, scoring="neg_mean_poisson_deviance", cv=KFold(3, shuffle=True, random_state=seed), random_state=seed)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_


def class_metrics(y, p):
    return {"auroc": round(float(roc_auc_score(y, p)), 3), "auprc": round(float(average_precision_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 4), "prevalence": round(float(np.mean(y)), 4)}


def bootstrap_ci(y, p, metric=roc_auc_score, n=300, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]
