"""Clinical NLP stand-in: TF-IDF + logistic regression (diagnosis), lexicon NER with negation, TF-IDF + ridge (complexity).
ClinicalBERT is not available here; real use would fine-tune Bio_ClinicalBERT on MIMIC-IV discharge summaries."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import label_binarize
from data.processing.synth import COND_KEYS, CONDITIONS, PRIMARY_PRIORITY

CLASSES = PRIMARY_PRIORITY + ["other"]
SEVERE_WORDS = ["poorly controlled", "brittle", "limited functional", "lives alone", "missed appointments", "frequent emergency"]


def dx_target(P):
    return P["primary_dx"].where(P["primary_dx"].isin(PRIMARY_PRIORITY), "other")


def diagnosis_classifier(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    Xtr, Xte = vec.fit_transform(train["note"]), vec.transform(test["note"])
    m = LogisticRegression(max_iter=3000, C=3).fit(Xtr, dx_target(train))
    prob = m.predict_proba(Xte)
    yb = label_binarize(dx_target(test), classes=m.classes_)
    aucs = [roc_auc_score(yb[:, i], prob[:, i]) for i in range(len(m.classes_)) if yb[:, i].sum() > 0]
    return {"macro_auroc": round(float(np.mean(aucs)), 3), "accuracy": round(float((m.predict(Xte) == dx_target(test)).mean()), 3), "n_test": len(test)}, (vec, m)


def extract_entities(note):
    """Condition mentions found by cue words; cues after 'no' / 'denies' within 4 tokens are negated (NegEx-style)."""
    toks = note.lower().split()
    found, negated = set(), set()
    for c in COND_KEYS:
        cues = set(CONDITIONS[c][6].split()) | {c.replace("_", " ")}
        for i, t in enumerate(toks):
            if t in cues:
                window = toks[max(0, i - 4):i]
                (negated if ("no" in window or "denies" in window) else found).add(c)
    return found, negated


def ner_f1(df, n=2000):
    tp = fp = fn = 0
    for _, r in df.head(n).iterrows():
        found, _ = extract_entities(r["note"])
        have = {c for c in COND_KEYS if r["cond_" + c] == 1}
        tp += len(found & have)
        fp += len(found - have)
        fn += len(have - found)
    p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(2 * p * rc / max(p + rc, 1e-9), 3),
            "note": "cue words are templated, and only up to 5 conditions are mentioned per note"}


def severity_cue_count(note):
    t = note.lower()
    return sum(w in t for w in SEVERE_WORDS)


def complexity_model(train, test, target="cost_12m"):
    """Cost from notes ALONE (no structured codes): TF-IDF -> SVD(32) -> ridge on ln(1+cost)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    svd = TruncatedSVD(32, random_state=0)
    Xtr = svd.fit_transform(vec.fit_transform(train["note"]))
    Xte = svd.transform(vec.transform(test["note"]))
    m = Ridge(alpha=1.0).fit(Xtr, np.log1p(train[target]))
    return {"R2_notes_only_ln_cost": round(float(r2_score(np.log1p(test[target]), m.predict(Xte))), 3)}, (vec, svd, m)


def complexity_scores(P, fitted, train_mask):
    """Out-of-fold complexity score for training rows (5 folds), model prediction for other rows."""
    vec, svd, m = fitted
    Z = svd.transform(vec.transform(P["note"]))
    out = m.predict(Z)
    idx = np.where(train_mask)[0]
    y = np.log1p(P["cost_12m"].values)
    for f in np.array_split(np.random.default_rng(0).permutation(idx), 5):
        tr = np.setdiff1d(idx, f)
        out[f] = Ridge(alpha=1.0).fit(Z[tr], y[tr]).predict(Z[f])
    return out
