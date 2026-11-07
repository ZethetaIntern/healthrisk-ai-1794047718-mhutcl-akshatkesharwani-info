"""Comorbidity and drug-interaction graphs with relational propagation (stand-in for a GAT / HGT).
No attention and no PyTorch Geometric: neighbour averaging over the graphs, then SVD to patient embeddings."""
import numpy as np
from data.processing.synth import COND_KEYS


def comorbidity_graph(P):
    """Lift of co-occurrence between conditions: P(a and b) / (P(a) P(b))."""
    C = P[["cond_" + c for c in COND_KEYS]].values.astype(float)
    p = C.mean(axis=0)
    lift = ((C.T @ C) / len(C)) / np.outer(p, p).clip(1e-9)
    np.fill_diagonal(lift, 0)
    return lift


def norm_adj(A):
    d = A.sum(axis=1, keepdims=True)
    return np.divide(A, d, out=np.zeros_like(A), where=d > 0)


def patient_embeddings(P, graph, dim=16, train_mask=None):
    """Smooth each patient's condition vector over the comorbidity graph (2 hops) and drug vector over the DDI graph (1 hop)."""
    C = P[["cond_" + c for c in COND_KEYS]].values.astype(float)
    lift = comorbidity_graph(P[train_mask] if train_mask is not None else P)
    Ac = norm_adj(np.clip(lift - 1, 0, None))
    Cs = C + C @ Ac + C @ Ac @ Ac
    D = np.zeros((len(P), graph["ddi"].shape[0]))
    for i, ds in enumerate(graph["drug_sets"][:len(P)]):
        D[i, ds] = 1
    Ds = D + D @ norm_adj((graph["ddi"] > 0).astype(float))
    M = np.hstack([Cs, Ds])
    ref = M[train_mask] if train_mask is not None else M
    mu = ref.mean(axis=0)
    _, _, vt = np.linalg.svd(ref - mu, full_matrices=False)
    return (M - mu) @ vt[:dim].T


def polypharmacy_risk(graph, drug_set):
    """Weighted interaction density: (3 x major + 2 x moderate + minor) / number of pairs."""
    k = len(drug_set)
    if k < 2:
        return 0.0
    sub = graph["ddi"][np.ix_(drug_set, drug_set)]
    return float(sub[np.triu_indices(k, 1)].sum() / (k * (k - 1) / 2))
