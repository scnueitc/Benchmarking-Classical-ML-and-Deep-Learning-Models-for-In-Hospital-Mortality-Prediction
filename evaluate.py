# -*- coding: utf-8 -*-
"""Evaluation metrics and statistical tests.

Discrimination / calibration:
    ece_score, hosmer_lemeshow, calibration_slope_intercept
Paired AUROC comparison (DeLong, Sun & Xu 2014 fast implementation):
    delong_test, delong_family (+ holm_adjust for multiple testing)
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------------ calibration
def ece_score(y, p, n_bins=10):
    """Expected calibration error over `n_bins` equal-width probability bins."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece, n = 0.0, len(y)
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return ece


def hosmer_lemeshow(y, p, g=10):
    """H-L goodness-of-fit statistic on `g` deciles of predicted risk."""
    q = pd.qcut(pd.Series(p).rank(method="first"), q=g, labels=False)
    obs, ntot = 0.0, 0
    for b in range(g):
        m = (q == b).values
        if m.sum() == 0:
            continue
        o, e, nn = y[m].sum(), p[m].sum(), m.sum()
        obs += (o - e) ** 2 / (e * (1 - e / nn) + 1e-12)
        ntot += 1
    dfree = max(ntot - 2, 1)
    pval = float(st.chi2.sf(obs, dfree))
    return obs, dfree, pval


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Logistic recalibration of the logit; slope 1 / intercept 0 = perfect."""
    lr = LogisticRegression(max_iter=2000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ------------------------------------------------------------------ DeLong
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fast_delong(preds_sorted, m):
    """preds_sorted: (k, N) sorted so positives come first; m = #positives."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    """Two-sided paired DeLong test for AUROC(p1) vs AUROC(p2). Returns (z, p)."""
    y = np.asarray(y).astype(int)
    assert set(np.unique(y)) <= {0, 1}
    order = (-y).argsort()
    m = int(y.sum())
    P = np.vstack([p1, p2])[:, order]
    aucs, cov = fast_delong(P, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        var = 1e-12
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = float(st.norm.sf(abs(z)) * 2)
    return z, p


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    n = len(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = p[idx] * (n - rank)
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def delong_family(y, preds_dict, tag, out_dir):
    """All pairwise DeLong tests among models in `preds_dict`; saves CSV."""
    names = list(preds_dict.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            z, p = delong_test(y, preds_dict[names[i]], preds_dict[names[j]])
            rows.append(dict(model1=names[i], model2=names[j], z=z, p_raw=p))
    d = pd.DataFrame(rows)
    d["p_holm"] = holm_adjust(d["p_raw"].values)
    d.to_csv(os.path.join(out_dir, f"delong_{tag}.csv"), index=False)
    return d
