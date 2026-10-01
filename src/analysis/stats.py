"""Small, dependency-light statistics helpers for pilot reports (numpy + scipy only)."""

import math

import numpy as np
from scipy import stats as _st


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion. Returns (p_hat, lo, hi); NaNs if n == 0."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_wilson(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.2f} [{lo:.2f}, {hi:.2f}]" if n else "0/0"


def auroc(scores, labels):
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney). NaN if a class is empty."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _st.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auprc(scores, labels):
    """Average precision (step-wise, sklearn convention)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def bootstrap_ci(fn, groups, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of fn(list_of_groups) resampling whole groups (e.g. episodes) with replacement."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        v = fn([groups[i] for i in idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(_st.spearmanr(x, y).statistic)
