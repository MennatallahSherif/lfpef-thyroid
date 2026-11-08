"""Discrimination, calibration, threshold and statistical-testing utilities."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- calibration
def fit_temperature(z, y):
    """Single global temperature minimising NLL (optimised on log T, so T > 0)."""
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(logT):
        p = np.clip(sigmoid(z / np.exp(logT)), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-6, 6), method="bounded")
    if abs(res.x) > 5.9:
        print(f"WARNING: temperature hit the search boundary (T={np.exp(res.x):.4g}); "
              "logits are probably near-constant - check the model.")
    return float(np.exp(res.x))


def fit_platt(z, y):
    """Logistic recalibration p = sigmoid(a + b*z). Unlike temperature scaling (b only),
    the intercept a can undo the prior shift caused by class-weighted training."""
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(params):
        q = np.clip(sigmoid(params[0] + params[1] * z), EPS, 1 - EPS)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    a, b = minimize(nll, x0=[0.0, 1.0], method="BFGS").x
    return float(a), float(b)


def calibration_slope_intercept(y, p):
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal a=0, b=1.
    Slope b: unpenalized logistic regression of y on logit(p).
    Intercept (calibration-in-the-large): a with b fixed at 1, i.e. the root of
    sum(y - sigmoid(a + logit(p))) = 0, found with a bracketing solver (exact, no optimizer tolerance)."""
    from scipy.optimize import brentq
    y, z = np.asarray(y, float), logit(p)

    def nll(params):
        q = np.clip(sigmoid(params[0] + params[1] * z), EPS, 1 - EPS)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    slope = minimize(nll, x0=[0.0, 1.0], method="BFGS").x[1]
    score = lambda a: np.sum(y - sigmoid(a + z))
    lo, hi = -20.0, 20.0
    citl = brentq(score, lo, hi, xtol=1e-10) if score(lo) * score(hi) < 0 else np.nan
    return float(citl), float(slope)


def ece(y, p, n_bins=10, strategy="uniform"):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:  # equal-mass bins
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    out = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def nll(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def prob_metrics(y, p, n_bins=10):
    """All threshold-free metrics. Includes the no-skill references reviewers ask for."""
    y = np.asarray(y).astype(int)
    prev = y.mean()
    b = brier(y, p)
    b_ref = prev * (1 - prev)                     # Brier of always predicting the prevalence
    citl, slope = calibration_slope_intercept(y, p)
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "PR-AUC (no-skill = prevalence)": prev,
        "Brier": b,
        "Brier (prevalence-only reference)": b_ref,
        "Brier skill score": 1 - b / b_ref,
        "NLL": nll(y, p),
        "ECE (uniform)": ece(y, p, n_bins, "uniform"),
        "ECE (equal-mass)": ece(y, p, n_bins, "quantile"),
        "Calibration intercept (CITL)": citl,
        "Calibration slope": slope,
    }


# ---------------------------------------------------------------- thresholds
def confusion_at(y, p, thr):
    y, yhat = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    return tp, fp, tn, fn


def wilson(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    ph = k / n
    den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def threshold_metrics(y, p, thr):
    tp, fp, tn, fn = confusion_at(y, p, thr)
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-12)
    s_lo, s_hi = wilson(tp, tp + fn); c_lo, c_hi = wilson(tn, tn + fp)
    p_lo, p_hi = wilson(tp, tp + fp); n_lo, n_hi = wilson(tn, tn + fn)
    return {"threshold": thr, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Sensitivity": sens, "Sens 95% CI": f"{s_lo:.3f}-{s_hi:.3f}",
            "Specificity": spec, "Spec 95% CI": f"{c_lo:.3f}-{c_hi:.3f}",
            "PPV": ppv, "PPV 95% CI": f"{p_lo:.3f}-{p_hi:.3f}",
            "NPV": npv, "NPV 95% CI": f"{n_lo:.3f}-{n_hi:.3f}", "F1": f1}


def youden_threshold(y, p):
    cands = np.unique(np.asarray(p))
    best_t, best_j = 0.5, -np.inf
    for t in cands:
        tp, fp, tn, fn = confusion_at(y, p, t)
        j = tp / max(tp + fn, 1) + tn / max(tn + fp, 1) - 1
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def min_sens_threshold(y, p, min_sens=0.90):
    """Highest threshold (=> best specificity) whose sensitivity is still >= min_sens."""
    best = None
    for t in np.unique(np.asarray(p)):
        tp, fp, tn, fn = confusion_at(y, p, t)
        if tp / max(tp + fn, 1) >= min_sens:
            best = float(t)
    return best if best is not None else float(np.min(p))


def net_benefit(y, p, thresholds):
    y = np.asarray(y).astype(int); n = len(y); prev = y.mean()
    rows = []
    for t in thresholds:
        tp, fp, _, _ = confusion_at(y, p, t)
        w = t / (1 - t)
        rows.append({"threshold": t, "model": tp / n - fp / n * w,
                     "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uncertainty
def stratified_bootstrap(y, scores: dict, fn, n_boot=2000, seed=42):
    """Resample positives and negatives separately (keeps 15 benign in every replicate).
    `scores` = {name: array}; fn(y, s) -> dict of metrics. Returns long DataFrame."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for name, s in scores.items():
            for k, v in fn(y[idx], np.asarray(s)[idx]).items():
                rows.append((b, name, k, v))
    return pd.DataFrame(rows, columns=["boot", "model", "metric", "value"])


def ci_table(point: dict, boot: pd.DataFrame, model: str):
    out = {}
    for k, v in point.items():
        bv = boot[(boot.model == model) & (boot.metric == k)]["value"].dropna()
        lo, hi = (np.percentile(bv, 2.5), np.percentile(bv, 97.5)) if len(bv) else (np.nan, np.nan)
        out[k] = {"value": v, "ci_low": lo, "ci_high": hi}
    return out


# DeLong (Sun & Xu 2014 fast implementation) ------------------------------------------
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong_test(y, s1, s2):
    """Paired DeLong test for two correlated ROC AUCs. Returns (auc1, auc2, diff, 95%CI, p)."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    y, preds = y[order], np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    m = int(y.sum()); n = len(y) - m
    tx = np.array([_midrank(p[:m]) for p in preds])
    ty = np.array([_midrank(p[m:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(diff) / se))
    return float(aucs[0]), float(aucs[1]), float(diff), (diff - 1.96 * se, diff + 1.96 * se), float(p)


def paired_bootstrap_diff(y, p_a, p_b, metric_fns: dict, n_boot=2000, seed=42):
    """Stratified paired bootstrap of metric(p_a) - metric(p_b) on the SAME resampled patients.
    Returns DataFrame: metric, diff (point), ci_low, ci_high, p_two_sided (share of replicates crossing 0)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); p_a = np.asarray(p_a, float); p_b = np.asarray(p_b, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    point = {k: f(y, p_a) - f(y, p_b) for k, f in metric_fns.items()}
    reps = {k: [] for k in metric_fns}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, f in metric_fns.items():
            reps[k].append(f(y[idx], p_a[idx]) - f(y[idx], p_b[idx]))
    rows = []
    for k in metric_fns:
        r = np.asarray(reps[k])
        pval = 2 * min((r <= 0).mean(), (r >= 0).mean())
        rows.append({"metric": k, "diff (A - B)": point[k], "ci_low": np.percentile(r, 2.5),
                     "ci_high": np.percentile(r, 97.5), "bootstrap p": min(pval, 1.0)})
    return pd.DataFrame(rows)
