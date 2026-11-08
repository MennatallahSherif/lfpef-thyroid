"""
STEP 05 - Paper figures (all read from step 03/04 outputs, nothing hard-coded).

  File names follow the figure numbers in the paper:
  fig3_leakage          image- vs patient-level split, every repeat shown as a dot     (Fig. 3)
  fig4_roc_pr           ROC + PR with no-skill line, LF-PEF vs AttnMIL, test cohort    (Fig. 4)
  fig5_calibration      reliability: uncalibrated / temperature / Platt, test cohort   (Fig. 5)
  fig6_decision_curve   net benefit vs threshold probability                           (Fig. 6)
  figS3_confusion       confusion matrices at the two OOF-derived operating points     (Fig. S3)

Usage:  python scripts/05_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score

import config as C
from lfpef import metrics as M

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600})
COL = {"LF-PEF": "#1f4e79", "LF-PEF (Platt)": "#2e8b57", "AttnMIL": "#c55a11", "LF-PEF (no TI-RADS aux)": "#7f7f7f"}


def save(name):
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(C.FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close()
    print("saved", name)


def main():
    S = pd.read_csv(C.PRED_DIR / "patient_scores_dev.csv")
    par = pd.read_csv(C.TABLE_DIR / "operating_params.csv").iloc[0]
    test = S[S.role == "test"]
    lf = test[test.model == "LF-PEF"]
    lf_pl = test[test.model == "LF-PEF (Platt)"].set_index("patient_id").loc[lf.patient_id] if \
        (test.model == "LF-PEF (Platt)").any() else None

    # ---- ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    rank_models = [m for m in ["LF-PEF", "LF-PEF (no TI-RADS aux)", "AttnMIL"] if (test.model == m).any()]
    for name, g in test[test.model.isin(rank_models)].groupby("model"):
        fpr, tpr, _ = roc_curve(g.y, g.p)
        ax[0].plot(fpr, tpr, color=COL.get(name), label=f"{name} (AUC {roc_auc_score(g.y, g.p):.3f})")
        pr, rc, _ = precision_recall_curve(g.y, g.p)
        ax[1].plot(rc, pr, color=COL.get(name), label=f"{name} (AP {average_precision_score(g.y, g.p):.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="(a) ROC, independent test cohort")
    prev = lf.y.mean()
    ax[1].axhline(prev, ls="--", c="grey", lw=1, label=f"No-skill (prevalence {prev:.3f})")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall", ylim=(min(0.85, prev - 0.05), 1.005))
    n0, n1 = int((lf.y == 0).sum()), int(lf.y.sum())
    for a in ax:
        a.legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle(f"Test cohort: {n1} malignant / {n0} benign patients", fontsize=10)
    save("fig4_roc_pr")

    # ---- reliability: one panel per calibration map (readable), histogram of scores underneath
    edges = np.linspace(0, 1, C.ECE_BINS + 1)
    curves = [("(a) Uncalibrated", M.sigmoid(lf.z), "#7f7f7f"), ("(b) Temperature scaling", lf.p.values, COL["LF-PEF"])]
    if lf_pl is not None:
        curves.append(("(c) Platt (intercept + slope)", lf_pl.p.values, COL["LF-PEF (Platt)"]))
    fig, ax = plt.subplots(2, len(curves), figsize=(3.6 * len(curves), 5.6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    for k, (lab, p, c) in enumerate(curves):
        p = np.asarray(p); idx = np.clip(np.digitize(p, edges[1:-1]), 0, C.ECE_BINS - 1)
        xs, ys, ns = [], [], []
        for b in range(C.ECE_BINS):
            m = idx == b
            if m.any():
                xs.append(p[m].mean()); ys.append(lf.y.values[m].mean()); ns.append(int(m.sum()))
        a = ax[0, k]
        a.plot([0, 1], [0, 1], "--", c="grey", lw=1)
        a.plot(xs, ys, "-", color=c, lw=1.2)
        a.scatter(xs, ys, s=[18 + 3 * n for n in ns], color=c, zorder=3, edgecolor="white", lw=0.5)
        a.set(xlim=(0, 1), ylim=(0, 1.03), title=lab)
        a.text(0.04, 0.96, f"ECE = {M.ece(lf.y, p, C.ECE_BINS):.3f}\nBrier = {M.brier(lf.y, p):.3f}",
               va="top", fontsize=9, transform=a.transAxes)
        ax[1, k].hist(p, bins=edges, color=c, alpha=0.8)
        ax[1, k].set(xlabel="Predicted probability")
    ax[0, 0].set_ylabel("Observed malignant fraction")
    ax[1, 0].set_ylabel("Patients")
    fig.suptitle(f"Test cohort (marker area proportional to patients per bin; {int((lf.y == 0).sum())} benign patients in total)", fontsize=9)
    save("fig5_calibration")

    # ---- confusion matrices at the two operating points (names match how they were chosen)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, (title, t) in zip(ax, [(f"Youden (OOF), t = {par.thr_youden:.3f}", par.thr_youden),
                                  (f"Sensitivity >= {par.min_sens:.2f} (OOF), t = {par.thr_sens:.3f}", par.thr_sens)]):
        tp, fp, tn, fn = M.confusion_at(lf.y, lf.p, t)
        cm = np.array([[tn, fp], [fn, tp]])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks([0, 1], ["Benign", "Malignant"]); a.set_yticks([0, 1], ["Benign", "Malignant"])
        a.set(xlabel="Predicted", ylabel="True", title=title)
        a.spines[:].set_visible(False)
    save("figS3_confusion")

    # ---- decision curve
    th = np.linspace(0.05, 0.95, 91)
    plt.figure(figsize=(6.5, 4.3))
    for name, g in test[test.model.isin(["LF-PEF", "LF-PEF (Platt)", "AttnMIL"])].groupby("model"):
        nb = M.net_benefit(g.y, g.p, th)
        plt.plot(nb.threshold, nb.model, color=COL.get(name), label=name)
    plt.plot(nb.threshold, nb.treat_all, "--", c="grey", label="Biopsy all")
    plt.axhline(0, c="black", lw=0.8, label="Biopsy none")
    plt.ylim(-0.05, max(prev, 0.05) + 0.03)
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title("Decision curve, test cohort")
    plt.legend(frameon=False, fontsize=9)
    save("fig6_decision_curve")

    # ---- leakage experiment
    f = C.TABLE_DIR / "leakage_runs.csv"
    if f.exists():
        R = pd.read_csv(f)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for a, col, title in [(ax[0], "img_auc", "Image-level AUC"), (ax[1], "pat_auc", "Patient-level AUC")]:
            for i, (prot, g) in enumerate(R.groupby("protocol", sort=False)):
                a.scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-.06, .06, len(g)), g[col], alpha=.7)
                a.errorbar(i + 0.18, g[col].mean(), yerr=g[col].std(), fmt="s", c="black", capsize=4)
            a.set_xticks(range(R.protocol.nunique()), R.protocol.unique()); a.set_title(title)
        ax[0].set_ylabel("Validation ROC-AUC (development cohort)")
        save("figS5_leakage_unmatched")

    # ---- matched leakage experiment (same evaluation images in both arms) -> Fig. 3
    f = C.TABLE_DIR / "leakage_matched_runs.csv"
    if f.exists():
        R = pd.read_csv(f)
        regs = [r for r in ["frozen", "finetuned"] if (R.regime == r).any()]
        fig, ax = plt.subplots(1, len(regs), figsize=(4.4 * len(regs), 4), sharey=True, squeeze=False)
        for a, reg in zip(ax[0], regs):
            g = R[R.regime == reg].sort_values("repeat")
            for _, r in g.iterrows():
                a.plot([0, 1], [r.pat_auc_clean, r.pat_auc_leaky], "-o", color="#7f7f7f", lw=1, ms=4)
            a.errorbar([-0.12, 1.12], [g.pat_auc_clean.mean(), g.pat_auc_leaky.mean()],
                       yerr=[g.pat_auc_clean.std(ddof=1), g.pat_auc_leaky.std(ddof=1)], fmt="s", c="black", capsize=4)
            d = g.pat_auc_diff
            a.set_xticks([0, 1], ["patient-disjoint\ntraining", "evaluation patients'\nother frames in training"])
            a.set_title(f"{'Frozen' if reg == 'frozen' else 'Fine-tuned'} backbone\n"
                        f"paired difference {d.mean():+.3f} (SD {d.std(ddof=1):.3f}), n = {len(d)}", fontsize=10)
            a.set_xlim(-0.4, 1.4)
        ax[0][0].set_ylabel("Patient-level ROC-AUC")
        save("fig3_leakage")


if __name__ == "__main__":
    main()
