"""
STEP 06 - Supplementary Information (no training; reads outputs of steps 00-05).

Writes outputs/supplementary/:
  supplementary.tex     self-contained LaTeX file (compile with pdflatex)
  figS1_training_curves, figS2_roc_oof_test_pooled, figS4_pooled_calibration  (.png/.pdf)
  figS3_confusion is copied from outputs/figures (made by step 05)
  numbers_for_paper.txt values the main text still needs (e.g. AttnMIL OOF AUC)

Usage:  python scripts/06_supplementary.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from lfpef import metrics as M
from lfpef.data import BENIGN_KWS, MALIGNANT_KWS

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600})
SUP = C.OUT / "supplementary"
notes = []


# ----------------------------------------------------------------- helpers
def esc(x):
    s = str(x)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 (">=", r"$\geq$"), ("<=", r"$\leq$")]:
        s = s.replace(a, b)
    return s


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "--" if not np.isfinite(v) else f"{v:.{nd}f}"
    return esc(v)


def latex_table(df, caption, label, colspec=None, size=r"\small"):
    colspec = colspec or "l" + "c" * (df.shape[1] - 1)
    head = " & ".join(esc(c) for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(fmt(v) for v in row) + r" \\" for row in df.itertuples(index=False))
    return (f"\\begin{{nftable}}\n{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\fitwidth{{\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}}}\n\\end{{nftable}}\n")


def save(name):
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(SUP / f"{name}.{ext}", bbox_inches="tight")
    plt.close()
    print("saved", name)


def read(p):
    return pd.read_csv(p) if Path(p).exists() else None


def patient_auc_from_preds(df):
    if df["file_name"].fillna("").astype(str).str.len().gt(0).any():      # image-level -> mean logit
        df = df.groupby("patient_id").agg(y=("y", "first"), logit=("logit", "mean")).reset_index()
    return roc_auc_score(df.y, df.logit) if df.y.nunique() > 1 else np.nan


# ----------------------------------------------------------------- tables
def table_labels():
    ex = read(C.TABLE_DIR / "excluded_histopathology.csv")
    n_ex = 0 if ex is None else len(ex)
    df = pd.DataFrame([
        {"Label": "Malignant (1)", "Rule (histopathology of nodule 1, case-insensitive)":
            "contains any of: " + ", ".join(MALIGNANT_KWS) + " (checked first)"},
        {"Label": "Benign (0)", "Rule (histopathology of nodule 1, case-insensitive)":
            "contains any of: " + ", ".join(BENIGN_KWS)},
        {"Label": "Excluded", "Rule (histopathology of nodule 1, case-insensitive)":
            f"report present but matches neither list ({n_ex} cases"
            + (": " + "; ".join(sorted(ex.histopathology.astype(str).unique())) if n_ex else "") + ")"},
        {"Label": "Not included", "Rule (histopathology of nodule 1, case-insensitive)":
            "no histopathology report (cytology-only or unverified cases)"},
    ])
    return latex_table(df, "Outcome definition.", "tab:S_labels", "lp{0.75\\linewidth}")


def table_folds():
    pat = pd.read_csv(C.PATIENTS_CSV)
    rows = []
    for col, name in [("fold_dev", "Development CV"), ("fold_all", "Pooled CV (all patients)")]:
        g = pat[pat[col] > 0].groupby(col).y.agg(["size", "sum"])
        for k, r in g.iterrows():
            rows.append({"Analysis": name, "Fold": int(k), "Patients": int(r["size"]),
                         "Malignant": int(r["sum"]), "Benign": int(r["size"] - r["sum"])})
    return latex_table(pd.DataFrame(rows), "Composition of the patient-level folds.", "tab:S_folds")


def table_hparams():
    rows = [("Backbone", f"{C.BACKBONE} (ImageNet)"), ("Input size", f"{C.IMG_SIZE} x {C.IMG_SIZE}"),
            ("Augmentation", "horizontal flip, brightness 0.12, contrast 0.85-1.15 (training only)"),
            ("Heads", "malignancy logit" + (f" + TI-RADS group (loss weight {C.TIRADS_LOSS_WEIGHT})" if C.USE_TIRADS_AUX else "")),
            ("Dropout", C.DROPOUT), ("Batch size (frame / MIL)", f"{C.BATCH_SIZE} / {C.MIL_BATCH}"),
            ("Optimizer", "Adam"),
            ("Stage 1 (frozen backbone)", f"lr {C.STAGE1_LR:g}, max {C.STAGE1_EPOCHS} epochs (MIL: {C.MIL_STAGE1_EPOCHS})"),
            ("Stage 2 (last layers unfrozen, BN frozen)", f"last {C.UNFREEZE_LAST} layers, lr {C.STAGE2_LR:g}, "
                                                          f"max {C.STAGE2_EPOCHS} epochs (MIL: {C.MIL_STAGE2_EPOCHS})"),
            ("Model selection", f"patient-level AUC on inner validation ({int(C.INNER_VAL_FRAC * 100)}% of training patients), patience {C.PATIENCE}"),
            ("Class imbalance", "class weights recomputed in every training fold"),
            ("MIL attention dimension", C.MIL_ATTN_DIM), ("Outer folds", C.N_FOLDS), ("Random seed", C.SEED),
            ("Leakage experiment, matched (primary)", "5 repeats; 20% of development patients as evaluation set; about half of "
             "each evaluation patient's frames added to training, training size equalized; no early stopping"),
            ("  frozen regime", "15 epochs, backbone frozen"),
            ("  fine-tuned regime", f"10 frozen epochs, then 10 epochs with the last {C.UNFREEZE_LAST} layers unfrozen (lr {C.STAGE2_LR:g})"),
            ("Leakage experiment, unmatched", f"{C.LEAK_REPEATS} repeats; image-level vs patient-level 80/20 split; "
             f"{C.LEAK_EPOCHS} epochs, backbone frozen; no early stopping"),
            ("Bootstrap replicates", f"{C.N_BOOT} (stratified by class)"), ("ECE bins", C.ECE_BINS)]
    df = pd.DataFrame(rows, columns=["Setting", "Value"]).astype(str)
    return latex_table(df, "Training and evaluation settings.", "tab:S_hparams",
                       "lp{0.62\\linewidth}")


def summarise_history(tag, cohort):
    rows = []
    for f in sorted(C.LOG_DIR.glob(f"{tag}_{cohort}_fold*_history.csv")):
        k = int(f.stem.split("fold")[1].split("_")[0])
        h = pd.read_csv(f)
        s1, s2 = h[h.stage == "stage1"], h[h.stage == "stage2"]
        b1 = s1.inner_patient_auc.max()
        b2 = s2.inner_patient_auc.max() if len(s2) else np.nan
        row = {"Model": {"frame": "LF-PEF CNN", "attnmil": "AttnMIL"}.get(tag, tag), "Fold": k,
               "Stage-1 epochs": len(s1), "Stage-1 best": b1, "Stage-2 epochs": len(s2),
               "Stage 2 kept": "yes" if (len(s2) and b2 > b1 + 1e-4) else "no",
               "Inner AUC (selected)": np.nanmax([b1, b2])}
        pred = read(C.PRED_DIR / f"{tag}_{cohort}_fold{k}.csv")
        if pred is not None:
            row["OOF AUC (fold)"] = patient_auc_from_preds(pred[pred.role == "oof"])
            if (pred.role == "test").any():
                row["Test AUC (single model)"] = patient_auc_from_preds(pred[pred.role == "test"])
        rows.append(row)
    return pd.DataFrame(rows)


def table_training():
    parts = [summarise_history(t, c) for t, c in [("frame", "dev"), ("attnmil", "dev")]]
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not len(df):
        return "% no training logs found\n"
    df = df[[c for c in ["Model", "Fold", "Stage-1 epochs", "Stage-1 best", "Stage-2 epochs", "Stage 2 kept",
                         "Inner AUC (selected)", "OOF AUC (fold)", "Test AUC (single model)"] if c in df.columns]]
    out = latex_table(df, "Per-fold training summary, development cross-validation.", "tab:S_training")
    pooled = summarise_history("frame", "all")
    if len(pooled):
        pooled = pooled.drop(columns=[c for c in ["Test AUC (single model)"] if c in pooled.columns])
        out += latex_table(pooled, "Per-fold training summary, pooled cross-validation.",
                           "tab:S_training_pooled")
    return out


def table_delong():
    d = read(C.TABLE_DIR / "table_delong.csv")
    if d is None:
        return "% table_delong.csv missing\n"
    d = d.rename(columns={"p (Holm, single-model family)": "p (Holm)"})
    return latex_table(d, "Paired DeLong comparisons.", "tab:S_delong")


def table_oof():
    d = read(C.TABLE_DIR / "table_oof_dev.csv")
    if d is None:
        return "% table_oof_dev.csv missing\n"
    keep = ["Model", "ROC-AUC", "PR-AUC", "Brier", "Brier (prevalence-only reference)", "Brier skill score",
            "ECE (uniform)", "Calibration intercept (CITL)", "Calibration slope"]
    d = d[[c for c in keep if c in d.columns]]
    for _, r in d.iterrows():
        notes.append(f"OOF ROC-AUC  {r['Model']}: {r['ROC-AUC']:.3f}")
    t = short_metric(transposed(d, "Model", [c for c in keep if c != "Model"],
                                {"LF-PEF (OOF, calibrated in-sample)": "LF-PEF (temperature)",
                                 "LF-PEF Platt (OOF, calibrated in-sample)": "LF-PEF (Platt)",
                                 "AttnMIL (OOF, calibrated in-sample)": "AttnMIL (temperature)"}))
    return latex_table(t, "Development OOF performance.", "tab:S_oof")


def table_leakage():
    d = read(C.TABLE_DIR / "leakage_runs.csv")
    if d is None:
        return "% leakage_runs.csv missing\n"
    d = d[["repeat", "protocol", "img_auc", "pat_auc", "img_prauc", "pat_prauc", "val_patients", "overlap_ratio"]].copy()
    d["repeat"] = d["repeat"] + 1
    d.columns = ["Repeat", "Protocol", "Image AUC", "Patient AUC", "Image PR-AUC", "Patient PR-AUC",
                 "Val. patients", "Overlap ratio"]
    return latex_table(d.sort_values(["Repeat", "Protocol"]), "Unmatched leakage experiment, individual repeats.", "tab:S_leakage")


def transposed(d, name_col, metrics, names=None):
    """Rows = metrics, columns = models/maps: much narrower than the wide CSV."""
    d = d.copy()
    if names:
        d[name_col] = d[name_col].map(lambda x: names.get(x, x))
    m = [c for c in metrics if c in d.columns]
    t = d.set_index(name_col)[m].T.reset_index().rename(columns={"index": "Metric"})
    t.columns = [str(c) for c in t.columns]
    return t


def short_metric(t):
    ren = {"Brier (prevalence-only reference)": "Brier (prevalence-only ref.)", "Brier (prevalence-only ref.)": "Brier (prevalence-only ref.)",
           "Calibration intercept (CITL)": "Calibration intercept", "ECE (uniform)": "ECE"}
    t["Metric"] = t["Metric"].map(lambda x: ren.get(x, x))
    return t


def csv_table(name, caption, label, drop=()):
    d = read(C.TABLE_DIR / name)
    if d is None:
        return f"% {name} missing\n"
    d = d.drop(columns=[c for c in drop if c in d.columns])
    return latex_table(d, caption, label)


CAL_METRICS = ["ROC-AUC", "Brier", "Brier (prevalence-only ref.)", "Brier skill score", "ECE (uniform)",
               "Calibration intercept", "Calibration slope"]


def table_leak_matched():
    d = read(C.TABLE_DIR / "leakage_matched_runs.csv")
    if d is None:
        return "% leakage_matched_runs.csv missing\n"
    d = d.sort_values(["regime", "repeat"])
    out = pd.DataFrame({"Regime": d.regime, "Repeat": d["repeat"], "AUC leakage-free": d.pat_auc_clean,
                        "AUC leaky": d.pat_auc_leaky, "Difference": d.pat_auc_diff, "95% CI": d.pat_diff_ci,
                        "DeLong p": d.pat_delong_p, "Frame-level diff.": d.img_auc_diff})
    return latex_table(out, "Matched leakage experiment, individual repeats (patient-level AUC).", "tab:S_leak_matched", size="")


def table_cal_dev():
    d = read(C.TABLE_DIR / "table_calibration_dev_crossfit.csv")
    if d is None:
        return "% table_calibration_dev_crossfit.csv missing\n"
    t = short_metric(transposed(d, "Calibration", CAL_METRICS,
                                {"temperature (cross-fitted, dev only)": "Temperature", "Platt (cross-fitted, dev only)": "Platt"}))
    return latex_table(t, "Development-only cross-fitted calibration.", "tab:S_cal_dev", size="")


def table_cal_diff():
    d = read(C.TABLE_DIR / "table_calibration_paired_diff.csv")
    if d is None:
        return "% table_calibration_paired_diff.csv missing\n"
    out = pd.DataFrame({"Cohort": d.cohort.str.replace("Development OOF (587), cross-fitted", "Development, cross-fitted", regex=False),
                        "Comparison (A minus B)": d.comparison.str.replace("single-model-OOF Platt minus nested-ensemble Platt",
                                                                       "Platt: single-model minus nested fit", regex=False),
                        "Metric": d.metric.str.replace("ECE (uniform)", "ECE", regex=False),
                        "Difference": d["diff (A - B)"],
                        "95% CI": [f"{a:.3f} to {b:.3f}" for a, b in zip(d.ci_low, d.ci_high)],
                        "Bootstrap p": ["$<$0.001" if v < 0.001 else f"{v:.3f}" for v in d["bootstrap p"]]})
    return latex_table(out, "Paired differences between calibration maps.", "tab:S_cal_diff", size="")


def table_cal_nested():
    d = read(C.TABLE_DIR / "table_calibration_nested.csv")
    if d is None:
        return "% table_calibration_nested.csv missing\n"
    t = short_metric(transposed(d, "Calibration map", CAL_METRICS + ["fitted parameters"],
                                {"Temperature fitted on nested-ensemble OOF": "Temperature (nested fit)",
                                 "Platt fitted on nested-ensemble OOF": "Platt (nested fit)",
                                 "Platt fitted on single-model OOF (main analysis)": "Platt (single-model fit)"}))
    t["Metric"] = t["Metric"].replace({"fitted parameters": "Fitted parameters"})
    return latex_table(t, "Nested-ensemble calibration on the test cohort.", "tab:S_cal_nested", size="")


# ----------------------------------------------------------------- figures
def fig_training():
    sets = [("frame", "dev", "LF-PEF CNN, development CV"), ("attnmil", "dev", "AttnMIL, development CV"),
            ("frame", "all", "LF-PEF CNN, pooled CV")]
    sets = [s for s in sets if list(C.LOG_DIR.glob(f"{s[0]}_{s[1]}_fold*_history.csv"))]
    if not sets:
        return False
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.4), sharey=True, squeeze=False)
    for ax, (tag, coh, title) in zip(axes[0], sets):
        for f in sorted(C.LOG_DIR.glob(f"{tag}_{coh}_fold*_history.csv")):
            h = pd.read_csv(f)
            k = f.stem.split("fold")[1].split("_")[0]
            x = np.arange(1, len(h) + 1)
            line, = ax.plot(x, h.inner_patient_auc, lw=1.2, label=f"fold {k}")
            n1 = (h.stage == "stage1").sum()
            if n1 < len(h):
                ax.axvline(n1 + 0.5, color=line.get_color(), ls=":", lw=0.8)
        ax.set(title=title, xlabel="Epoch (stage 1, then stage 2 after dotted line)")
        ax.legend(frameon=False, fontsize=7)
    axes[0][0].set_ylabel("Inner-validation patient AUC")
    save("figS1_training_curves")
    return True


def fig_roc():
    dev = read(C.PRED_DIR / "patient_scores_dev.csv")
    pooled = read(C.PRED_DIR / "patient_scores_all_frame_platt.csv")
    if dev is None:
        return False
    plt.figure(figsize=(4.8, 4.4))
    curves = [("Development OOF (single fold models)", dev[(dev.role == "oof") & (dev.model == "LF-PEF")]),
              ("Test (5-model ensemble)", dev[(dev.role == "test") & (dev.model == "LF-PEF")])]
    if pooled is not None:
        curves.append(("Pooled CV, all patients", pooled))
    for lab, g in curves:
        fpr, tpr, _ = roc_curve(g.y, g.p)
        plt.plot(fpr, tpr, label=f"{lab} (AUC {roc_auc_score(g.y, g.p):.3f}; n={len(g)}, benign={int((g.y == 0).sum())})")
    plt.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.legend(frameon=False, fontsize=7, loc="lower right")
    save("figS2_roc_oof_test_pooled")
    return True


def fig_pooled_calibration():
    t = read(C.PRED_DIR / "patient_scores_all_frame.csv")
    p = read(C.PRED_DIR / "patient_scores_all_frame_platt.csv")
    if t is None or p is None:
        return False
    edges = np.linspace(0, 1, C.ECE_BINS + 1)
    plt.figure(figsize=(5, 4.4))
    for lab, d, c in [("Temperature (cross-fitted)", t, "#1f4e79"), ("Platt (cross-fitted)", p, "#2e8b57")]:
        idx = np.clip(np.digitize(d.p, edges[1:-1]), 0, C.ECE_BINS - 1)
        xs, ys, ns = [], [], []
        for b in range(C.ECE_BINS):
            m = idx == b
            if m.any():
                xs.append(d.p.values[m].mean()); ys.append(d.y.values[m].mean()); ns.append(int(m.sum()))
        plt.plot(xs, ys, "o-", color=c, label=f"{lab}: ECE={M.ece(d.y, d.p, C.ECE_BINS):.3f}, "
                                              f"Brier={M.brier(d.y, d.p):.3f}")
        for x, y, n in zip(xs, ys, ns):
            plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(3, -9), fontsize=6, color=c)
    plt.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed malignant fraction")
    plt.title("Pooled CV, 796 patients (numbers = patients per bin)", fontsize=9)
    plt.legend(frameon=False, fontsize=7, loc="upper left")
    save("figS4_pooled_calibration")
    return True


def figure_block(name, caption, label):
    if not (SUP / f"{name}.png").exists():
        return f"% {name} not available\n"
    return (f"\\begin{{nffigure}}\n\\includegraphics[width=0.85\\linewidth]{{{name}.png}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{nffigure}}\n")


# ----------------------------------------------------------------- main
def main():
    SUP.mkdir(parents=True, exist_ok=True)
    fig_training(); fig_roc(); fig_pooled_calibration()
    for nm in ["figS3_confusion", "figS5_leakage_unmatched"]:
        src = C.FIG_DIR / f"{nm}.png"
        if src.exists():
            shutil.copy(src, SUP / f"{nm}.png")

    pooled = read(C.TABLE_DIR / "table_pooled_cv_all796.csv")
    if pooled is not None:
        for _, r in pooled.iterrows():
            notes.append(f"Pooled {r.get('Calibration', '')}: AUC {r['ROC-AUC']} | Brier {r['Brier']} | "
                         f"BSS {r['Brier skill score']} | ECE {r['ECE (uniform)']} | slope {r['Calibration slope']} | "
                         f"intercept {r.get('Calibration intercept', '')}")

    tex = [r"""\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb}
% shrink a table only if it is wider than the text (graphicx only; no extra packages)
\newsavebox{\fitbox}
\newcommand{\fitwidth}[1]{\savebox{\fitbox}{#1}\ifdim\wd\fitbox>\linewidth\resizebox{\linewidth}{!}{\usebox{\fitbox}}\else\usebox{\fitbox}\fi}
\makeatletter
\renewcommand{\fnum@table}{\textbf{\tablename~\thetable}}
\renewcommand{\fnum@figure}{\textbf{\figurename~\thefigure}}
% non-floating table/figure environments (no extra packages): items stay in their section
\newenvironment{nftable}{\par\addvspace{8pt}\noindent\begin{minipage}{\linewidth}\centering\def\@captype{table}}{\end{minipage}\par\addvspace{8pt}}
\newenvironment{nffigure}{\par\addvspace{8pt}\noindent\begin{minipage}{\linewidth}\centering\def\@captype{figure}}{\end{minipage}\par\addvspace{8pt}}
\makeatother
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\title{Supplementary Information\\[4pt]\large Leakage-Free Patient-Level Evaluation of Thyroid Ultrasound Malignancy Prediction}
\date{}
\begin{document}
\maketitle
All tables and figures in this document were generated by \texttt{scripts/06\_supplementary.py} from the saved
outputs of the analysis pipeline; no value was entered by hand. Labels were derived from the histopathology report only;
cytology was not used to define the outcome. Inner AUC in the training summaries is the patient-level AUC on the inner
validation subset used for model selection (about nine benign patients per fold); it is noisy and is not a performance
estimate. Calibration metrics on the development OOF set are in-sample, because the maps were fitted on these predictions.
Holm adjustment in the DeLong table is applied across the five ensemble-versus-fold-model comparisons. In the matched
leakage experiment both arms are evaluated on identical frames of the same patients; in the unmatched experiment the two
protocols use different validation patients, so their difference mixes leakage with split composition. In the pooled and
development-only calibration analyses, the map applied to each fold was fitted on the other four folds; paired differences
evaluate both maps on the same resampled patients; nested-ensemble maps were fitted on four-model ensemble OOF predictions.

\section*{S1. Outcome definition and cohort}
""", table_labels(), table_folds(),
           r"\section*{S2. Training configuration}" + "\n", table_hparams(), table_training(),
           figure_block("figS1_training_curves", "Inner-validation patient-level AUC per epoch. Dotted lines mark the start of stage 2.", "fig:S_training"),
           r"\section*{S3. Development, test and pooled estimates}" + "\n", table_oof(),
           figure_block("figS2_roc_oof_test_pooled", "ROC curves on the development OOF set, the test cohort and the pooled cross-validation.", "fig:S_roc"),
           table_delong(),
           r"\section*{S4. Operating points}" + "\n",
           figure_block("figS3_confusion", "Test-cohort confusion matrices at the two operating points.",
                        "fig:S_confusion"),
           r"\section*{S5. Calibration}" + "\n",
           figure_block("figS4_pooled_calibration", "Reliability diagram, pooled cross-validation.",
                        "fig:S_pooled_cal"),
           r"\section*{S6. Leakage experiments}" + "\n",
           table_leak_matched(),
           table_leakage(),
           figure_block("figS5_leakage_unmatched", "Unmatched leakage experiment.",
                        "fig:S_leak_unmatched"),
           r"\section*{S7. Additional calibration analyses}" + "\n",
           table_cal_dev(), table_cal_diff(), table_cal_nested(),
           r"""\section*{S8. Reporting checklist}
The completed TRIPOD+AI checklist is provided as a separate file.

\end{document}
"""]
    (SUP / "supplementary.tex").write_text("\n".join(tex), encoding="utf-8")
    (SUP / "numbers_for_paper.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"\nWritten {SUP / 'supplementary.tex'}\n\nNumbers for the main text:\n" + "\n".join(notes))


if __name__ == "__main__":
    main()
