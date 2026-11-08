"""
STEP 04 - Patient-level evaluation (no training; runs in seconds).

  1. image logits -> patient logits (mean of logits, as written in the paper, Eq. 6)
  2. OOF patient logits (dev)  -> global temperature T*  -> calibrated OOF probabilities
  3. thresholds from calibrated OOF: Youden  and  highest threshold with sensitivity >= 0.90
  4. test: average the 5 fold models' image logits -> patient mean -> apply T* -> apply thresholds
  5. ablation (single model -> 5-fold ensemble -> + temperature), aggregation variants
  6. AttnMIL under the same steps; paired DeLong tests
  7. stratified bootstrap CIs (benign/malignant resampled separately), Wilson CIs for sens/spec

Usage:
  python scripts/04_evaluate.py                 # official split (main results)
  python scripts/04_evaluate.py --cohort all    # pooled CV, cross-fitted calibration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from lfpef import metrics as M
from lfpef.utils import load_cfg


# ------------------------------------------------------------------ loading
def load_preds(cfg, tag, cohort):
    files = sorted(cfg.PRED_DIR.glob(f"{tag}_{cohort}_fold*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    folds = sorted(df.fold.unique())
    if len(folds) != cfg.N_FOLDS:
        print(f"WARNING: {tag} has only folds {folds} - results are provisional.")
    return df


def patient_logits(img, how="mean_logit"):
    g = img.groupby("patient_id")
    if how == "mean_logit":
        z = g["logit"].mean()
    elif how == "mean_prob":
        z = g["logit"].apply(lambda s: M.logit(M.sigmoid(s).mean()))
    elif how == "max_logit":
        z = g["logit"].max()
    else:
        raise ValueError(how)
    return pd.DataFrame({"y": g["y"].first(), "z": z}).reset_index()


def oof_patients(pred, how="mean_logit"):
    o = pred[pred.role == "oof"]
    return pd.concat([patient_logits(g, how).assign(fold=k) for k, g in o.groupby("fold")], ignore_index=True)


def test_patients(pred, how="mean_logit", folds=None):
    t = pred[pred.role == "test"]
    if folds is not None:
        t = t[t.fold.isin(folds)]
    key = ["patient_id", "file_name"]
    ens = t.groupby(key).agg(y=("y", "first"), logit=("logit", "mean")).reset_index()   # ensemble = mean of logits
    return patient_logits(ens, how)


# ------------------------------------------------------------------ helpers
def calibrate(oof, test=None):
    T = M.fit_temperature(oof.z, oof.y)
    oof = oof.assign(p=M.sigmoid(oof.z / T))
    if test is not None:
        test = test.assign(p=M.sigmoid(test.z / T))
    return T, oof, test


def crossfit_calibrate(oof):
    """Pooled analysis: T for fold k is fitted on the OTHER folds (no in-sample calibration)."""
    parts, Ts = [], []
    for k, g in oof.groupby("fold"):
        rest = oof[oof.fold != k]
        T = M.fit_temperature(rest.z, rest.y)
        Ts.append(T)
        parts.append(g.assign(p=M.sigmoid(g.z / T)))
    return float(np.mean(Ts)), pd.concat(parts)


def crossfit_platt(oof):
    """Pooled analysis: Platt (a, b) for fold k fitted on the OTHER folds."""
    parts, ab = [], []
    for k, g in oof.groupby("fold"):
        rest = oof[oof.fold != k]
        a, b = M.fit_platt(rest.z, rest.y)
        ab.append((a, b))
        parts.append(g.assign(p=M.sigmoid(a + b * g.z)))
    return tuple(np.mean(ab, axis=0)), pd.concat(parts)


def fmt_ci(d):
    return {k: (f"{v['value']:.3f} ({v['ci_low']:.3f} to {v['ci_high']:.3f})" if np.isfinite(v["ci_low"])
                else f"{v['value']:.3f}") for k, v in d.items()}


def boot_fn(y, p):
    out = M.prob_metrics(y, p)
    return {k: out[k] for k in ["ROC-AUC", "PR-AUC", "Brier", "Brier skill score", "ECE (uniform)", "Calibration slope"]}


def to_latex(df, path, caption):
    try:
        df.to_latex(path, index=False, escape=True, caption=caption, float_format="%.3f")
    except Exception as e:  # jinja2 missing etc.
        print(f"(LaTeX skipped for {path.name}: {e})")


# ------------------------------------------------------------------ main (official split)
def run_dev(cfg, frame_tag):
    frame = load_preds(cfg, frame_tag, "dev")
    assert frame is not None, f"No predictions for '{frame_tag}'. Run step 01 first."
    mil = load_preds(cfg, "attnmil", "dev")
    noaux = load_preds(cfg, "frame_noaux", "dev") if frame_tag != "frame_noaux" else None
    folds = sorted(frame.fold.unique())

    # ---------------- LF-PEF
    oof = oof_patients(frame)
    tst = test_patients(frame)
    T, oof, tst = calibrate(oof, tst)
    thr_y = M.youden_threshold(oof.y, oof.p)
    thr_s = M.min_sens_threshold(oof.y, oof.p, cfg.MIN_SENS)
    print(f"LF-PEF  T* = {T:.4f} | thresholds from OOF: Youden = {thr_y:.4f}, Sens>={cfg.MIN_SENS} = {thr_s:.4f}")

    a_pl, b_pl = M.fit_platt(oof.z, oof.y)
    oof_pl = oof.assign(p=M.sigmoid(a_pl + b_pl * oof.z))
    tst_pl = tst.assign(p=M.sigmoid(a_pl + b_pl * tst.z))
    print(f"Platt (from OOF): a = {a_pl:.3f}, b = {b_pl:.3f}")

    scores = {"LF-PEF": tst, "LF-PEF (Platt)": tst_pl}
    # ---------------- AttnMIL (same protocol)
    if mil is not None:
        m_oof = mil[mil.role == "oof"][["patient_id", "y", "logit", "fold"]].rename(columns={"logit": "z"})
        m_tst = (mil[mil.role == "test"].groupby("patient_id").agg(y=("y", "first"), z=("logit", "mean")).reset_index())
        T_m, m_oof, m_tst = calibrate(m_oof, m_tst)
        scores["AttnMIL"] = m_tst
        mthr_y = M.youden_threshold(m_oof.y, m_oof.p)
        mthr_s = M.min_sens_threshold(m_oof.y, m_oof.p, cfg.MIN_SENS)

    # ---------------- ablation (test set, all patient-level)
    abl = []
    def add(name, y, p, note=""):
        r = M.prob_metrics(y, p); abl.append({"Variant": name, **{k: r[k] for k in
              ["ROC-AUC", "PR-AUC", "Brier", "Brier skill score", "ECE (uniform)", "Calibration slope"]}, "Note": note})

    single = []
    for k in folds:
        o_k = oof[oof.fold == k][["patient_id", "y", "z"]]
        t_k = test_patients(frame, folds=[k])
        single.append(roc_auc_score(t_k.y, t_k.z))
        if k == folds[0]:
            add(f"(1) Single model (fold {k}), uncalibrated", t_k.y, M.sigmoid(t_k.z))
            T_k, _, t_kc = calibrate(o_k, t_k)
            add(f"(2) Single model (fold {k}) + temperature", t_k.y, t_kc.p, f"T fitted on its outer fold, T={T_k:.3f}")
    add("(3) 5-fold ensemble, uncalibrated", tst.y, M.sigmoid(tst.z))
    add("(4) 5-fold ensemble + temperature (LF-PEF)", tst.y, tst.p, f"T*={T:.3f} from OOF")
    add("(4d) 5-fold ensemble + Platt (intercept + slope)", tst.y, tst_pl.p, f"a={a_pl:.3f}, b={b_pl:.3f} from OOF")
    for how in ["mean_prob", "max_logit"]:
        o_h, t_h = oof_patients(frame, how), test_patients(frame, how)
        _, _, t_hc = calibrate(o_h, t_h)
        add(f"(4b) LF-PEF with {how} aggregation", t_hc.y, t_hc.p)
    if noaux is not None:
        o_n, t_n = oof_patients(noaux), test_patients(noaux)
        _, _, t_nc = calibrate(o_n, t_n)
        add("(4c) LF-PEF without TI-RADS auxiliary head", t_nc.y, t_nc.p)
        scores["LF-PEF (no TI-RADS aux)"] = t_nc
    if mil is not None:
        add("(5) AttnMIL ensemble + temperature (same protocol)", m_tst.y, m_tst.p, f"T={T_m:.3f}")
    agg_rows = []
    for how in ["mean_logit", "mean_prob", "max_logit"]:
        o_h, t_h = oof_patients(frame, how), test_patients(frame, how)
        agg_rows.append({"Aggregation": how, "OOF ROC-AUC (dev, n=%d)" % len(o_h): roc_auc_score(o_h.y, o_h.z),
                         "Test ROC-AUC": roc_auc_score(t_h.y, t_h.z)})
    agg_tab = pd.DataFrame(agg_rows)
    agg_tab.to_csv(cfg.TABLE_DIR / "table_aggregation_oof_vs_test.csv", index=False)
    print("\nAGGREGATION - choose on the OOF column, never on the test column\n", agg_tab.round(3).to_string(index=False))
    abl = pd.DataFrame(abl)
    abl.loc[len(abl)] = {"Variant": "Single fold models: test AUC mean (SD)",
                         "ROC-AUC": np.mean(single), "Note": f"SD={np.std(single, ddof=1):.3f}; range {min(single):.3f}-{max(single):.3f}"}
    abl.to_csv(cfg.TABLE_DIR / "table_ablation_test.csv", index=False)
    to_latex(abl.drop(columns="Note"), cfg.TABLE_DIR / "table_ablation_test.tex", "Ablation on the independent test cohort.")
    print("\nABLATION (test)\n", abl.round(3).to_string(index=False))

    # ---------------- main table with bootstrap CIs
    y = tst.y.values
    aligned = {n: s.set_index("patient_id").loc[tst.patient_id, "p"].values for n, s in scores.items()}
    boot = M.stratified_bootstrap(y, aligned, boot_fn, cfg.N_BOOT, cfg.SEED)
    main_rows, thr_rows = [], []
    for n, p in aligned.items():
        pm = M.prob_metrics(y, p)
        ci = M.ci_table({k: pm[k] for k in boot.metric.unique()}, boot, n)
        main_rows.append({"Model": n, **fmt_ci(ci),
                          "Brier (prevalence-only ref.)": f"{pm['Brier (prevalence-only reference)']:.3f}",
                          "PR-AUC no-skill": f"{pm['PR-AUC (no-skill = prevalence)']:.3f}",
                          "Calibration intercept": f"{pm['Calibration intercept (CITL)']:.3f}",
                          "ECE (equal-mass)": f"{pm['ECE (equal-mass)']:.3f}"})
    for n, (ty, ts, oo) in {"LF-PEF": (thr_y, thr_s, oof),
                            "LF-PEF (Platt)": (M.youden_threshold(oof_pl.y, oof_pl.p),
                                               M.min_sens_threshold(oof_pl.y, oof_pl.p, cfg.MIN_SENS), oof_pl),
                            **({"AttnMIL": (mthr_y, mthr_s, m_oof)} if mil is not None else {})}.items():
        p = aligned[n]
        for lab, t in [("Youden (from OOF)", ty), (f"Sensitivity >= {cfg.MIN_SENS} (from OOF)", ts)]:
            thr_rows.append({"Model": n, "Operating point": lab, "Cohort": "OOF (dev)", **M.threshold_metrics(oo.y, oo.p, t)})
            thr_rows.append({"Model": n, "Operating point": lab, "Cohort": "Test", **M.threshold_metrics(y, p, t)})
    main = pd.DataFrame(main_rows); thr = pd.DataFrame(thr_rows)
    main.to_csv(cfg.TABLE_DIR / "table_main_test_with_ci.csv", index=False)
    thr.to_csv(cfg.TABLE_DIR / "table_thresholds.csv", index=False)
    to_latex(main, cfg.TABLE_DIR / "table_main_test_with_ci.tex", "Patient-level test performance (95% bootstrap CI).")
    to_latex(thr.round(3), cfg.TABLE_DIR / "table_thresholds.tex", "Operating points derived from OOF predictions.")
    print("\nMAIN (test, 95% CI)\n", main.to_string(index=False))
    print("\nTHRESHOLDS\n", thr.round(3).to_string(index=False))

    # ---------------- OOF (dev) metrics: same quantities on the 587 development patients
    oof_rows = [{"Model": "LF-PEF (OOF, calibrated in-sample)", **M.prob_metrics(oof.y, oof.p)},
                {"Model": "LF-PEF Platt (OOF, calibrated in-sample)", **M.prob_metrics(oof_pl.y, oof_pl.p)}]
    if mil is not None:
        oof_rows.append({"Model": "AttnMIL (OOF, calibrated in-sample)", **M.prob_metrics(m_oof.y, m_oof.p)})
    pd.DataFrame(oof_rows).to_csv(cfg.TABLE_DIR / "table_oof_dev.csv", index=False)

    # ---------------- paired DeLong tests
    tests = []
    pairs = []
    for k in folds:      # ensemble vs EVERY single fold model (not only fold 1, which may be the weakest)
        tk = test_patients(frame, folds=[k]).set_index("patient_id").loc[tst.patient_id]
        pairs.append(("LF-PEF", f"Single model (fold {k})", aligned["LF-PEF"], tk.z.values))
    if mil is not None:
        pairs.append(("LF-PEF", "AttnMIL", aligned["LF-PEF"], aligned["AttnMIL"]))
    if "LF-PEF (no TI-RADS aux)" in aligned:
        pairs.append(("LF-PEF", "LF-PEF (no TI-RADS aux)", aligned["LF-PEF"], aligned["LF-PEF (no TI-RADS aux)"]))
    for a, b, sa, sb in pairs:
        a1, a2, d, (lo, hi), p = M.delong_test(y, sa, sb)
        tests.append({"Cohort": "Test", "A": a, "B": b, "AUC A": a1, "AUC B": a2, "Diff": d,
                      "Diff 95% CI": f"{lo:.3f} to {hi:.3f}", "p (DeLong)": p})
    if "LF-PEF (no TI-RADS aux)" in aligned and mil is not None:
        a1, a2, d, (lo, hi), p = M.delong_test(y, aligned["LF-PEF (no TI-RADS aux)"], aligned["AttnMIL"])
        tests.append({"Cohort": "Test", "A": "LF-PEF (no TI-RADS aux)", "B": "AttnMIL", "AUC A": a1, "AUC B": a2,
                      "Diff": d, "Diff 95% CI": f"{lo:.3f} to {hi:.3f}", "p (DeLong)": p})
    # same comparisons on the 587 development OOF patients (one prediction per patient per model)
    oof_sets = {"LF-PEF": oof.set_index("patient_id")[["y", "z"]]}
    if mil is not None:
        oof_sets["AttnMIL"] = m_oof.set_index("patient_id")[["y", "z"]]
    if noaux is not None:
        oof_sets["LF-PEF (no TI-RADS aux)"] = oof_patients(noaux).set_index("patient_id")[["y", "z"]]
    for a_name, b_name in [("LF-PEF", "AttnMIL"), ("LF-PEF", "LF-PEF (no TI-RADS aux)"),
                           ("LF-PEF (no TI-RADS aux)", "AttnMIL")]:
        if a_name in oof_sets and b_name in oof_sets:
            oo = oof_sets[a_name].join(oof_sets[b_name][["z"]], rsuffix="_b", how="inner")
            a1, a2, d, (lo, hi), p = M.delong_test(oo.y.values, oo.z.values, oo.z_b.values)
            tests.append({"Cohort": "OOF (dev)", "A": a_name, "B": b_name, "AUC A": a1, "AUC B": a2, "Diff": d,
                          "Diff 95% CI": f"{lo:.3f} to {hi:.3f}", "p (DeLong)": p})
    tests = pd.DataFrame(tests)
    # Holm correction across the 5 ensemble-vs-single comparisons
    m = tests.B.str.startswith("Single model")
    pv = tests.loc[m, "p (DeLong)"].values
    order = np.argsort(pv); holm = np.empty_like(pv); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(pv) - r) * pv[i])); holm[i] = run
    tests["p (Holm, single-model family)"] = np.nan
    tests.loc[m, "p (Holm, single-model family)"] = holm
    tests.to_csv(cfg.TABLE_DIR / "table_delong.csv", index=False)
    print("\nDeLong (test)\n", tests.round(4).to_string(index=False))

    # ---------------- calibration: paired differences, development-only check, nested ensemble
    cal_fns = {"Brier": M.brier, "ECE (uniform)": lambda yy, pp: M.ece(yy, pp, cfg.ECE_BINS), "NLL": M.nll}
    diffs = []
    d_test = M.paired_bootstrap_diff(tst.y, tst.p, tst_pl.p, cal_fns, cfg.N_BOOT, cfg.SEED)
    diffs.append(d_test.assign(cohort="Test (209)", comparison="temperature minus Platt"))
    # development-only confirmation: cross-fitted over the 5 development folds, test cohort never used
    dev_oof = oof[["patient_id", "y", "z", "fold"]].copy()
    T_cf, dev_t = crossfit_calibrate(dev_oof)
    (a_cf, b_cf), dev_p = crossfit_platt(dev_oof)
    dev_p = dev_p.set_index("patient_id").loc[dev_t.patient_id].reset_index()
    dev_rows = []
    for cal, o in [("temperature (cross-fitted, dev only)", dev_t), ("Platt (cross-fitted, dev only)", dev_p)]:
        boot = M.stratified_bootstrap(o.y.values, {cal: o.p.values}, boot_fn, cfg.N_BOOT, cfg.SEED)
        pm = M.prob_metrics(o.y, o.p)
        ci = M.ci_table({k: pm[k] for k in boot.metric.unique()}, boot, cal)
        dev_rows.append({"Calibration": cal, "Patients": len(o), "Benign": int((o.y == 0).sum()), **fmt_ci(ci),
                         "Calibration intercept": f"{pm['Calibration intercept (CITL)']:.3f}",
                         "Brier (prevalence-only ref.)": f"{pm['Brier (prevalence-only reference)']:.3f}"})
    pd.DataFrame(dev_rows).to_csv(cfg.TABLE_DIR / "table_calibration_dev_crossfit.csv", index=False)
    print("\nCALIBRATION, development cohort only (cross-fitted; test never used)\n",
          pd.DataFrame(dev_rows).to_string(index=False))
    d_dev = M.paired_bootstrap_diff(dev_t.y, dev_t.p, dev_p.p, cal_fns, cfg.N_BOOT, cfg.SEED)
    diffs.append(d_dev.assign(cohort="Development OOF (587), cross-fitted", comparison="temperature minus Platt"))

    nested_files = sorted((cfg.PRED_DIR / "nested").glob("outer*_inner*.csv")) if (cfg.PRED_DIR / "nested").exists() else []
    if len(nested_files) >= 20:
        nst = pd.concat([pd.read_csv(f) for f in nested_files])
        ens = nst.groupby(["patient_id", "file_name"]).agg(y=("y", "first"), logit=("logit", "mean"),
                                                           outer=("outer", "first")).reset_index()
        n_oof = patient_logits(ens).merge(ens.groupby("patient_id").outer.first().rename("fold").reset_index())
        T_n = M.fit_temperature(n_oof.z, n_oof.y)
        a_n, b_n = M.fit_platt(n_oof.z, n_oof.y)
        nested_scores = {"Temperature fitted on nested-ensemble OOF": M.sigmoid(tst.z / T_n),
                         "Platt fitted on nested-ensemble OOF": M.sigmoid(a_n + b_n * tst.z),
                         "Platt fitted on single-model OOF (main analysis)": tst_pl.p.values}
        boot = M.stratified_bootstrap(y, nested_scores, boot_fn, cfg.N_BOOT, cfg.SEED)
        n_rows = []
        for nme, pp in nested_scores.items():
            pm = M.prob_metrics(y, pp)
            ci = M.ci_table({k: pm[k] for k in boot.metric.unique()}, boot, nme)
            n_rows.append({"Calibration map": nme, **fmt_ci(ci),
                           "Calibration intercept": f"{pm['Calibration intercept (CITL)']:.3f}"})
        n_tab = pd.DataFrame(n_rows)
        n_tab["fitted parameters"] = [f"T={T_n:.3f}", f"a={a_n:.3f}, b={b_n:.3f}", f"a={a_pl:.3f}, b={b_pl:.3f}"]
        n_tab.to_csv(cfg.TABLE_DIR / "table_calibration_nested.csv", index=False)
        # independent re-computation check of calibration-in-the-large (4 decimals)
        chk = {"Temperature (single-model OOF fit)": tst.p.values, **nested_scores}
        print("\nCITL CHECK (intercept, slope; 4 decimals; each computed independently)")
        for nme, pp in chk.items():
            a_, b_ = M.calibration_slope_intercept(y, pp)
            print(f"   {nme:52s} intercept={a_:.4f}  slope={b_:.4f}  mean p={np.mean(pp):.4f}")
        print(f"\nNESTED-ENSEMBLE CALIBRATION (nested OOF ROC-AUC = {roc_auc_score(n_oof.y, n_oof.z):.3f})\n",
              n_tab.to_string(index=False))
        d_n = M.paired_bootstrap_diff(y, tst_pl.p.values, nested_scores["Platt fitted on nested-ensemble OOF"],
                                      cal_fns, cfg.N_BOOT, cfg.SEED)
        diffs.append(d_n.assign(cohort="Test (209)", comparison="single-model-OOF Platt minus nested-ensemble Platt"))
        scores["LF-PEF (Platt, nested)"] = tst.assign(p=nested_scores["Platt fitted on nested-ensemble OOF"])
    elif nested_files:
        print(f"\n(nested calibration: only {len(nested_files)}/20 models finished - skipped)")
    diffs = pd.concat(diffs, ignore_index=True)
    diffs.to_csv(cfg.TABLE_DIR / "table_calibration_paired_diff.csv", index=False)
    print("\nPAIRED CALIBRATION DIFFERENCES (positive = second map better for Brier/ECE/NLL)\n",
          diffs.round(4).to_string(index=False))

    # ---------------- save patient-level scores for the figures
    out = [oof.assign(role="oof", model="LF-PEF")[["patient_id", "y", "z", "p", "role", "model"]]]
    for n, s in scores.items():
        out.append(s.assign(role="test", model=n)[["patient_id", "y", "z", "p", "role", "model"]])
    if mil is not None:
        out.append(m_oof.assign(role="oof", model="AttnMIL")[["patient_id", "y", "z", "p", "role", "model"]])
    pd.concat(out).to_csv(cfg.PRED_DIR / "patient_scores_dev.csv", index=False)
    pd.DataFrame([{"T_lfpef": T, "thr_youden": thr_y, "thr_sens": thr_s, "min_sens": cfg.MIN_SENS}]).to_csv(
        cfg.TABLE_DIR / "operating_params.csv", index=False)
    print(f"\nAll tables in {cfg.TABLE_DIR}")


# ------------------------------------------------------------------ pooled analysis
def run_all(cfg, frame_tag):
    rows = []
    for tag in [frame_tag, "attnmil"]:
        pred = load_preds(cfg, tag, "all")
        if pred is None:
            continue
        oof = (oof_patients(pred) if tag != "attnmil" else
               pred[pred.role == "oof"][["patient_id", "y", "logit", "fold"]].rename(columns={"logit": "z"}))
        T, oof_t = crossfit_calibrate(oof)
        (a, b), oof_p = crossfit_platt(oof)
        for cal, o, par in [("temperature (cross-fitted)", oof_t, f"mean T={T:.3f}"),
                            ("Platt (cross-fitted)", oof_p, f"mean a={a:.3f}, b={b:.3f}")]:
            name = f"{tag} | {cal}"
            boot = M.stratified_bootstrap(o.y.values, {name: o.p.values}, boot_fn, cfg.N_BOOT, cfg.SEED)
            pm = M.prob_metrics(o.y, o.p)
            ci = M.ci_table({k: pm[k] for k in boot.metric.unique()}, boot, name)
            rows.append({"Model": tag, "Calibration": cal, "Patients": len(o), "Benign": int((o.y == 0).sum()),
                         "Parameters": par, **fmt_ci(ci),
                         "Calibration intercept": f"{pm['Calibration intercept (CITL)']:.3f}",
                         "Brier (prevalence-only ref.)": f"{pm['Brier (prevalence-only reference)']:.3f}"})
        cal_fns = {"Brier": M.brier, "ECE (uniform)": lambda yy, pp: M.ece(yy, pp, cfg.ECE_BINS), "NLL": M.nll}
        op = oof_p.set_index("patient_id").loc[oof_t.patient_id]
        dd = M.paired_bootstrap_diff(oof_t.y, oof_t.p, op.p.values, cal_fns, cfg.N_BOOT, cfg.SEED)
        dd.assign(model=tag, comparison="temperature minus Platt (cross-fitted, pooled)").to_csv(
            cfg.TABLE_DIR / f"table_pooled_paired_diff_{tag}.csv", index=False)
        print(f"\nPooled paired differences ({tag})\n", dd.round(4).to_string(index=False))
        oof_t.assign(model=tag, calibration="temperature").to_csv(cfg.PRED_DIR / f"patient_scores_all_{tag}.csv", index=False)
        oof_p.assign(model=tag, calibration="platt").to_csv(cfg.PRED_DIR / f"patient_scores_all_{tag}_platt.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(cfg.TABLE_DIR / "table_pooled_cv_all796.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="dev", choices=["dev", "all"])
    ap.add_argument("--frame-tag", default="frame")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(C, quick=a.quick)
    C.ensure_dirs()
    (run_dev if a.cohort == "dev" else run_all)(cfg, a.frame_tag)
