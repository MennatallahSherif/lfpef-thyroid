"""
STEP 03b - MATCHED leakage experiment (answers reviewer comment 1.1).

The first leakage experiment (step 03) compared two DIFFERENT validation sets, so its AUC
difference mixed leakage with split composition. Here both arms are evaluated on EXACTLY the
same images of the same patients:

  For each repeat r:
    V      = 20% of development patients (stratified), restricted to patients with >= 2 images
    V_leak = about half of each V patient's images  (at least 1)
    V_eval = the remaining images of each V patient  (at least 1)   <- evaluation set, both arms
    clean arm : train on images of patients NOT in V
    leaky arm : train on the same images + V_leak; to keep the training-set size identical,
                the same number of images is removed at random from the non-V patients
  Both arms: same seed, same epochs, no early stopping, evaluated on V_eval.

Two training regimes (frozen backbone; and fine-tuned backbone) because memorisation is
expected to be larger when the backbone can adapt.

Output: outputs/tables/leakage_matched_runs.csv  (one row per repeat x regime x arm)
        outputs/tables/table_leakage_matched_summary.csv (paired differences, mean, 95% CI)
Resumable: finished runs are skipped.

Usage:  python scripts/03b_leakage_matched.py               (5 repeats x 2 regimes x 2 arms)
        python scripts/03b_leakage_matched.py --repeats 10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config as C
from lfpef import datasets as D
from lfpef import metrics as M
from lfpef import train as T
from lfpef.models import build_frame_model, unfreeze_last
from lfpef.utils import base_parser, load_cfg, load_master

REGIMES = {
    "frozen":    {"stage1": 15, "stage2": 0},
    "finetuned": {"stage1": 10, "stage2": 10},
}


def make_split(dev, dev_pat, seed):
    rng = np.random.default_rng(seed)
    n_img = dev.groupby("patient_id").size()
    elig = dev_pat[dev_pat.patient_id.isin(n_img[n_img >= 2].index)]
    _, v_ids = train_test_split(elig.patient_id, test_size=0.2 * len(dev_pat) / len(elig),
                                random_state=seed, stratify=elig.y)
    v_ids = set(v_ids)
    leak_rows, eval_rows = [], []
    for pid, g in dev[dev.patient_id.isin(v_ids)].groupby("patient_id"):
        idx = rng.permutation(g.index.values)
        k = len(idx) // 2                      # >= 1 because every V patient has >= 2 images
        leak_rows += list(idx[:k]); eval_rows += list(idx[k:])
    base = dev[~dev.patient_id.isin(v_ids)]
    clean_train = base
    drop = rng.choice(base.index.values, size=len(leak_rows), replace=False)
    leaky_train = pd.concat([base.drop(index=drop), dev.loc[leak_rows]])
    return clean_train, leaky_train, dev.loc[eval_rows], v_ids


def run_arm(d_tr, d_eval, cfg, regime, seed, use_aux):
    keras.backend.clear_session()
    T.set_seed(seed)
    model, backbone = build_frame_model(cfg, use_aux)
    ds = D.frame_train_ds(d_tr, cfg, use_aux, seed)
    T.compile_frame(use_aux, cfg.TIRADS_LOSS_WEIGHT)(model, cfg.STAGE1_LR)
    model.fit(ds, epochs=regime["stage1"], verbose=2)
    if regime["stage2"] > 0:
        unfreeze_last(backbone, cfg.UNFREEZE_LAST)
        T.compile_frame(use_aux, cfg.TIRADS_LOSS_WEIGHT)(model, cfg.STAGE2_LR)
        model.fit(ds, epochs=regime["stage2"], verbose=2)
    z = T._logits(model, D.frame_predict_ds(d_eval, cfg))
    return d_eval[["patient_id", "file_name", "y"]].assign(logit=z)


def main():
    ap = base_parser("Matched leakage experiment")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--regimes", default="frozen,finetuned")
    args = ap.parse_args()
    cfg = load_cfg(C, quick=args.quick)
    if args.no_weights:
        cfg.BACKBONE_WEIGHTS = None
    if args.quick:
        for r in REGIMES.values():
            r["stage1"], r["stage2"] = 1, min(r["stage2"], 1)
    use_aux = cfg.USE_TIRADS_AUX
    C.ensure_dirs()
    df, pat = load_master(cfg)
    dev = df[df.cohort == "dev"].reset_index(drop=True)
    dev_pat = pat[pat.cohort == "dev"]
    pred_dir = cfg.PRED_DIR / "leakage_matched"; pred_dir.mkdir(parents=True, exist_ok=True)

    for r in range(args.repeats):
        seed = cfg.SEED + 1000 + r
        clean_tr, leaky_tr, d_eval, v_ids = make_split(dev, dev_pat, seed)
        for reg in args.regimes.split(","):
            for arm, d_tr in [("clean", clean_tr), ("leaky", leaky_tr)]:
                f = pred_dir / f"r{r}_{reg}_{arm}.csv"
                if f.exists():
                    continue
                print(f"\n--- repeat {r + 1}/{args.repeats} | {reg} | {arm} | train images={len(d_tr)} "
                      f"| eval patients={d_eval.patient_id.nunique()} images={len(d_eval)} ---")
                run_arm(d_tr, d_eval, cfg, REGIMES[reg], seed, use_aux).to_csv(f, index=False)

    # ------------------------------------------------ paired analysis
    rows = []
    for f in sorted(pred_dir.glob("r*_clean.csv")):
        r, reg = f.stem.split("_")[0][1:], f.stem.split("_")[1]
        g = pred_dir / f"r{r}_{reg}_leaky.csv"
        if not g.exists():
            continue
        a, b = pd.read_csv(g), pd.read_csv(f)          # a = leaky, b = clean, identical rows/order
        assert (a.file_name.values == b.file_name.values).all()
        pa = a.groupby("patient_id").agg(y=("y", "first"), z=("logit", "mean"))
        pb = b.groupby("patient_id").agg(y=("y", "first"), z=("logit", "mean")).loc[pa.index]
        a1, a2, d, (lo, hi), p = M.delong_test(pa.y.values, pa.z.values, pb.z.values)
        rows.append({"repeat": int(r) + 1, "regime": reg,
                     "img_auc_leaky": roc_auc_score(a.y, a.logit), "img_auc_clean": roc_auc_score(b.y, b.logit),
                     "pat_auc_leaky": a1, "pat_auc_clean": a2, "pat_auc_diff": d,
                     "pat_diff_ci": f"{lo:.3f} to {hi:.3f}", "pat_delong_p": p,
                     "pat_prauc_leaky": average_precision_score(pa.y, pa.z),
                     "pat_prauc_clean": average_precision_score(pb.y, pb.z),
                     "eval_patients": len(pa), "eval_benign": int((pa.y == 0).sum())})
    runs = pd.DataFrame(rows)
    if not len(runs):
        print("No completed pairs yet."); return
    runs["img_auc_diff"] = runs.img_auc_leaky - runs.img_auc_clean
    runs.to_csv(cfg.TABLE_DIR / "leakage_matched_runs.csv", index=False)

    summ = []
    for reg, g in runs.groupby("regime"):
        for col in ["img_auc_diff", "pat_auc_diff"]:
            v = g[col].values; n = len(v)
            m, sd = v.mean(), v.std(ddof=1) if n > 1 else np.nan
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            summ.append({"regime": reg, "quantity": col, "repeats": n, "mean paired diff (leaky - clean)": m,
                         "SD": sd, "95% CI": f"{m - half:.3f} to {m + half:.3f}",
                         "positive in": f"{(v > 0).sum()}/{n}",
                         "clean AUC mean (SD)": f"{g[col.replace('diff', 'clean')].mean():.3f} ({g[col.replace('diff', 'clean')].std(ddof=1):.3f})"})
    summ = pd.DataFrame(summ)
    summ.to_csv(cfg.TABLE_DIR / "table_leakage_matched_summary.csv", index=False)
    print("\nPER RUN\n", runs.round(3).to_string(index=False))
    print("\nSUMMARY (paired, same evaluation images)\n", summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
