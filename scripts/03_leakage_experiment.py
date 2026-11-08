"""
STEP 03 - Quantifying leakage bias (repeated, not a single run).

For each of LEAK_REPEATS seeds, inside the DEVELOPMENT cohort only:
   A) image-level random 80/20 split (patients can appear in both sides)  -> leaky
   B) patient-level stratified 80/20 split                                -> leakage-free
Same model as step 01, same fixed epoch budget, NO early stopping on the evaluation set
(early stopping on the evaluated split would add its own optimism to both arms).
Reports image-level and patient-level AUC, and mean +/- SD of the inflation.

Usage:  python scripts/03_leakage_experiment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import keras
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import config as C
from lfpef import datasets as D
from lfpef import train as T
from lfpef.models import build_frame_model
from lfpef.utils import base_parser, load_cfg, load_master


def run_once(d_tr, d_va, cfg, seed, use_aux):
    keras.backend.clear_session()
    T.set_seed(seed)
    model, _ = build_frame_model(cfg, use_aux)
    T.compile_frame(use_aux, cfg.TIRADS_LOSS_WEIGHT)(model, cfg.STAGE1_LR)
    model.fit(D.frame_train_ds(d_tr, cfg, use_aux, seed), epochs=cfg.LEAK_EPOCHS, verbose=2)
    z = T._logits(model, D.frame_predict_ds(d_va, cfg))
    agg = pd.DataFrame({"pid": d_va.patient_id.values, "y": d_va.y.values, "z": z}).groupby("pid").agg(
        y=("y", "first"), z=("z", "mean"))
    overlap = set(d_tr.patient_id) & set(d_va.patient_id)
    return {"img_auc": roc_auc_score(d_va.y, z), "img_prauc": average_precision_score(d_va.y, z),
            "pat_auc": roc_auc_score(agg.y, agg.z), "pat_prauc": average_precision_score(agg.y, agg.z),
            "val_images": len(d_va), "val_patients": d_va.patient_id.nunique(),
            "overlap_patients": len(overlap), "overlap_ratio": len(overlap) / d_va.patient_id.nunique()}


def main():
    args = base_parser("Leakage experiment").parse_args()
    cfg = load_cfg(C, quick=args.quick)
    if args.no_weights:
        cfg.BACKBONE_WEIGHTS = None
    use_aux = cfg.USE_TIRADS_AUX
    C.ensure_dirs()
    df, pat = load_master(cfg)
    dev = df[df.cohort == "dev"].reset_index(drop=True)
    dev_pat = pat[pat.cohort == "dev"]
    out_csv = cfg.TABLE_DIR / "leakage_runs.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    rows = done.to_dict("records")
    for r in range(cfg.LEAK_REPEATS):
        seed = cfg.SEED + 100 * r
        for protocol in ["image-level split", "patient-level split"]:
            if len(done) and ((done.repeat == r) & (done.protocol == protocol)).any():
                continue
            if protocol == "image-level split":
                tr, va = train_test_split(dev, test_size=0.2, random_state=seed, stratify=dev.y)
            else:
                p_tr, p_va = train_test_split(dev_pat.patient_id, test_size=0.2, random_state=seed, stratify=dev_pat.y)
                tr, va = dev[dev.patient_id.isin(p_tr)], dev[dev.patient_id.isin(p_va)]
            print(f"\n--- repeat {r + 1}/{cfg.LEAK_REPEATS} | {protocol} ---")
            res = run_once(tr, va, cfg, seed, use_aux)
            rows.append({"repeat": r, "seed": seed, "protocol": protocol, **res})
            pd.DataFrame(rows).to_csv(out_csv, index=False)   # save after each run (resume support)

    runs = pd.DataFrame(rows)
    summ = runs.groupby("protocol")[["img_auc", "pat_auc", "img_prauc", "pat_prauc", "overlap_ratio"]].agg(["mean", "std"])
    wide = runs.pivot(index="repeat", columns="protocol", values="img_auc")
    infl = wide["image-level split"] - wide["patient-level split"]
    summ.to_csv(cfg.TABLE_DIR / "table_leakage_summary.csv")
    print("\n", summ.round(3).to_string())
    print(f"\nImage-level AUC inflation: {infl.mean():.3f} +/- {infl.std():.3f} "
          f"(positive in {(infl > 0).sum()}/{len(infl)} repeats)")


if __name__ == "__main__":
    main()
