"""
STEP 02 - AttnMIL baseline under the IDENTICAL protocol
(same folds, same inner validation, same two-stage schedule, same calibration in step 04).
This makes the comparison "architecture vs architecture", which is what a reviewer will accept.

Outputs: outputs/predictions/attnmil_<cohort>_fold<k>.csv  (patient-level logits)

Usage:  python scripts/02_train_attnmil_cv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import pandas as pd

import config as C
from lfpef import data as Dt
from lfpef import datasets as D
from lfpef import train as T
from lfpef.models import build_attnmil
from lfpef.utils import base_parser, load_cfg, load_master


def main():
    args = base_parser("AttnMIL 5-fold patient-level CV").parse_args()
    cfg = load_cfg(C, quick=args.quick)
    if args.no_weights:
        cfg.BACKBONE_WEIGHTS = None
    tag = args.tag or "attnmil"
    C.ensure_dirs()

    df, pat = load_master(cfg)
    t_max = int(df.groupby("patient_id").size().max())
    bags = D.make_bag_table(df, t_max).merge(pat[["patient_id", "cohort", "fold_dev", "fold_all"]], on="patient_id")
    fold_col = "fold_dev" if args.cohort == "dev" else "fold_all"
    b_test = bags[bags.cohort == "test"] if args.cohort == "dev" else bags.iloc[0:0]

    for k in [int(f) for f in args.folds.split(",")]:
        name = f"{tag}_{args.cohort}_fold{k}"
        out_csv = cfg.PRED_DIR / f"{name}.csv"
        if out_csv.exists():
            print(f"[skip] {name} already done")
            continue
        print(f"\n===== {name} (T_max={t_max}) =====")
        keras.backend.clear_session()
        T.set_seed(cfg.SEED + k)
        tr_ids, in_ids, out_ids = Dt.fold_split(pat, k, fold_col, cfg.INNER_VAL_FRAC, cfg.SEED)
        b_tr, b_in, b_out = (bags[bags.patient_id.isin(s)] for s in (tr_ids, in_ids, out_ids))

        model, backbone = build_attnmil(cfg, t_max)
        train_ds = D.bag_ds(b_tr, cfg, training=True, seed=cfg.SEED + k)
        cw = D.class_weights(b_tr["y"])

        def score_fn(m, _b=b_in):
            return _b["y"].values, T._logits(m, D.bag_ds(_b, cfg, training=False))

        hist, best = T.two_stage_fit(model, backbone, T.compile_mil, train_ds, score_fn, cfg,
                                     cfg.MIL_STAGE1_EPOCHS, cfg.MIL_STAGE2_EPOCHS, class_weight=cw)
        hist.to_csv(cfg.LOG_DIR / f"{name}_history.csv", index=False)

        def pred(b, role):
            z = T._logits(model, D.bag_ds(b, cfg, training=False))
            return pd.DataFrame({"patient_id": b.patient_id.values, "file_name": "", "y": b.y.values,
                                 "logit": z, "role": role})

        parts = [pred(b_out, "oof")]
        T.check_not_collapsed(parts[0]["logit"].values, name, cfg.COLLAPSE_MIN_STD)
        if len(b_test):
            parts.append(pred(b_test, "test"))
        out = pd.concat(parts).assign(model=tag, cohort=args.cohort, fold=k, inner_best_auc=best)
        model.save(cfg.MODEL_DIR / f"{name}.keras")
        out.to_csv(out_csv, index=False)
        print(f"saved {out_csv.name}")


if __name__ == "__main__":
    main()
