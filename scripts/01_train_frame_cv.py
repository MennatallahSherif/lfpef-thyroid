"""
STEP 01 - Frame-level CNN, patient-level 5-fold CV  (the model behind LF-PEF).

For every outer fold k:
   train  = folds != k   minus an inner-validation subset (15 % of those patients)
   inner  = early stopping + checkpoint selection (patient-level AUC)
   outer  = fold k  -> OOF image logits
   test   = independent test cohort -> image logits from this fold model (ensembled in step 04)

Outputs: outputs/predictions/<tag>_<cohort>_fold<k>.csv   (image-level logits)
         outputs/models/<tag>_<cohort>_fold<k>.keras
         outputs/logs/<tag>_<cohort>_fold<k>_history.csv
Already finished folds are skipped, so you can stop and resume (important on CPU).

Usage examples:
   python scripts/01_train_frame_cv.py                       # main model (TI-RADS aux on)
   python scripts/01_train_frame_cv.py --no-aux --tag frame_noaux   # ablation: no TI-RADS head
   python scripts/01_train_frame_cv.py --cohort all          # pooled CV over all 796 patients
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
from lfpef.models import build_frame_model
from lfpef.utils import base_parser, load_cfg, load_master


def predict_images(model, df, cfg):
    z = T._logits(model, D.frame_predict_ds(df, cfg))
    return df[["patient_id", "file_name", "y"]].assign(logit=z)


def main():
    ap = base_parser("Frame CNN 5-fold patient-level CV")
    ap.add_argument("--no-aux", action="store_true", help="disable the TI-RADS auxiliary head")
    args = ap.parse_args()
    cfg = load_cfg(C, quick=args.quick)
    if args.no_weights:
        cfg.BACKBONE_WEIGHTS = None
    use_aux = cfg.USE_TIRADS_AUX and not args.no_aux
    tag = args.tag or ("frame" if use_aux else "frame_noaux")
    C.ensure_dirs()

    df, pat = load_master(cfg)
    fold_col = "fold_dev" if args.cohort == "dev" else "fold_all"
    df_test = df[df.cohort == "test"] if args.cohort == "dev" else df.iloc[0:0]
    if use_aux and (df["tirads_grp"] < 0).any():
        raise ValueError("Some images have no TI-RADS; run with --no-aux or fix the data.")

    for k in [int(f) for f in args.folds.split(",")]:
        name = f"{tag}_{args.cohort}_fold{k}"
        out_csv = cfg.PRED_DIR / f"{name}.csv"
        if out_csv.exists():
            print(f"[skip] {name} already done")
            continue
        print(f"\n===== {name} =====")
        keras.backend.clear_session()
        T.set_seed(cfg.SEED + k)
        tr_ids, in_ids, out_ids = Dt.fold_split(pat, k, fold_col, cfg.INNER_VAL_FRAC, cfg.SEED)
        d_tr, d_in, d_out = (df[df.patient_id.isin(s)] for s in (tr_ids, in_ids, out_ids))
        assert not (tr_ids & in_ids or tr_ids & out_ids or in_ids & out_ids), "patient overlap!"
        print(f"patients  train={len(tr_ids)}  inner-val={len(in_ids)}  outer(OOF)={len(out_ids)}"
              f"  | benign train/inner/outer = {(d_tr.groupby('patient_id').y.first() == 0).sum()}/"
              f"{(d_in.groupby('patient_id').y.first() == 0).sum()}/{(d_out.groupby('patient_id').y.first() == 0).sum()}")

        model, backbone = build_frame_model(cfg, use_aux)
        train_ds = D.frame_train_ds(d_tr, cfg, use_aux, seed=cfg.SEED + k)

        def score_fn(m, _d=d_in):
            _, y, s = T.frame_patient_scores(m, _d, cfg)
            return y, s

        hist, best = T.two_stage_fit(model, backbone, T.compile_frame(use_aux, cfg.TIRADS_LOSS_WEIGHT),
                                     train_ds, score_fn, cfg, cfg.STAGE1_EPOCHS, cfg.STAGE2_EPOCHS)
        hist.to_csv(cfg.LOG_DIR / f"{name}_history.csv", index=False)

        oof = predict_images(model, d_out, cfg).assign(role="oof")
        T.check_not_collapsed(oof["logit"].values, name, cfg.COLLAPSE_MIN_STD)
        parts = [oof]
        if len(df_test):
            parts.append(predict_images(model, df_test, cfg).assign(role="test"))
        pred = pd.concat(parts).assign(model=tag, cohort=args.cohort, fold=k, inner_best_auc=best)
        model.save(cfg.MODEL_DIR / f"{name}.keras")
        pred.to_csv(out_csv, index=False)          # written last = marker that the fold is complete
        print(f"saved {out_csv.name}")


if __name__ == "__main__":
    main()
