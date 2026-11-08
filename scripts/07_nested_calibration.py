"""
STEP 07 - Nested ensemble predictions for calibration (answers reviewer comment 1.7).

Problem: calibration maps were fitted on SINGLE-model OOF logits but applied to the
FIVE-model averaged logits on the test cohort. Averaging shrinks logits, so the maps did not
match (test calibration slope about 2).

Fix: build ensemble-type OOF predictions for every development patient.
  For each outer fold k (patients F_k):
      for each inner fold j != k:
          train a model on the development folds other than {k, j}
          (with its own inner-validation subset for early stopping, exactly as in step 01)
          predict the frames of F_k
      ensemble OOF logit of a patient in F_k = mean over the 4 inner models of frame logits,
      then mean over frames.
  -> 20 models; every patient in F_k is scored by 4 models that never saw it, so the
     calibration map can be fitted on predictions with the same "averaged-ensemble" spread
     as the test predictions. (4-model ensembles trained on 3/5 of the data are a close but
     not perfect match to the 5-model test ensemble trained on 4/5; this is reported.)

Outputs: outputs/predictions/nested/outer{k}_inner{j}.csv  (frame logits for F_k)
Step 04 picks these up automatically. Resumable.

Usage:  python scripts/07_nested_calibration.py
        python scripts/07_nested_calibration.py --outer 1,2      (run part of it)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
from lfpef import datasets as D
from lfpef import train as T
from lfpef.models import build_frame_model
from lfpef.utils import base_parser, load_cfg, load_master


def main():
    ap = base_parser("Nested ensemble OOF for calibration")
    ap.add_argument("--outer", default="1,2,3,4,5")
    args = ap.parse_args()
    cfg = load_cfg(C, quick=args.quick)
    if args.no_weights:
        cfg.BACKBONE_WEIGHTS = None
    use_aux = cfg.USE_TIRADS_AUX
    C.ensure_dirs()
    out_dir = cfg.PRED_DIR / "nested"; out_dir.mkdir(parents=True, exist_ok=True)
    df, pat = load_master(cfg)
    dev_pat = pat[pat.cohort == "dev"]
    folds = sorted(dev_pat.fold_dev.unique())

    for k in [int(x) for x in args.outer.split(",")]:
        outer_ids = set(dev_pat.loc[dev_pat.fold_dev == k, "patient_id"])
        d_outer = df[df.patient_id.isin(outer_ids)]
        for j in [f for f in folds if f != k]:
            f_out = out_dir / f"outer{k}_inner{j}.csv"
            if f_out.exists():
                print(f"[skip] outer {k} inner {j}"); continue
            pool = dev_pat[~dev_pat.fold_dev.isin([k, j])]
            tr_ids, in_ids = train_test_split(pool.patient_id.values, test_size=cfg.INNER_VAL_FRAC,
                                              random_state=cfg.SEED + 10 * k + j, stratify=pool.y.values)
            tr_ids, in_ids = set(tr_ids), set(in_ids)
            assert not (tr_ids & outer_ids or in_ids & outer_ids), "outer fold leaked into training!"
            d_tr, d_in = df[df.patient_id.isin(tr_ids)], df[df.patient_id.isin(in_ids)]
            print(f"\n===== nested: outer {k}, inner {j} | train={len(tr_ids)} inner-val={len(in_ids)} "
                  f"outer={len(outer_ids)} =====")
            keras.backend.clear_session()
            T.set_seed(cfg.SEED + 10 * k + j)
            model, backbone = build_frame_model(cfg, use_aux)

            def score_fn(m, _d=d_in):
                _, y, s = T.frame_patient_scores(m, _d, cfg)
                return y, s

            T.two_stage_fit(model, backbone, T.compile_frame(use_aux, cfg.TIRADS_LOSS_WEIGHT),
                            D.frame_train_ds(d_tr, cfg, use_aux, seed=cfg.SEED + 10 * k + j),
                            score_fn, cfg, cfg.STAGE1_EPOCHS, cfg.STAGE2_EPOCHS)
            z = T._logits(model, D.frame_predict_ds(d_outer, cfg))
            T.check_not_collapsed(z, f"nested {k}-{j}", cfg.COLLAPSE_MIN_STD)
            d_outer[["patient_id", "file_name", "y"]].assign(logit=z, outer=k, inner=j).to_csv(f_out, index=False)
            print(f"saved {f_out.name}")


if __name__ == "__main__":
    main()
