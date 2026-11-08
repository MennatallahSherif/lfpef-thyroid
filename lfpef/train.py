"""
Training utilities shared by the frame CNN and AttnMIL.

Early stopping / checkpoint selection use PATIENT-LEVEL ROC-AUC on an INNER validation
set drawn from the training folds. The outer fold is never seen during training or
model selection, so its predictions are genuine out-of-fold predictions.
"""
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import keras

from . import datasets as D
from .models import unfreeze_last


def set_seed(seed):
    keras.utils.set_random_seed(int(seed))


def _logits(model, ds):
    out = model.predict(ds, verbose=0)
    if isinstance(out, dict):
        out = out["y"]
    return np.asarray(out).reshape(-1).astype("float64")


def frame_patient_scores(model, df, cfg):
    """Image logits -> patient score (mean of logits). Returns (patient_ids, y, score)."""
    z = _logits(model, D.frame_predict_ds(df, cfg))
    g = pd.DataFrame({"pid": df["patient_id"].values, "y": df["y"].values, "z": z}).groupby("pid")
    agg = g.agg(y=("y", "first"), z=("z", "mean")).reset_index()
    return agg["pid"].values, agg["y"].values, agg["z"].values


class PatientAUCEarlyStopping(keras.callbacks.Callback):
    """Patient-level AUC on the inner validation set; keeps best weights in memory."""

    def __init__(self, score_fn, patience, best_score=-np.inf, best_weights=None, tag=""):
        super().__init__()
        self.score_fn, self.patience, self.tag = score_fn, patience, tag
        self.best_score, self.best_weights = best_score, best_weights
        self.wait, self.history = 0, []

    def on_epoch_end(self, epoch, logs=None):
        y, s = self.score_fn(self.model)
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        std = float(np.std(s))
        self.history.append({"stage": self.tag, "epoch": epoch + 1, "inner_patient_auc": auc,
                             "score_std": std, **{k: float(v) for k, v in (logs or {}).items()}})
        msg = f"   [{self.tag}] epoch {epoch + 1}: inner patient AUC = {auc:.4f} (score std {std:.3f})"
        if auc > self.best_score + 1e-4:
            self.best_score, self.wait = auc, 0
            self.best_weights = self.model.get_weights()
            msg += "  <- best"
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                msg += "  (early stop)"
        print(msg)

    def restore(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def check_not_collapsed(scores, name, min_std=1e-3):
    """Guard against the failure seen in the old baseline (all outputs ~ identical)."""
    if np.std(scores) < min_std:
        raise RuntimeError(f"{name}: predictions collapsed (std={np.std(scores):.2e}). "
                           "Check input scaling / learning rate before using this model.")


def two_stage_fit(model, backbone, compile_fn, train_ds, score_fn, cfg,
                  stage1_epochs, stage2_epochs, class_weight=None):
    """Stage 1: frozen backbone. Stage 2: unfreeze last N layers (BN frozen), low LR."""
    t0 = time.time()
    compile_fn(model, cfg.STAGE1_LR)
    cb1 = PatientAUCEarlyStopping(score_fn, cfg.PATIENCE, tag="stage1")
    model.fit(train_ds, epochs=stage1_epochs, callbacks=[cb1], class_weight=class_weight, verbose=2)
    cb1.restore()
    hist = cb1.history
    best, best_w = cb1.best_score, cb1.best_weights

    if stage2_epochs > 0:
        unfreeze_last(backbone, cfg.UNFREEZE_LAST)
        compile_fn(model, cfg.STAGE2_LR)
        cb2 = PatientAUCEarlyStopping(score_fn, cfg.PATIENCE, best, best_w, tag="stage2")
        model.fit(train_ds, epochs=stage2_epochs, callbacks=[cb2], class_weight=class_weight, verbose=2)
        cb2.restore()                      # stage-1 weights are kept if stage 2 never improved
        hist += cb2.history
        best = cb2.best_score
    print(f"   done in {(time.time() - t0) / 60:.1f} min | best inner patient AUC = {best:.4f}")
    return pd.DataFrame(hist), best


def compile_frame(use_aux, tirads_w):
    def _c(model, lr):
        loss = {"y": keras.losses.BinaryCrossentropy(from_logits=True)}
        lw = {"y": 1.0}
        if use_aux:
            loss["tr"] = keras.losses.SparseCategoricalCrossentropy()
            lw["tr"] = tirads_w
        model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, loss_weights=lw)
    return _c


def compile_mil(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True))
