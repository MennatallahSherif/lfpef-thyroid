"""Small helpers shared by the scripts."""
import argparse
import types

import pandas as pd


def load_cfg(config_module, quick=False):
    """Copy config.py into a mutable namespace. --quick shrinks everything for a smoke test."""
    cfg = types.SimpleNamespace(**{k: getattr(config_module, k) for k in dir(config_module) if k.isupper()})
    cfg.COLLAPSE_MIN_STD = 1e-3
    if quick:
        cfg.COLLAPSE_MIN_STD = 0.0          # random-init smoke tests are allowed to be flat
        cfg.STAGE1_EPOCHS = cfg.STAGE2_EPOCHS = 1
        cfg.MIL_STAGE1_EPOCHS = cfg.MIL_STAGE2_EPOCHS = 1
        cfg.LEAK_EPOCHS, cfg.LEAK_REPEATS = 1, 2
        cfg.N_BOOT = 50
        cfg.IMG_SIZE = 64
        cfg.BATCH_SIZE = 8
    return cfg


def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--cohort", default="dev", choices=["dev", "all"],
                    help="dev = official split (5-fold CV on dev + independent test); "
                         "all = pooled 5-fold CV over all 796 patients (secondary analysis)")
    ap.add_argument("--folds", default="1,2,3,4,5", help="comma list, e.g. 1,2 (resume support)")
    ap.add_argument("--tag", default=None, help="name of this model variant in the outputs")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (1 epoch, 64px)")
    ap.add_argument("--no-weights", action="store_true", help="random init (offline smoke tests only)")
    return ap


def load_master(cfg):
    df = pd.read_csv(cfg.MASTER_CSV)
    pat = pd.read_csv(cfg.PATIENTS_CSV)
    return df, pat
