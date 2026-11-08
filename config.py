"""
Central configuration for the LF-PEF project.
==============================================
Edit ONLY this file to change paths or experiment settings.
Every script reads from here, so all experiments share one source of truth.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) Paths  (edit DATA_ROOT to your machine / Colab drive)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("LFPEF_DATA_ROOT", "data/ThyroidXL"))  # folder containing train/ test/ stats/
PROJECT_ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("LFPEF_OUT", PROJECT_ROOT / "outputs"))  # everything the pipeline writes goes here

MASTER_CSV = OUT / "data" / "master_images.csv"     # one row per image (+ split, fold, labels)
PATIENTS_CSV = OUT / "data" / "master_patients.csv" # one row per patient
PRED_DIR = OUT / "predictions"            # image-level logits of every model
MODEL_DIR = OUT / "models"                # .keras checkpoints
TABLE_DIR = OUT / "tables"                # CSV + LaTeX tables for the paper
FIG_DIR = OUT / "figures"                 # PNG + PDF figures for the paper
LOG_DIR = OUT / "logs"

# ---------------------------------------------------------------------------
# 2) Data / split
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
INNER_VAL_FRAC = 0.15     # patients carved out of each training fold for early stopping
                          # (keeps the outer fold untouched -> OOF stays unbiased)
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# 3) Frame-level CNN (the model behind LF-PEF)
# ---------------------------------------------------------------------------
BACKBONE = "EfficientNetV2B0"
BACKBONE_WEIGHTS = "imagenet"   # set to None only for offline smoke tests
USE_TIRADS_AUX = True           # auxiliary TI-RADS head (loss weight below)
TIRADS_LOSS_WEIGHT = 0.5
DROPOUT = 0.30
BATCH_SIZE = 32

STAGE1_EPOCHS = 50              # frozen backbone
STAGE1_LR = 3e-4
STAGE2_EPOCHS = 15              # fine-tune last N layers (set 0 to skip stage 2)
STAGE2_LR = 1e-5
UNFREEZE_LAST = 60
PATIENCE = 6

# ---------------------------------------------------------------------------
# 4) AttnMIL baseline (same folds, same inner validation, same calibration)
# ---------------------------------------------------------------------------
MIL_BATCH = 4
MIL_ATTN_DIM = 128
MIL_STAGE1_EPOCHS = 30
MIL_STAGE2_EPOCHS = 10          # set 0 to keep the backbone frozen

# ---------------------------------------------------------------------------
# 5) Leakage experiment
# ---------------------------------------------------------------------------
LEAK_REPEATS = 5                # repeat each protocol with 5 different seeds
LEAK_EPOCHS = 15

# ---------------------------------------------------------------------------
# 6) Evaluation
# ---------------------------------------------------------------------------
N_BOOT = 2000
MIN_SENS = 0.90                 # high-sensitivity operating point
ECE_BINS = 10


def ensure_dirs():
    for d in [OUT, MASTER_CSV.parent, PRED_DIR, MODEL_DIR, TABLE_DIR, FIG_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
