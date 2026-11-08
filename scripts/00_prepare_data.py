"""
STEP 00 - Prepare data (run once).
  * builds the histopathology-confirmed image table from the raw ThyroidXL JSONs
  * assigns patient-level folds (dev folds + pooled folds over all patients)
  * runs hard sanity checks (no leakage possible past this point)
  * writes Table 1 (cohort characteristics) and Fig. 1 (sample images, labelled by HISTOPATHOLOGY)

Usage:  python scripts/00_prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as C
from lfpef import data as Dt


def main(check_files=True):
    C.ensure_dirs()
    df, excluded = Dt.build_master_table(C.DATA_ROOT)
    df, pat = Dt.assign_folds(df, C.N_FOLDS, C.SEED)
    Dt.sanity_checks(df, pat, check_files=check_files)

    df.to_csv(C.MASTER_CSV, index=False)
    pat.to_csv(C.PATIENTS_CSV, index=False)
    excluded.to_csv(C.TABLE_DIR / "excluded_histopathology.csv", index=False)

    t1 = Dt.dataset_summary(df, pat)
    t1.to_csv(C.TABLE_DIR / "table1_cohort.csv", index=False)
    print("\nTable 1 - cohort characteristics\n", t1.to_string(index=False))

    folds = (pat[pat.cohort == "dev"].groupby("fold_dev")["y"]
             .agg(patients="size", malignant="sum").assign(benign=lambda d: d.patients - d.malignant))
    folds.to_csv(C.TABLE_DIR / "fold_composition_dev.csv")
    print("\nDevelopment folds\n", folds.to_string())
    print(f"\nExcluded (indeterminate histopathology): {len(excluded)} cases -> {excluded.histopathology.tolist()}")

    # Fig. 1: 2 benign + 2 malignant images chosen by the histopathology label (not COCO category_id)
    rng = np.random.default_rng(C.SEED)
    picks = []
    for lab in [0, 0, 1, 1]:
        cand = df[(df.y == lab) & (df.cohort == "dev") & ~df.file_name.isin([p.file_name for p in picks])]
        picks.append(cand.iloc[rng.integers(len(cand))])
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, r in zip(ax, picks):
        if Path(r.img_path).exists():
            a.imshow(Image.open(r.img_path).convert("L"), cmap="gray")
        a.set_title(f"{'Malignant' if r.y else 'Benign'}\n{r.histopathology}", fontsize=8)
        a.axis("off")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(C.FIG_DIR / f"fig1_samples.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {C.MASTER_CSV}\n       {C.PATIENTS_CSV}\n       tables + fig1 in {C.OUT}")


if __name__ == "__main__":
    main(check_files="--no-file-check" not in sys.argv)
