"""
Data preparation: ThyroidXL -> histopathology-confirmed image table with patient folds.

Design rules (these are the "leakage-free" guarantees, checked by assertions):
  * every patient has exactly ONE label
  * folds are assigned to PATIENTS, never to images
  * dev / test patients never overlap
  * the inner early-stopping split is drawn from the training folds only
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

MALIGNANT_KWS = ["papillary", "ptc", "medullary", "mtc", "anaplastic", "carcin", "cancer", "malig"]
BENIGN_KWS = ["benign", "cyst", "colloid", "adenoma", "goiter", "thyroiditis", "hashimoto", "hyperplasia"]


def histopath_to_y(h):
    """Malignant keywords are checked first, so 'PTC, chronic thyroiditis' -> 1."""
    if h is None:
        return None
    s = str(h).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    if any(k in s for k in MALIGNANT_KWS):
        return 1
    if any(k in s for k in BENIGN_KWS):
        return 0
    return None  # e.g. "Follicular thyroid tumor" (indeterminate) -> excluded


def tirads_group(tr):
    tr = int(tr)
    return 0 if tr <= 3 else (1 if tr == 4 else 2)


def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_master_table(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (images_df, excluded_df). images_df has one row per hist-confirmed image."""
    data_root = Path(data_root)
    stats = data_root / "stats"
    id2info_path = next(p for p in [stats / "id2info_eng.json", stats / "new_id2info.json"] if p.exists())
    id2info = _load_json(id2info_path)

    rows, excluded = [], []
    for split in ["train", "test"]:
        ann = _load_json(data_root / split / f"{split}_annotations.json")
        img_dir = data_root / split / "images"
        for im in ann["images"]:
            fname = Path(im["file_name"]).name
            case_key = fname.split("_")[0]
            info = id2info.get(case_key, {})
            n1 = info.get("nodule_1") or {}
            h = n1.get("Histopathology")
            y = histopath_to_y(h)
            if y is None:
                continue
            rows.append({
                "official_split": split,
                "img_path": str(img_dir / fname),
                "file_name": fname,
                "case_key": case_key,
                "patient_id": int(im["patient_id"]),
                "histopathology": h,
                "y": int(y),
                "tirads": int(n1.get("TIRADS")) if n1.get("TIRADS") is not None else -1,
                "has_nodule2": info.get("nodule_2") is not None,
            })
    for case_key, info in id2info.items():   # cases WITH histopathology that could not be mapped
        h = (info.get("nodule_1") or {}).get("Histopathology")
        if h is not None and histopath_to_y(h) is None:
            excluded.append({"case_key": case_key, "histopathology": h})
    df = pd.DataFrame(rows)
    df["tirads_grp"] = df["tirads"].apply(lambda t: tirads_group(t) if t >= 0 else -1)
    return df, pd.DataFrame(excluded, columns=["case_key", "histopathology"])


def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """
    Patient-level stratified folds.
      fold_dev : 1..K for official-train patients (development cohort), 0 for test patients
      fold_all : 1..K over ALL patients (secondary pooled analysis, 92 benign instead of 15)
    Patients are sorted by id before splitting so the folds are reproducible
    (and identical to the folds used by the original OOF notebook).
    """
    pat = (df.groupby("patient_id")
             .agg(y=("y", "first"), official_split=("official_split", "first"),
                  n_imgs=("file_name", "size"), tirads=("tirads", "max"),
                  has_nodule2=("has_nodule2", "first"))
             .reset_index().sort_values("patient_id").reset_index(drop=True))

    pat["cohort"] = np.where(pat["official_split"] == "test", "test", "dev")
    pat["fold_dev"] = 0
    dev = pat[pat["cohort"] == "dev"].reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(dev["patient_id"], dev["y"]), start=1):
        pat.loc[dev.loc[va, "index"].values, "fold_dev"] = k

    pat["fold_all"] = 0
    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf2.split(pat["patient_id"], pat["y"]), start=1):
        pat.loc[va, "fold_all"] = k

    df = df.drop(columns=[c for c in ["cohort", "fold_dev", "fold_all"] if c in df.columns])
    df = df.merge(pat[["patient_id", "cohort", "fold_dev", "fold_all"]], on="patient_id", how="left")
    return df, pat


def sanity_checks(df: pd.DataFrame, pat: pd.DataFrame, check_files=True):
    """Hard assertions. If any fails, the pipeline must not continue."""
    n_lab = df.groupby("patient_id")["y"].nunique()
    assert (n_lab == 1).all(), f"Patients with mixed labels: {n_lab[n_lab > 1].index.tolist()[:10]}"

    dev_ids = set(pat.loc[pat.cohort == "dev", "patient_id"])
    test_ids = set(pat.loc[pat.cohort == "test", "patient_id"])
    assert dev_ids.isdisjoint(test_ids), "Dev/test patient overlap!"

    per_pat_fold = df[df.cohort == "dev"].groupby("patient_id")["fold_dev"].nunique()
    assert (per_pat_fold == 1).all(), "A patient's images ended up in different folds!"
    assert (pat.loc[pat.cohort == "dev", "fold_dev"] > 0).all()

    if check_files:
        missing = [p for p in df["img_path"] if not Path(p).exists()]
        assert not missing, f"{len(missing)} image files missing, e.g. {missing[:3]}"
    print("All sanity checks passed: single label per patient, folds by patient, dev/test disjoint.")


def dataset_summary(df: pd.DataFrame, pat: pd.DataFrame) -> pd.DataFrame:
    """Table 1 of the paper (cohort characteristics)."""
    rows = []
    for name, p in [("Development", pat[pat.cohort == "dev"]), ("Test", pat[pat.cohort == "test"]), ("Total", pat)]:
        d = df[df.patient_id.isin(p.patient_id)]
        rows.append({
            "Cohort": name,
            "Patients": len(p),
            "Malignant patients": int(p.y.sum()),
            "Benign patients": int((p.y == 0).sum()),
            "Malignant prevalence (%)": round(100 * p.y.mean(), 2),
            "Images": len(d),
            "Images / patient (mean)": round(p.n_imgs.mean(), 2),
            "Images / patient (min-max)": f"{p.n_imgs.min()}-{p.n_imgs.max()}",
            "Patients with 2nd nodule": int(p.has_nodule2.sum()),
        })
    return pd.DataFrame(rows)


def fold_split(pat: pd.DataFrame, fold: int, fold_col: str, inner_val_frac: float, seed: int):
    """
    For outer fold k return three disjoint patient-id sets:
      train_ids  : used for gradient updates
      inner_ids  : used ONLY for early stopping / checkpoint selection
      outer_ids  : the held-out fold -> its predictions become the OOF predictions
    """
    pool = pat[pat[fold_col] > 0]
    outer = pool[pool[fold_col] == fold]
    rest = pool[pool[fold_col] != fold]
    tr, inner = train_test_split(rest["patient_id"].values, test_size=inner_val_frac,
                                 random_state=seed + fold, stratify=rest["y"].values)
    return set(tr), set(inner), set(outer["patient_id"])
