"""Builds a tiny fake ThyroidXL folder (same JSON structure) to smoke-test the pipeline offline."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(sys.argv[1]); rng = np.random.default_rng(0)
(root / "stats").mkdir(parents=True, exist_ok=True)
id2info, pid = {}, 100
for split, n_pat in [("train", 90), ("test", 30)]:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    images, anns, iid = [], [], 1
    for _ in range(n_pat):
        pid += 1; key = f"{pid:08d}"
        y = int(rng.random() > 0.25)
        hist = "Papillary thyroid carcinoma" if y else rng.choice(["Thyroid cyst disease", "Colloid nodule"])
        id2info[key] = {"age": 40, "gender": 0, "conclusion": 3,
                        "nodule_1": {"TIRADS": int(rng.integers(3, 6)) if y else int(rng.integers(1, 5)), "Histopathology": hist},
                        "nodule_2": None, "images": []}
        for j in range(int(rng.integers(1, 5))):
            a = rng.normal(90, 25, (96, 96)); r = 25 if y else 15
            yy, xx = np.ogrid[:96, :96]; a[(yy - 48) ** 2 + (xx - 48) ** 2 < r * r] += 60 if y else 10
            fn = f"{key}_{j:04X}ABCD_{j}.png"
            Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(root / split / "images" / fn)
            images.append({"id": iid, "file_name": fn, "patient_id": pid}); anns.append({"id": iid, "image_id": iid, "category_id": y}); iid += 1
    # one indeterminate case that must be excluded
    json.dump({"info": {}, "licenses": [], "categories": [], "images": images, "annotations": anns},
              open(root / split / f"{split}_annotations.json", "w"))
id2info["00000999"] = {"nodule_1": {"TIRADS": 4, "Histopathology": "Follicular thyroid tumor"}, "nodule_2": None}
json.dump(id2info, open(root / "stats" / "id2info_eng.json", "w"))
print("synthetic dataset at", root)
