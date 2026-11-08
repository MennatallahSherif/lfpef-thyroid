# LF-PEF: Leakage-free patient-level evaluation of thyroid ultrasound malignancy prediction

Code for the manuscript

> **Leakage-Free Patient-Level Evaluation of Thyroid Ultrasound Malignancy Prediction: Split Variability, Fold Ensembling, and Prior-Shift Recalibration**
> Mennatallah Sherif, Eman K. Elsayed, Mohanad A. Deif

The pipeline reproduces every number, table and figure in the paper and its supplementary information from the raw ThyroidXL files. No value is entered by hand.

## What the pipeline does

- Patient-level split of the histopathology-confirmed ThyroidXL subset (796 patients, 2,225 images) into a development and an independent test cohort.
- Patient-level stratified 5-fold cross-validation with an **inner validation subset** for early stopping and checkpoint selection, so that out-of-fold (OOF) predictions are never used for model selection.
- Frame-level EfficientNetV2-B0 classifier (optional TI-RADS auxiliary head), mean pooling of frame logits per patient, and five-model ensembling on the test cohort.
- Attention-based multiple-instance learning (AttnMIL) baseline under the same partitioning, model selection and evaluation.
- Temperature scaling and logistic (Platt) recalibration fitted on OOF predictions only; development-only cross-fitted, nested-ensemble and pooled cross-fitted calibration analyses.
- Matched leakage experiment (leaky vs. leakage-free training evaluated on identical frames).
- Stratified bootstrap and Wilson intervals, paired DeLong tests (Holm-adjusted where applicable), paired bootstrap differences, calibration intercept/slope and decision curves.

## Requirements

Python 3.10–3.12 and

```
pip install -r requirements.txt
```

All experiments in the paper were run on a CPU (no GPU required). Approximate run times on a laptop CPU are given below.

## Data

ThyroidXL is distributed by its original authors and is **not** included in this repository:
<https://huggingface.co/datasets/hunglc007/ThyroidXL>

Place the dataset so that the folder contains `train/`, `test/` and `stats/`, then either

- put it at `data/ThyroidXL/` (default), or
- set the environment variable `LFPEF_DATA_ROOT` to its location, or
- edit `DATA_ROOT` in `config.py`.

All outputs are written to `outputs/` (override with `LFPEF_OUT`).

## Run order

| Step | Command | What it produces | Time (CPU) |
|---|---|---|---|
| 00 | `python scripts/00_prepare_data.py` | cohort table, patient-level folds, sanity checks, Fig. 1 | ~1 min |
| 01 | `python scripts/01_train_frame_cv.py` | frame CNN, 5 folds: OOF + test logits | ~1 h |
| 01b | `python scripts/01_train_frame_cv.py --no-aux` | CNN without TI-RADS head (ablation) | ~1 h |
| 01c | `python scripts/01_train_frame_cv.py --cohort all` | pooled 5-fold CV over all patients | ~1.5 h |
| 02 | `python scripts/02_train_attnmil_cv.py` | AttnMIL baseline, 5 folds | 3–4 h |
| 03 | `python scripts/03_leakage_experiment.py` | unmatched leakage experiment (supplementary) | ~30 min |
| 03b | `python scripts/03b_leakage_matched.py` | matched leakage experiment (Table 2, Fig. 3) | ~1.5 h |
| 07 | `python scripts/07_nested_calibration.py` | 20 nested models for ensemble calibration | ~2–3 h |
| 04 | `python scripts/04_evaluate.py` and `python scripts/04_evaluate.py --cohort all` | all tables and statistics | seconds |
| 05 | `python scripts/05_figures.py` | Figs. 3–6 and supplementary figures | seconds |
| 06 | `python scripts/06_supplementary.py` | `outputs/supplementary/supplementary.tex` (Tables S1–S12, Figs. S1–S5) | seconds |

Every training script is resumable: completed folds/runs are detected and skipped.

### Quick smoke test (no dataset needed, a few minutes)

```
python tests/make_synthetic_thyroidxl.py synth
LFPEF_DATA_ROOT=synth LFPEF_OUT=synth_out python scripts/00_prepare_data.py
LFPEF_DATA_ROOT=synth LFPEF_OUT=synth_out python scripts/01_train_frame_cv.py --quick --no-weights
LFPEF_DATA_ROOT=synth LFPEF_OUT=synth_out python scripts/04_evaluate.py --quick
```
(On Windows PowerShell set the variables with `$env:LFPEF_DATA_ROOT="synth"` first.)

## Repository layout

```
config.py              all paths and hyper-parameters
lfpef/                 library code (data, models, training, metrics)
scripts/               numbered pipeline steps (see table above)
tests/                 synthetic dataset for smoke tests
figures_src/           TikZ source of Fig. 2 (pipeline overview)
```

## Mapping to the paper

| Paper | Produced by | File in `outputs/` |
|---|---|---|
| Table 1 | 00 | `tables/table1_cohort.csv` |
| Table 2, Fig. 3 | 03b, 05 | `tables/table_leakage_matched_summary.csv`, `figures/fig3_leakage.*` |
| Table 3, Fig. 4 | 04, 05 | `tables/table_main_test_with_ci.csv`, `figures/fig4_roc_pr.*` |
| Table 4 | 04 | `tables/table_ablation_test.csv` |
| Table 5, Fig. 5 | 04, 07, 05 | `tables/table_calibration_*.csv`, `tables/table_pooled_cv_all796.csv`, `figures/fig5_calibration.*` |
| Table 6, Fig. 6 | 04, 05 | `tables/table_thresholds.csv`, `figures/fig6_decision_curve.*` |
| DeLong tests | 04 | `tables/table_delong.csv` |
| Supplementary | 06 | `supplementary/supplementary.tex` |

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## License

MIT License (see `LICENSE`). The ThyroidXL dataset is subject to its own license.
