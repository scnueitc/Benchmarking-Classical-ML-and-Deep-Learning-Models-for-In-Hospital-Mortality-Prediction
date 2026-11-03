# Benchmarking Classical ML and Deep Learning Models for In-Hospital Mortality Prediction

> Benchmarking Classical Machine Learning and Deep Learning Models for In-Hospital
> Mortality Prediction in ICU Patients: An Internal–External Validation Study on the
> PhysioNet/CinC Challenge 2012 Cohort

## 1. Study design

- **Data**: 12,000 ICU admissions × 81 columns from the PhysioNet/Computing in
  Cardiology Challenge 2012 (pre-imputed). Rows are ordered and split into
  **set A (rows 1–4,000, development)**, **set B (rows 4,001–8,000, external 1)**,
  **set C (rows 8,001–12,000, external 2)**.
- **Label / score columns**: `In.hospital_death` (0/1), `SAPS.I` (SAPS-I score),
  `SOFA` (SOFA score). The remaining **78 columns** are summary features
  (demographics + min/max/median/mean of physiological variables over the first 48 h).
- **Models**: LR, RF, XGBoost (classical) and MLP, 1D-CNN, LSTM, FT-Transformer (deep),
  all with a fixed seed (42).
- **Protocol**: five-fold cross-validation on set A; all models then applied to
  sets B and C **without refitting**.
- **Metrics**: AUROC, AUPRC, Brier, ECE, calibration slope/intercept,
  Hosmer–Lemeshow, decision curve analysis; paired DeLong tests with
  Holm–Bonferroni adjustment (21 comparisons per cohort); intersection of the
  top-15 predictors of LR / RF / XGBoost.

## 2. Code layout (training / prediction / evaluation separated)

```
upload_code/
├── config.py             # paths, seed, label, 78-dim feature list, A/B/C split
├── models.py             # MODEL DEFINITIONS: LR/RF/XGBoost factory + MLP, 1D-CNN,
│                         #   LSTM, FT-Transformer (PyTorch)
├── train.py              # TRAINING: train_torch() early-stopping trainer,
│                         #   fit_full() refits any model on the full set A
├── predict.py            # PREDICTION: predict_proba_batch() / score_dataframe()
│                         #   + CLI that exports every prediction to results/*.csv
├── evaluate.py           # EVALUATION: ECE, Hosmer–Lemeshow, calibration slope,
│                         #   paired DeLong test (Sun & Xu 2014), Holm adjustment
├── run_experiment.py     # MAIN PIPELINE: CV on set A -> external validation ->
│                         #   baselines -> DeLong -> interpretability
│                         #   (--smoke = 2-fold/3-epoch quick validation)
├── fig.py                # generates the 6 publication figures from the outputs
├── data/
│   └── completedone1.csv # source dataset (12,000 x 81)

```

## 3. Environment

Python 3.13 (any ≥3.10 should work). Install dependencies:

```
pip install -r requirements.txt

```

## 4. Key results (summary)

| Finding | Value |
|---|---|
| XGBoost best discrimination | CV AUROC 0.838; external B / C 0.852 / 0.857 |
| FT-Transformer best calibration | external ECE 0.013–0.014; calibration slope ≈ 1.00 |
| Sequence models lag | LSTM 0.59–0.63, 1D-CNN 0.68–0.69 |
| SAPS-II / SOFA baselines | AUROC 0.63–0.66 |
| DeLong (set A) | 15/21 pairs significant after Holm adjustment |
| Top-15 intersection (LR∩RF∩XGB) | medianGCS, baseAge, meanHCO3, medianLactate, minBUN |

## 5. Notes

- Row order defines the set split; if your source CSV is concatenated in a
  different order, adjust `split_sets()` in `config.py`.
- Severity scores (SAPS-I, SOFA) are evaluated as baselines only (AUROC/AUPRC);
  calibration metrics are not applicable to raw scores.
- The DeLong implementation follows Sun & Xu (2014) fast covariance; Holm–Bonferroni
  adjustment is applied per cohort across all 21 pairs.
