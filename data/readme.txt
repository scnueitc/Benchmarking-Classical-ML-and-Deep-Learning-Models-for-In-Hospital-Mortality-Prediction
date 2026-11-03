- **Data**: 12,000 ICU admissions × 81 columns from the PhysioNet/Computing in
  Cardiology Challenge 2012 (pre-imputed). Rows are ordered and split into
  **set A (rows 1–4,000, development)**, **set B (rows 4,001–8,000, external 1)**,
  **set C (rows 8,001–12,000, external 2)**.
- **Label / score columns**: `In.hospital_death` (0/1), `SAPS.I` (SAPS-I score),
  `SOFA` (SOFA score). The remaining **78 columns** are summary features
  (demographics + min/max/median/mean of physiological variables over the first 48 h).
