# -*- coding: utf-8 -*-
"""Global configuration: paths, seed, label, and the set A/B/C split."""
import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "ICU_data.csv")

# Output locations (overridable, e.g. run_experiment.py --smoke writes to *_smoke)
OUT = os.path.join(HERE, "results")
FIGDATA = os.path.join(HERE, "figdata")

LABEL = "In.hospital_death"          # binary outcome
SCORE_COLS = ("SAPS.I", "SOFA")      # severity scores used as baselines only
N_PER_SET = 4000                     # set A / B / C sizes

os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGDATA, exist_ok=True)


def load_data(data_path=DATA):
    """Load the source CSV, coerce to numeric, impute with column medians.

    Returns (df, FEATURES) where FEATURES is the 78-dim feature list.
    """
    df = pd.read_csv(data_path)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    features = [c for c in df.columns if c not in (LABEL,) + SCORE_COLS]
    assert len(features) == 78, f"expected 78 features, got {len(features)}"
    return df, features


def split_sets(df):
    """Row order defines the cohorts: A (dev) / B (ext 1) / C (ext 2)."""
    n = N_PER_SET
    A = df.iloc[0:n].reset_index(drop=True)
    B = df.iloc[n:2 * n].reset_index(drop=True)
    C = df.iloc[2 * n:3 * n].reset_index(drop=True)
    return A, B, C
