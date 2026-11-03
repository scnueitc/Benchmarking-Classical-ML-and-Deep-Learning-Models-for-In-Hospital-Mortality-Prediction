# -*- coding: utf-8 -*-
"""Prediction utilities.

- predict_proba_batch : score any (n, 78) feature matrix with a trained model.
- score_dataframe     : convenience wrapper taking a raw DataFrame.
- export_predictions  : CLI - dumps every stored prediction (5-fold OOF on set A,
                        external sets B/C) from figdata/predictions.npz to
                        results/predictions_*.csv for inspection.

Usage:
    python predict.py            # export all prediction tables
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch

from config import LABEL, SCORE_COLS, OUT, FIGDATA
from models import DEEP, ALL_MODELS


def predict_proba_batch(model, X):
    """Predicted mortality probabilities for an (n, d) float matrix."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
    if was_training:
        model.train()
    return p


def score_dataframe(model, scaler, df, features):
    """Score a raw patient DataFrame; returns a 1-D probability array."""
    X = scaler.transform(df[features].values.astype(np.float32))
    if type(model).__name__ in {c.__name__ for c in DEEP.values()}:
        return predict_proba_batch(model, X)
    return model.predict_proba(X)[:, 1]


def export_predictions(figdata=FIGDATA, out_dir=OUT):
    """Write all stored predictions as tidy CSV tables."""
    z = np.load(os.path.join(figdata, "predictions.npz"))

    oof = pd.DataFrame({"y_true": z["y_a"].astype(int)})
    for m in ALL_MODELS:
        oof[m] = z[f"oof_{m}"]
    oof[SCORE_COLS[0]] = z["saps_a"]
    oof[SCORE_COLS[1]] = z["sofa_a"]
    oof.insert(0, "row_id", np.arange(1, len(oof) + 1))
    p1 = os.path.join(out_dir, "predictions_oof_setA.csv")
    oof.to_csv(p1, index=False)

    rows = []
    for s in ("B", "C"):
        d = pd.DataFrame({"set": s, "y_true": z[f"y_{s.lower()}"].astype(int)})
        for m in ALL_MODELS:
            d[m] = z[f"ext_{m}_{s}"]
        d[SCORE_COLS[0]] = z[f"saps_{s.lower()}"]
        d[SCORE_COLS[1]] = z[f"sofa_{s.lower()}"]
        rows.append(d)
    ext = pd.concat(rows, ignore_index=True)
    ext.insert(1, "row_id", ext.groupby("set").cumcount() + 1)
    p2 = os.path.join(out_dir, "predictions_external_BC.csv")
    ext.to_csv(p2, index=False)

    print("written:", p1, oof.shape)
    print("written:", p2, ext.shape)
    return p1, p2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export stored model predictions.")
    ap.add_argument("--figdata", default=FIGDATA)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    export_predictions(args.figdata, args.out)
