# -*- coding: utf-8 -*-
"""One-command benchmark experiment (the training + prediction pipeline).

Pipeline
  1. Load data, split into set A (development) / B, C (external).
  2. Five-fold stratified CV on set A for all 7 models -> out-of-fold predictions.
  3. Refit every model on the full set A and predict sets B and C WITHOUT refitting.
  4. Save per-fold and summary metrics, baseline severity scores (SAPS-II/SOFA),
     DeLong + Holm tables for all three cohorts, feature importances and the
     top-15 intersection, plus all raw predictions (figdata/predictions.npz).

Usage
  python run_experiment.py             # full run (~50 min on 4-core CPU)
  python run_experiment.py --smoke     # quick validation (2 folds, 3 epochs,
                                       # writes to results_smoke/ figdata_smoke/)
"""
import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import SEED, LABEL, SCORE_COLS, load_data, split_sets
from models import DEEP, ALL_MODELS, make_classic
from train import train_torch, fit_full
from predict import predict_proba_batch
from evaluate import (ece_score, hosmer_lemeshow, calibration_slope_intercept,
                      delong_family)

warnings.filterwarnings("ignore")


def main(smoke=False, folds=5, epochs=30, patience=6):
    here = os.path.dirname(os.path.abspath(__file__))
    if smoke:
        out_dir = os.path.join(here, "results_smoke")
        fig_dir = os.path.join(here, "figdata_smoke")
        folds, epochs, patience = 2, 3, 2
        print("SMOKE MODE: 2 folds, 3 epochs -> results_smoke/ figdata_smoke/")
    else:
        out_dir = os.path.join(here, "results")
        fig_dir = os.path.join(here, "figdata")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    np.random.seed(SEED)

    # ------------------------------------------------ 1. data
    df, FEATURES = load_data()
    A, B, C = split_sets(df)
    print("prevalence  A: %.4f  B: %.4f  C: %.4f" %
          (A[LABEL].mean(), B[LABEL].mean(), C[LABEL].mean()))

    Xa, ya = A[FEATURES].values.astype(np.float32), A[LABEL].values.astype(int)

    # ------------------------------------------------ 2. cross-validation on set A
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(A)) for m in ALL_MODELS}
    fold_rows = []
    for fold, (itr, ite) in enumerate(skf.split(Xa, ya), 1):
        sc = StandardScaler().fit(Xa[itr])
        Xtr, Xte = sc.transform(Xa[itr]), sc.transform(Xa[ite])
        # internal early-stop split from the training part
        itr2, iva = train_test_split(itr, test_size=0.15, stratify=ya[itr],
                                     random_state=SEED)
        sc2 = StandardScaler().fit(Xa[itr2])
        Xtr2, Xva = sc2.transform(Xa[itr2]), sc2.transform(Xa[iva])
        for name in ALL_MODELS:
            if name in DEEP:                                # ---- deep models
                mdl = train_torch(DEEP[name], Xtr2, ya[itr2], Xva, ya[iva],
                                  max_epochs=epochs, patience=patience)
                p = predict_proba_batch(mdl, Xte)
            else:                                           # ---- classical models
                mdl = make_classic(name).fit(Xtr, ya[itr])
                p = mdl.predict_proba(Xte)[:, 1]
            oof[name][ite] = p
            yte = ya[ite]
            cs, ci = calibration_slope_intercept(yte, p)
            hl, hldf, hlp = hosmer_lemeshow(yte, p)
            fold_rows.append(dict(model=name, fold=fold,
                                  AUROC=roc_auc_score(yte, p),
                                  AUPRC=average_precision_score(yte, p),
                                  Brier=brier_score_loss(yte, p),
                                  ECE=ece_score(yte, p),
                                  CalibSlope=cs, CalibIntercept=ci,
                                  HL_stat=hl, HL_df=hldf, HL_p=hlp))
        print(f"fold {fold} done", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(out_dir, "cv_per_fold.csv"), index=False)
    fold_df.groupby("model").agg(["mean", "std"]).round(4).to_csv(
        os.path.join(out_dir, "cv_summary.csv"))
    print("\n== CV (set A) ==")
    print(fold_df.groupby("model")[["AUROC", "AUPRC", "Brier", "ECE"]].mean().round(4))

    # ------------------------------------------------ 3. external validation
    ext_pred, ext_rows = {}, []
    for name in ALL_MODELS:
        mdl, sc = fit_full(name, Xa, ya, max_epochs=epochs, patience=patience)
        for sname, S in (("B", B), ("C", C)):
            Xs = S[FEATURES].values.astype(np.float32)
            ys = S[LABEL].values.astype(int)
            p = (predict_proba_batch(mdl, sc.transform(Xs)) if name in DEEP
                 else mdl.predict_proba(sc.transform(Xs))[:, 1])
            ext_pred[(name, sname)] = p
            cs, ci = calibration_slope_intercept(ys, p)
            hl, hldf, hlp = hosmer_lemeshow(ys, p)
            ext_rows.append(dict(model=name, set=sname,
                                 AUROC=roc_auc_score(ys, p),
                                 AUPRC=average_precision_score(ys, p),
                                 Brier=brier_score_loss(ys, p),
                                 ECE=ece_score(ys, p),
                                 CalibSlope=cs, CalibIntercept=ci,
                                 HL_stat=hl, HL_df=hldf, HL_p=hlp))
        print(f"external fit done: {name}", flush=True)

    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(os.path.join(out_dir, "external_validation.csv"), index=False)
    print("\n== external validation ==")
    print(ext_df.round(4).to_string(index=False))

    # save all raw predictions so later stages can never lose them
    np.savez_compressed(
        os.path.join(fig_dir, "predictions.npz"),
        y_a=ya, y_b=B[LABEL].values.astype(int), y_c=C[LABEL].values.astype(int),
        **{f"oof_{m}": oof[m] for m in ALL_MODELS},
        **{f"ext_{m}_{s}": ext_pred[(m, s)] for m in ALL_MODELS for s in ("B", "C")},
        saps_a=A[SCORE_COLS[0]].values, sofa_a=A[SCORE_COLS[1]].values,
        saps_b=B[SCORE_COLS[0]].values, sofa_b=B[SCORE_COLS[1]].values,
        saps_c=C[SCORE_COLS[0]].values, sofa_c=C[SCORE_COLS[1]].values)
    print("predictions saved.", flush=True)

    # ------------------------------------------------ 4. severity-score baselines
    base_rows = []
    for sname, S in (("A", A), ("B", B), ("C", C)):
        ys = S[LABEL].values.astype(int)
        for score_name in SCORE_COLS:
            p = S[score_name].values.astype(float)   # scores, not probabilities
            base_rows.append(dict(model=score_name, set=sname,
                                  AUROC=roc_auc_score(ys, p),
                                  AUPRC=average_precision_score(ys, p)))
    pd.DataFrame(base_rows).to_csv(os.path.join(out_dir, "baseline_scores.csv"),
                                   index=False)

    # ------------------------------------------------ 5. DeLong + Holm (3 cohorts)
    delong_family(ya, oof, "setA_oof", out_dir)
    delong_family(B[LABEL].values.astype(int),
                  {m: ext_pred[(m, "B")] for m in ALL_MODELS}, "setB", out_dir)
    delong_family(C[LABEL].values.astype(int),
                  {m: ext_pred[(m, "C")] for m in ALL_MODELS}, "setC", out_dir)
    print("\nDeLong + Holm tables saved.")

    # ------------------------------------------------ 6. interpretability
    mdl_lr, sc = fit_full("LR", Xa, ya, max_epochs=epochs, patience=patience)
    mdl_rf, _ = fit_full("RF", Xa, ya, max_epochs=epochs, patience=patience)
    mdl_xgb, _ = fit_full("XGBoost", Xa, ya, max_epochs=epochs, patience=patience)
    Xa_s = sc.transform(Xa)
    imp = pd.DataFrame({
        "feature": FEATURES,
        "LR_abscoef": np.abs(mdl_lr.coef_[0]),
        "RF_importance": mdl_rf.feature_importances_,
        "XGB_importance": mdl_xgb.feature_importances_})
    imp.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
    top15 = {m: list(imp.sort_values(m, ascending=False)["feature"].head(15))
             for m in ("LR_abscoef", "RF_importance", "XGB_importance")}
    inter = (set(top15["LR_abscoef"]) & set(top15["RF_importance"])
             & set(top15["XGB_importance"]))
    pd.DataFrame({"rank": range(1, 16),
                  "LR": top15["LR_abscoef"],
                  "RF": top15["RF_importance"],
                  "XGBoost": top15["XGB_importance"]}).to_csv(
        os.path.join(out_dir, "top15_predictors.csv"), index=False)
    with open(os.path.join(out_dir, "top15_intersection.json"), "w") as f:
        json.dump(sorted(inter), f, indent=1)
    print("\ntop-15 intersection:", sorted(inter))
    print("\nALL DONE  (outputs in %s)" % out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="7-model ICU mortality benchmark")
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline validation (2 folds / 3 epochs, *_smoke dirs)")
    args = ap.parse_args()
    main(smoke=args.smoke)
