# -*- coding: utf-8 -*-
"""Training utilities.

- train_torch : Adam + BCEWithLogits, mini-batch training with AUROC-based
                early stopping on a validation split; returns the best model.
- fit_full    : fits any of the 7 models on the full development set
                (deep models use an internal 85/15 early-stopping split),
                returning (model, scaler) ready for prediction.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from config import SEED
from models import DEEP, make_classic


def train_torch(model_cls, Xtr, ytr, Xva, yva, max_epochs=30, patience=6,
                lr=1e-3, bs=512):
    """Train a tabular deep model with early stopping on validation AUROC."""
    torch.manual_seed(SEED)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    best_auc, best_state, bad = -1, None, 0
    n = len(Xtr_t)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best_auc + 1e-5:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def fit_full(name, X, y, max_epochs=30, patience=6):
    """Fit model `name` on the full development set.

    Returns (model, scaler). For classical models the scaler is fitted on all
    rows; deep models use an internal stratified 85/15 split for early stopping
    and the scaler is fitted on the training part only.
    """
    itr2, iva = train_test_split(np.arange(len(y)), test_size=0.15,
                                 stratify=y, random_state=SEED)
    sc = StandardScaler().fit(X[itr2])
    if name in DEEP:
        mdl = train_torch(DEEP[name], sc.transform(X[itr2]), y[itr2],
                          sc.transform(X[iva]), y[iva],
                          max_epochs=max_epochs, patience=patience)
    else:
        mdl = make_classic(name).fit(sc.transform(X), y)
    return mdl, sc
