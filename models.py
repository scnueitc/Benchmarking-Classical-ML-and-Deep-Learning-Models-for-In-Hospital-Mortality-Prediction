# -*- coding: utf-8 -*-
"""Model definitions for the 7-model benchmark.

Classical: LR (penalised-free logistic regression), RF, XGBoost.
Deep (PyTorch, tabular): MLP, 1D-CNN, LSTM, FT-Transformer.
"""
import os
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import SEED

torch.manual_seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.use_deterministic_algorithms(False)


# ------------------------------------------------------------------ deep nets
class MLPNet(nn.Module):
    """3-layer perceptron with dropout."""
    def __init__(self, d=78):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1DNet(nn.Module):
    """Treats the 78 features as a 1-D signal (conv + global average pooling)."""
    def __init__(self, d=78):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.body(x.unsqueeze(1)).squeeze(-1)
        return self.head(h).squeeze(-1)


class LSTMNet(nn.Module):
    """Treats the 78 features as a length-78 sequence of scalars."""
    def __init__(self, d=78):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        h, _ = self.lstm(x.unsqueeze(-1))
        return self.head(h[:, -1]).squeeze(-1)


class FTTransformer(nn.Module):
    """Feature-tokeniser + Transformer encoder (Gorishniy et al., 2021)."""
    def __init__(self, d=78, d_token=32, n_layers=2, n_heads=4):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d, d_token))
        nn.init.xavier_uniform_(self.W)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads,
                                           dim_feedforward=d_token * 2,
                                           dropout=0.2, batch_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        tok = torch.einsum("bf,fd->bfd", x, self.W)  # (B, d, d_token)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = self.enc(torch.cat([cls, tok], dim=1))
        return self.head(h[:, 0]).squeeze(-1)


# ------------------------------------------------------------------ classical
def make_classic(name):
    """Factory for the scikit-learn / XGBoost models (all hyper-parameters fixed)."""
    if name == "LR":
        return LogisticRegression(max_iter=5000, random_state=SEED)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                      random_state=SEED, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=SEED,
                             n_jobs=-1, tree_method="hist", verbosity=0)
    raise ValueError(name)


DEEP = {"MLP": MLPNet, "1D-CNN": CNN1DNet, "LSTM": LSTMNet,
        "FT-Transformer": FTTransformer}
ALL_MODELS = ["LR", "RF", "XGBoost", "MLP", "1D-CNN", "LSTM", "FT-Transformer"]
