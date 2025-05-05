import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
import config

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum += val*n; self.count += n; self.avg = self.sum/self.count

def accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.best = None
        self.epochs_no_improve = 0
        self.patience = patience
        self.verbose = verbose
        self.best_state = None

    def step(self, metric, model):
        if self.best is None or metric > self.best:
            self.best = metric
            self.epochs_no_improve = 0
            self.best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered (no improvement for {self.patience} epochs).")
                return True
            return False

def _pad_2d(t, H, W):
    _, h, w = t.shape
    return F.pad(t, (0, W - w, 0, H - h))  # right, bottom

def _pad_4d(v, H, W):
    # v: (C, T, h, w)
    C, T, h, w = v.shape
    return F.pad(v, (0, W - w, 0, H - h))  # last two dims

def custom_collate(batch):
    out = {"label": torch.tensor([b["label"] for b in batch])}

    # -------- imgae (audio, ecg, eda, rr) --------
    for key in ["audio", "ecg", "eda", "rr"]:
        imgs = [b[key] for b in batch]
        maxH = max(img.shape[1] for img in imgs)
        maxW = max(img.shape[2] for img in imgs)
        out[key] = torch.stack([_pad_2d(img, maxH, maxW) for img in imgs])

    # -------- video --------
    vids = [b["video"] for b in batch]
    maxH = max(v.shape[2] for v in vids)
    maxW = max(v.shape[3] for v in vids)
    out["video"] = torch.stack([_pad_4d(v, maxH, maxW) for v in vids])

    return out

def get_class_weights(split="train"):
    df = pd.read_csv(config.CSV_PATH)
    counts = df[df["split"] == split]["affect3-class"].value_counts().sort_index()
    weights = 1.0 / counts
    normed = weights / weights.sum()
    return torch.tensor(normed.values, dtype=torch.float32)

def unweighted_accuracy(preds, labels):
    """UAR: 평균 클래스별 Recall"""
    return recall_score(labels, preds, average="macro", zero_division=0)