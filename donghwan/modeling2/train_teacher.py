# ────────────────────────── teacher_train.py (수정본) ──────────────────────────
import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

import config, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy, FocalLoss
# ───────────────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------------------- #
# 1.  모달 선택 →   python teacher_train.py --modal video
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--modal", required=True,
                    choices=["video", "ecg", "eda", "rr"],
                    help="Teacher 입력 모달")
args = parser.parse_args()

config.set_modal(args.modal)                 # ⬅️ 반드시 한 번 호출
SAVE_DIR   = config.SAVE_SUBDIR              # checkpoints/<modal>/
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2.  모달에 맞춰 배치에서 한 가지만 꺼내는 helper
# --------------------------------------------------------------------------- #
def pick_modal(batch):
    m = config.INPUT_MODAL
    x = batch[m].to(config.DEVICE, non_blocking=True)   # "video" → batch["video"] ...
    return x

# --------------------------------------------------------------------------- #
# 3.  epoch 루프
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    loss_m, acc_m = utils.AverageMeter(), utils.AverageMeter()
    preds, trues  = [], []

    for batch in loader:
        x = pick_modal(batch)                           # ⬅️ 하나만 전달
        y = batch["label"].to(config.DEVICE, non_blocking=True)

        logits, _ = model(x)                            # TeacherNet 인풋 1개로 수정
        loss = criterion(logits, y)

        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        acc = utils.accuracy(logits, y)
        loss_m.update(loss.item(), y.size(0))
        acc_m .update(acc,        y.size(0))
        preds.append(torch.argmax(logits, 1).cpu())
        trues.append(y.cpu())

    preds, trues = torch.cat(preds), torch.cat(trues)
    uar = unweighted_accuracy(preds, trues)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return loss_m.avg, acc_m.avg, uar, f1, preds, trues

# --------------------------------------------------------------------------- #
# 4.  학습 스크립트 본문
# --------------------------------------------------------------------------- #
def main():
    train_ds = datasets.StressMultimodalDataset("train", nf=16)
    val_ds   = datasets.StressMultimodalDataset("val",   nf=16)

    train_ld = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_ld   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # TeacherNet: 출력 feature dim은 config.TEACHER_FEAT_DIM
    model = models.TeacherNet(
            modal=config.INPUT_MODAL,              # ← 필수 인자
            out_feat_dim=config.TEACHER_FEAT_DIM   # 선택 인자
        ).to(config.DEVICE)
    cls_w     = get_class_weights("train").to(config.DEVICE)
    criterion = FocalLoss(weight=cls_w, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=0.1, patience=5, min_lr=1e-7)
    stopper   = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    tr_l, vl_l, tr_a, vl_a = [], [], [], []
    best_p, best_t = None, None

    for ep in range(config.NUM_EPOCHS):
        tl, ta, tu, tf, _, _ = run_epoch(model, train_ld, criterion, optimizer)
        vl, va, vu, vf, p, t = run_epoch(model, val_ld,   criterion)

        print(f"[{ep:03d}][{args.modal}] "
              f"train loss {tl:.4f} acc {ta:.4f} uar {tu:.4f} f1 {tf:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} uar {vu:.4f} f1 {vf:.4f}")

        tr_l.append(tl); vl_l.append(vl); tr_a.append(ta); vl_a.append(va)
        scheduler.step(vu)

        if stopper.step(vu, model): break
        best_p, best_t = p, t

    # best 저장
    model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), SAVE_DIR / "teacher_best.pth")

    # 그래프
    def plot(vals1, vals2, title, ylab, fn):
        plt.figure(); plt.plot(vals1, label="Train"); plt.plot(vals2, label="Val")
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylab); plt.legend()
        plt.savefig(SAVE_DIR / fn); plt.close()

    plot(tr_a, vl_a, "Accuracy", "Acc",  "acc.png")
    plot(tr_l, vl_l, "Loss",      "Loss", "loss.png")

    cm = confusion_matrix(best_t, best_p)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(SAVE_DIR / "confmat_teacher.png"); plt.close()
    print("✓ Best model & plots saved →", SAVE_DIR)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
