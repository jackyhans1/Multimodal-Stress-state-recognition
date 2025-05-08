# ───────────────────────────── train_student.py ─────────────────────────────
import os, argparse, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config_student, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy, FocalLoss
# ──────────────────────────────────────────────────────────────────────────── #

# 1) 모달 선택  --------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--modal", required=True,
                    choices=["video", "ecg", "eda", "rr"],
                    help="Teacher 가 입력으로 받는 모달")
args = parser.parse_args()

config_student.set_modal(args.modal)                 # 필수! → save 경로 · feat_dim 자동세팅
SAVE_DIR = config_student.SAVE_SUBDIR                # checkpoints/<modal>/
os.makedirs(SAVE_DIR, exist_ok=True)

# 2) 배치에서 Teacher 모달만 뽑아주는 helper -----------------------------------
def pick_modal(batch):
    return batch[config_student.INPUT_MODAL].to(config_student.DEVICE, non_blocking=True)

# 3) epoch 루프 ---------------------------------------------------------------
def run_epoch(student, teacher, loader, crit_ce, optimizer=None):
    train = optimizer is not None
    student.train() if train else student.eval()
    teacher.eval()

    meter_l, meter_a = utils.AverageMeter(), utils.AverageMeter()
    preds, trues = [], []

    for batch in loader:
        # ── 데이터 ─────────────────────────────────────────────────────────
        x_t   = pick_modal(batch)                                    # Teacher 입력
        x_s   = batch["audio"].to(config_student.DEVICE, non_blocking=True)  # Student 입력
        y     = batch["label"].to(config_student.DEVICE, non_blocking=True)

        # ── Forward ───────────────────────────────────────────────────────
        with torch.no_grad():
            _, t_feat = teacher(x_t)          # (B, feat_dim)   (grad 없음)

        logits, s_feat = student(x_s)         # (B, feat_dim)

        # ── Loss ──────────────────────────────────────────────────────────
        loss_ce   = crit_ce(logits, y)
        loss_dist = nn.functional.mse_loss(s_feat, t_feat)
        loss      = loss_ce + config_student.ALPHA * loss_dist

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ── Metrics ───────────────────────────────────────────────────────
        acc = utils.accuracy(logits, y)
        meter_l.update(loss.item(), y.size(0))
        meter_a.update(acc,        y.size(0))
        preds.append(torch.argmax(logits, 1).cpu())
        trues.append(y.cpu())

    preds, trues = torch.cat(preds), torch.cat(trues)
    uar = unweighted_accuracy(preds, trues)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return meter_l.avg, meter_a.avg, uar, f1, preds, trues

# 4) 플롯 ----------------------------------------------------------------------
def plot_curve(tr, vl, title, ylabel, fname):
    plt.figure(); plt.plot(tr, label="Train"); plt.plot(vl, label="Val")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend()
    plt.savefig(SAVE_DIR / fname); plt.close()

# 5) main ---------------------------------------------------------------------
def main():
    # 데이터
    train_set = datasets.StressMultimodalDataset("train", nf=16)
    val_set   = datasets.StressMultimodalDataset("val",   nf=16)
    train_ld  = DataLoader(train_set, batch_size=config_student.BATCH_SIZE, shuffle=True,
                           num_workers=2, pin_memory=True, collate_fn=custom_collate)
    val_ld    = DataLoader(val_set,   batch_size=config_student.BATCH_SIZE, shuffle=False,
                           num_workers=2, pin_memory=True, collate_fn=custom_collate)

    # 모델
    teacher = models.TeacherNet(modal=args.modal).to(config_student.DEVICE)
    teacher_ckpt = SAVE_DIR / "teacher_best.pth"
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=config_student.DEVICE))

    student = models.StudentNet().to(config_student.DEVICE)
    student.classifier = nn.Sequential(
        nn.Dropout(config_student.DROPOUT),
        student.classifier
    )

    # Loss & Optimizer
    cls_w    = get_class_weights("train").to(config_student.DEVICE)
    crit_ce  = FocalLoss(weight=cls_w, gamma=2.0)
    optimizer = optim.AdamW(student.parameters(),
                            lr=config_student.LEARNING_RATE,
                            weight_decay=config_student.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  factor=0.1, patience=5, min_lr=1e-7)
    stopper   = utils.EarlyStopping(patience=config_student.EARLY_STOPPING_PATIENCE)

    tr_l, vl_l, tr_a, vl_a = [], [], [], []
    best_p, best_t = None, None

    # ── Training loop ────────────────────────────────────────────────────
    for ep in range(config_student.NUM_EPOCHS):
        tl, ta, tu, tf, _, _ = run_epoch(student, teacher, train_ld, crit_ce, optimizer)
        vl, va, vu, vf, p, t = run_epoch(student, teacher, val_ld,   crit_ce)

        print(f"[{ep:03d}][{args.modal}] "
              f"train loss {tl:.4f} acc {ta:.4f} uar {tu:.4f} f1 {tf:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} uar {vu:.4f} f1 {vf:.4f}")

        tr_l.append(tl); vl_l.append(vl); tr_a.append(ta); vl_a.append(va)
        scheduler.step(vu)

        if stopper.step(vu, student): break
        best_p, best_t = p, t

    # ── Save best student ───────────────────────────────────────────────
    student.load_state_dict(stopper.best_state)
    torch.save(student.state_dict(), SAVE_DIR / "student_best.pth")

    # ── Plots & Confusion matrix ────────────────────────────────────────
    plot_curve(tr_a, vl_a, "Accuracy over Epochs", "Acc",  "acc_student.png")
    plot_curve(tr_l, vl_l, "Loss over Epochs",     "Loss", "loss_student.png")

    cm = confusion_matrix(best_t, best_p)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(SAVE_DIR / "confmat_student.png"); plt.close()
    print("✓ Best student saved →", SAVE_DIR / "student_best.pth")

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
