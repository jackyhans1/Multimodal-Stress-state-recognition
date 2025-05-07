# modeling 3 + focal loss
import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

import config, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy, FocalLoss

SAVE_DIR = "/home/ai/Internship/stressID/Multimodal-Stress-state-recognition/jihan/modeling5/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    meter_l, meter_a = utils.AverageMeter(), utils.AverageMeter()
    preds, trues = [], []

    for batch in loader:
        video = batch["video"].to(config.DEVICE, non_blocking=True)
        ecg   = batch["ecg"  ].to(config.DEVICE, non_blocking=True)
        eda   = batch["eda"  ].to(config.DEVICE, non_blocking=True)
        rr    = batch["rr"   ].to(config.DEVICE, non_blocking=True)
        y     = batch["label"].to(config.DEVICE, non_blocking=True)

        logit, _ = model(video, ecg, eda, rr)
        loss = criterion(logit, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = utils.accuracy(logit, y)
        meter_l.update(loss.item(), y.size(0))
        meter_a.update(acc,        y.size(0))
        preds.append(torch.argmax(logit, 1).cpu())
        trues.append(y.cpu())

    preds = torch.cat(preds); trues = torch.cat(trues)
    uar = unweighted_accuracy(preds, trues)
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return meter_l.avg, meter_a.avg, uar, f1, preds, trues

def plot_metrics(tr, vl, title, ylabel, fname):
    plt.figure(); plt.plot(tr, label="Train"); plt.plot(vl, label="Val")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, fname)); plt.close()

def main():
    train_ds = datasets.StressMultimodalDataset("train", nf=16)
    val_ds   = datasets.StressMultimodalDataset("val",   nf=16)

    train_ld = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_ld   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=custom_collate)

    model = models.TeacherNet().to(config.DEVICE)

    cls_w = get_class_weights(split="train").to(config.DEVICE)
    criterion = FocalLoss(weight=cls_w, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LEARNING_RATE,
                            weight_decay=config.WEIGHT_DECAY)
    stopper = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    tr_ls, vl_ls, tr_acc, vl_acc = [], [], [], []
    best_p, best_t = None, None

    for ep in range(config.NUM_EPOCHS):
        tl, ta, tu, tf, _, _ = run_epoch(model, train_ld, criterion, optimizer)
        vl, va, vu, vf, p, t = run_epoch(model, val_ld,   criterion)

        print(f"[{ep:03d}] train loss {tl:.4f} acc {ta:.4f} uar {tu:.4f} f1 {tf:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} uar {vu:.4f} f1 {vf:.4f}")

        tr_ls.append(tl); vl_ls.append(vl); tr_acc.append(ta); vl_acc.append(va)

        if stopper.step(vu, model):
            break
        best_p, best_t = p, t

    model.load_state_dict(stopper.best_state)
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "teacher_best.pth"))

    plot_metrics(tr_acc, vl_acc, "Accuracy over Epochs", "Accuracy", "acc_teacher.png")
    plot_metrics(tr_ls, vl_ls, "Loss over Epochs", "Loss", "loss_teacher.png")

    cm = confusion_matrix(best_t, best_p)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(os.path.join(SAVE_DIR, "confmat_teacher.png")); plt.close()
    print("Best teacher model and plots saved.")

if __name__ == "__main__":
    main()
