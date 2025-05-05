import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import config, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy

SAVE_DIR = "/home/ai/Internship/stressID/jihan/modeling2/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_epoch(student, teacher, loader, criterion_ce, optimizer=None):
    train = optimizer is not None
    student.train() if train else student.eval()
    teacher.eval()

    loss_meter, acc_meter = utils.AverageMeter(), utils.AverageMeter()
    all_preds, all_labels = [], []

    for batch in loader:
        audio = batch["audio"].to(config.DEVICE, non_blocking=True)
        video = batch["video"].to(config.DEVICE, non_blocking=True)
        ecg   = batch["ecg"].to(config.DEVICE, non_blocking=True)
        eda   = batch["eda"].to(config.DEVICE, non_blocking=True)
        rr    = batch["rr"].to(config.DEVICE, non_blocking=True)
        y     = batch["label"].to(config.DEVICE, non_blocking=True)

        with torch.no_grad():
            _, teacher_feat = teacher(video, ecg, eda, rr)

        logits, student_feat = student(audio)
        loss_ce = criterion_ce(logits, y)

        idx = 0
        fv_dim = config.VIDEO_FEAT_DIM
        fp_dim = config.PHYSIO_FEAT_DIM
        f_video = teacher_feat[:, idx:idx+fv_dim]; idx += fv_dim
        f_ecg   = teacher_feat[:, idx:idx+fp_dim]; idx += fp_dim
        f_eda   = teacher_feat[:, idx:idx+fp_dim]; idx += fp_dim
        f_rr    = teacher_feat[:, idx:idx+fp_dim]

        mse = nn.functional.mse_loss
        loss_distill = (
            config.ALPHA * mse(student_feat, f_video) +
            config.BETA  * mse(student_feat, f_ecg)   +
            config.GAMMA * mse(student_feat, f_eda)   +
            config.DELTA * mse(student_feat, f_rr)
        )
        loss = loss_ce + loss_distill

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = utils.accuracy(logits, y)
        loss_meter.update(loss.item(), y.size(0))
        acc_meter.update(acc, y.size(0))
        all_preds.append(torch.argmax(logits, dim=1).cpu())
        all_labels.append(y.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    uar = unweighted_accuracy(all_preds, all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return loss_meter.avg, acc_meter.avg, uar, f1, all_preds, all_labels

def plot_metrics(train_vals, val_vals, title, ylabel, filename):
    plt.figure()
    plt.plot(train_vals, label='Train')
    plt.plot(val_vals, label='Val')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

def main():
    train_set = datasets.StressMultimodalDataset("train", nf=16)
    val_set   = datasets.StressMultimodalDataset("val", nf=16)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=custom_collate)
    val_loader   = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate)

    student = models.StudentNet().to(config.DEVICE)
    teacher = models.TeacherNet().to(config.DEVICE)
    teacher.load_state_dict(torch.load(os.path.join(SAVE_DIR, "teacher_best.pth"), map_location=config.DEVICE))

    class_weights = get_class_weights(split="train").to(config.DEVICE)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(student.parameters(), lr=config.LEARNING_RATE)
    stopper = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_preds, best_labels = None, None

    for epoch in range(config.NUM_EPOCHS):
        tr_loss, tr_acc, tr_uar, tr_f1, _, _ = run_epoch(student, teacher, train_loader, criterion_ce, optimizer)
        val_loss, val_acc, val_uar, val_f1, preds, labels = run_epoch(student, teacher, val_loader, criterion_ce)

        print(f"[{epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} uar {tr_uar:.4f} f1 {tr_f1:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} uar {val_uar:.4f} f1 {val_f1:.4f}")

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc);    val_accs.append(val_acc)

        if stopper.step(val_uar, student):
            break
        best_preds, best_labels = preds, labels

    student.load_state_dict(stopper.best_state)
    torch.save(student.state_dict(), os.path.join(SAVE_DIR, "student_best.pth"))

    plot_metrics(train_accs, val_accs, "Accuracy over Epochs", "Accuracy", "acc_student.png")
    plot_metrics(train_losses, val_losses, "Loss over Epochs", "Loss", "loss_student.png")

    cm = confusion_matrix(best_labels, best_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(os.path.join(SAVE_DIR, "confmat_student.png"))
    plt.close()

    print("Best student model and plots saved.")

if __name__ == "__main__":
    main()
