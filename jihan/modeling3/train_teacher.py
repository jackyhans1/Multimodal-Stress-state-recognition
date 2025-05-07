import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import config, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy

SAVE_DIR = "/home/ai/Internship/stressID/Multimodal-Stress-state-recognition/jihan/modeling3/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_meter, acc_meter = utils.AverageMeter(), utils.AverageMeter()
    all_preds, all_labels = [], []

    for batch in loader:
        video = batch["video"].to(config.DEVICE, non_blocking=True)
        ecg   = batch["ecg"].to(config.DEVICE, non_blocking=True)
        eda   = batch["eda"].to(config.DEVICE, non_blocking=True)
        rr    = batch["rr"].to(config.DEVICE, non_blocking=True)
        y     = batch["label"].to(config.DEVICE, non_blocking=True)

        logits, _ = model(video, ecg, eda, rr)
        loss = criterion(logits, y)

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
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


def main():
    train_set = datasets.StressMultimodalDataset("train", nf=16)
    val_set   = datasets.StressMultimodalDataset("val", nf=16)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_loader   = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    model = models.TeacherNet().to(config.DEVICE)

    weights = get_class_weights(split="train").to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    stopper   = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_preds, best_labels = None, None

    for epoch in range(config.NUM_EPOCHS):
        tr_loss, tr_acc, tr_uar, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_uar, val_f1, preds, labels = run_epoch(model, val_loader, criterion)

        print(f"[{epoch:03d}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} uar {tr_uar:.4f} f1 {tr_f1:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} uar {val_uar:.4f} f1 {val_f1:.4f}")

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc);    val_accs.append(val_acc)

        if stopper.step(val_uar, model):
            break
        best_preds, best_labels = preds, labels

    model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "teacher_best.pth"))

    plot_metrics(train_accs, val_accs, "Accuracy over Epochs", "Accuracy", "acc_teacher.png")
    plot_metrics(train_losses, val_losses, "Loss over Epochs", "Loss", "loss_teacher.png")

    cm = confusion_matrix(best_labels, best_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(os.path.join(SAVE_DIR, "confmat_teacher.png"))
    plt.close()

    print("Best teacher model and plots saved.")

if __name__ == "__main__":
    main()
