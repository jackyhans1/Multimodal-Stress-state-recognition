import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import config, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning"
)

# 체크포인트 디렉토리
SAVE_DIR = "/home/ai/Internship/stressID/Multimodal-Stress-state-recognition/donghwan/modeling/checkpoint"
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

        # Forward
        logits, _, attn_w = model(video, ecg, eda, rr)
        loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Metrics
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
    plt.plot(val_vals,   label="Val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()

def main():
    # 1) DataLoader
    train_set = datasets.StressMultimodalDataset("train", nf=16)
    val_set   = datasets.StressMultimodalDataset("val",   nf=16)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )

    # 2) Model (+ dropout 추가)
    model = models.TeacherNet().to(config.DEVICE)
    # classifier 앞에 dropout 삽입
    model.classifier = nn.Sequential(
        nn.Dropout(config.DROPOUT),
        model.classifier
    )

    # 3) Loss, Optimizer(+ weight decay), Scheduler, EarlyStopping
    weights   = get_class_weights(split="train").to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',     # UAR 기준이라 'max'
        factor=0.1,     # lr *= 0.1
        patience=5      # 5 에폭 개선 없으면
    )
    stopper = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_preds, best_labels   = None, None

    # 4) Training Loop
    for epoch in range(config.NUM_EPOCHS):
        tr_loss, tr_acc, tr_uar, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_uar, val_f1, preds, labels = run_epoch(model, val_loader, criterion)

        print(
            f"[{epoch:03d}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} uar {tr_uar:.4f} f1 {tr_f1:.4f} | "
            f"val   loss {val_loss:.4f} acc {val_acc:.4f} uar {val_uar:.4f} f1 {val_f1:.4f}"
        )

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc);    val_accs.append(val_acc)

        # scheduler, early stopping
        scheduler.step(val_uar)
        if stopper.step(val_uar, model):
            break
        best_preds, best_labels = preds, labels

    # 5) Save best
    model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "teacher_best.pth"))

    # 6) Plot & Confusion Matrix & Attention Map
    plot_metrics(train_accs, val_accs, "Accuracy over Epochs", "Accuracy", "teacher_acc.png")
    plot_metrics(train_losses, val_losses, "Loss over Epochs", "Loss", "teacher_loss.png")

    cm = confusion_matrix(best_labels, best_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(os.path.join(SAVE_DIR, "teacher_confmat.png"))
    plt.close()

    batch = next(iter(val_loader))
    video, ecg, eda, rr = (
        batch["video"].to(config.DEVICE),
        batch["ecg"].to(config.DEVICE),
        batch["eda"].to(config.DEVICE),
        batch["rr"].to(config.DEVICE),
    )
    model.eval()
    with torch.no_grad():
        _, _, attn_w = model(video, ecg, eda, rr)
    attn_map = attn_w[0].cpu().numpy()
    labels = ["Video","ECG","EDA","RR"]
    plt.figure(figsize=(5,4))
    sns.heatmap(attn_map, annot=True, xticklabels=labels, yticklabels=labels, cmap="viridis")
    plt.title("Modal Attention Map (Val Sample 0)")
    plt.savefig(os.path.join(SAVE_DIR, "teacher_attnmap.png"))
    plt.close()

    print("Best teacher model and plots saved.")

if __name__ == "__main__":
    main()
