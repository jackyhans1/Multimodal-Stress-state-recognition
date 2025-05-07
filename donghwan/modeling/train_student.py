import os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

import config, datasets, models, utils
from utils import custom_collate, get_class_weights, unweighted_accuracy

# 체크포인트 디렉토리
SAVE_DIR = "/home/ai/Internship/stressID/Multimodal-Stress-state-recognition/donghwan/modeling/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_epoch(student, teacher, loader, criterion_ce, optimizer=None):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()
    teacher.eval()

    loss_meter = utils.AverageMeter()
    acc_meter  = utils.AverageMeter()
    all_preds, all_labels = [], []

    for batch in loader:
        audio = batch["audio"].to(config.DEVICE, non_blocking=True)
        video = batch["video"].to(config.DEVICE, non_blocking=True)
        ecg   = batch["ecg"].to(config.DEVICE, non_blocking=True)
        eda   = batch["eda"].to(config.DEVICE, non_blocking=True)
        rr    = batch["rr"].to(config.DEVICE, non_blocking=True)
        y     = batch["label"].to(config.DEVICE, non_blocking=True)

        # teacher로부터 fused feature만 추출 (grad 없음)
        with torch.no_grad():
            _, teacher_feat, _ = teacher(video, ecg, eda, rr)

        # student 순전파
        logits, student_feat = student(audio)
        loss_ce = criterion_ce(logits, y)

        # feature‐matching distillation
        loss_distill = nn.functional.mse_loss(student_feat, teacher_feat)
        loss = loss_ce + config.ALPHA * loss_distill

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics 업데이트
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
    plt.plot(val_vals,   label='Val')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


def main():
    # 1) 데이터셋 & DataLoader
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

    # 2) 모델 불러오기 & Dropout 적용
    student = models.StudentNet().to(config.DEVICE)
    # classifier 앞에 dropout 삽입
    student.classifier = nn.Sequential(
        nn.Dropout(config.DROPOUT),
        student.classifier
    )

    teacher = models.TeacherNet().to(config.DEVICE)
    teacher_ckpt = os.path.join(SAVE_DIR, "teacher_best.pth")
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=config.DEVICE))

    # 3) 손실함수, 옵티마이저(+ weight decay), 스케줄러, 얼리스토핑
    class_weights = get_class_weights(split="train").to(config.DEVICE)
    criterion_ce  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = optim.Adam(
        student.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',     # UAR이 클수록 좋으므로
        factor=0.1,     # lr ← lr * 0.1
        patience=5
    )
    stopper = utils.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_preds,   best_labels = None, None

    # 4) 학습 루프
    for epoch in range(config.NUM_EPOCHS):
        tr_loss, tr_acc, tr_uar, tr_f1, _, _ = run_epoch(
            student, teacher, train_loader, criterion_ce, optimizer
        )
        val_loss, val_acc, val_uar, val_f1, preds, labels = run_epoch(
            student, teacher, val_loader, criterion_ce
        )

        print(
            f"[{epoch:03d}] "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} uar {tr_uar:.4f} f1 {tr_f1:.4f} | "
            f"val   loss {val_loss:.4f} acc {val_acc:.4f} uar {val_uar:.4f} f1 {val_f1:.4f}"
        )

        train_losses.append(tr_loss); val_losses.append(val_loss)
        train_accs.append(tr_acc);    val_accs.append(val_acc)

        # 스케줄러 & 얼리스토핑
        scheduler.step(val_uar)
        if stopper.step(val_uar, student):
            break
        best_preds, best_labels = preds, labels

    # 5) 베스트 student 저장
    student.load_state_dict(stopper.best_state)
    torch.save(student.state_dict(), os.path.join(SAVE_DIR, "student_best.pth"))

    # 6) 결과 시각화 & 저장
    plot_metrics(train_accs, val_accs, "Accuracy over Epochs", "Accuracy", "student_acc.png")
    plot_metrics(train_losses, val_losses, "Loss over Epochs",     "Loss",     "student_loss.png")

    cm = confusion_matrix(best_labels, best_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig(os.path.join(SAVE_DIR, "student_confmat.png"))
    plt.close()

    print("Best student model and plots saved.")


if __name__ == "__main__":
    main()
