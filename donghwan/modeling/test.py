import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import config, datasets, models, utils
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np
from utils import custom_collate

SAVE_DIR = "/home/ai/Internship/stressID/Multimodal-Stress-state-recognition/donghwan/modeling/checkpoint"
os.makedirs(SAVE_DIR, exist_ok=True)

def eval_model(model, loader):
    model.eval()
    loss_meter, acc_meter = utils.AverageMeter(), utils.AverageMeter()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(config.DEVICE, non_blocking=True)
            y     = batch["label"].to(config.DEVICE, non_blocking=True)

            logits, _ = model(audio)
            loss = F.cross_entropy(logits, y)
            acc  = utils.accuracy(logits, y)

            loss_meter.update(loss.item(), y.size(0))
            acc_meter.update(acc, y.size(0))
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(y.cpu())

    return loss_meter.avg, acc_meter.avg, torch.cat(all_preds), torch.cat(all_labels)


def plot_metrics(labels, preds):
    # confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(SAVE_DIR, "student_confmat_test.png"))
    plt.close()

    # 기타 지표 계산
    acc       = np.mean(preds.numpy() == labels.numpy())
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall    = recall_score(labels, preds, average='macro', zero_division=0)
    f1        = f1_score(labels, preds, average='macro', zero_division=0)

    metrics = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
    names, values = zip(*metrics.items())

    # 바 차트
    plt.figure()
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("Student Test Performance Metrics")
    plt.ylabel("Score")
    plt.savefig(os.path.join(SAVE_DIR, "student_metrics_test.png"))
    plt.close()


def main():
    # 데이터로더
    test_set = datasets.StressMultimodalDataset("test", nf=16)
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )

    # 모델 로드 (학습 때와 동일하게 Dropout 적용)
    model = models.StudentNet().to(config.DEVICE)
    model.classifier = nn.Sequential(
        nn.Dropout(config.DROPOUT),
        model.classifier
    )
    model.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, "student_best.pth"),
        map_location=config.DEVICE
    ))

    # 평가
    loss, acc, preds, labels = eval_model(model, test_loader)
    print(f"[TEST] loss {loss:.4f} | accuracy {acc:.4f}")

    # 결과 플롯 & 저장
    plot_metrics(labels, preds)


if __name__ == "__main__":
    main()
