import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import config, datasets, models, utils
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np
from utils import custom_collate

SAVE_DIR = "/home/ai/Internship/stressID/jihan/modeling2/checkpoint"

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
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(SAVE_DIR, "confmat_test.png"))
    plt.close()

    acc       = np.mean(preds.numpy() == labels.numpy())
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall    = recall_score(labels, preds, average='macro', zero_division=0)
    f1        = f1_score(labels, preds, average='macro', zero_division=0)

    metrics = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure()
    plt.bar(names, values)
    plt.ylim(0, 1)
    plt.title("Test Performance Metrics")
    plt.ylabel("Score")
    plt.savefig(os.path.join(SAVE_DIR, "metrics_test.png"))
    plt.close()

def main():
    test_set = datasets.StressMultimodalDataset("test", nf=16)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate)

    model = models.StudentNet().to(config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "student_best.pth"), map_location=config.DEVICE))

    loss, acc, preds, labels = eval_model(model, test_loader)
    print(f"Test loss {loss:.4f} | Test accuracy {acc:.4f}")

    plot_metrics(labels, preds)

if __name__ == "__main__":
    main()
