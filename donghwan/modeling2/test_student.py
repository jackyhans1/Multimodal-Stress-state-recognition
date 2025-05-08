# ───────────────────────────── test_student.py ─────────────────────────────
import os, argparse, torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np

import config_student as config          # ← modal 설정용
import datasets, models, utils
from utils import custom_collate, unweighted_accuracy

# ────────────────────────────── CLI 옵션 ────────────────────────────── #
parser = argparse.ArgumentParser()
parser.add_argument("--modal",
                    choices=["video", "ecg", "eda", "rr", "ensemble"],
                    required=True,
                    help="'ensemble' - 4개 Student 평균 / "
                         "그 외 - 단일-modal Student 평가")
args = parser.parse_args()

# ──────────────────────── 평가용 데이터로더 준비 ────────────────────── #
config.set_modal("video")                       # 아무 모달이나; 데이터 경로만 쓰임
test_ds = datasets.StressMultimodalDataset("test", nf=16)
test_ld = DataLoader(
    test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, collate_fn=custom_collate
)

# ─────────────────────── ① Student(들) 로드 ──────────────────────── #
def load_student(modal: str):
    """
    저장된 checkoints/<modal>/student_best.pth 를 읽어서
    학습 때와 동일한 구조( Dropout + 원래 classifier )로 복원.
    """
    config.set_modal(modal)
    net = models.StudentNet().to(config.DEVICE)

    # 학습 시 감싸 두었던 Dropout 래핑 복원 ------------------------
    net.classifier = torch.nn.Sequential(
        torch.nn.Dropout(config.DROPOUT),  # index 0 : 바깥 Dropout
        net.classifier                     # index 1 : 안쪽 Dropout + Linear
    )
    # -------------------------------------------------------------

    ckpt = config.SAVE_SUBDIR / "student_best.pth"
    net.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    net.eval()
    return net


if args.modal == "ensemble":
    MODALS = ["video", "ecg", "eda", "rr"]
    students = {m: load_student(m) for m in MODALS}
else:                                            # 단일 모달
    student = load_student(args.modal)

# ───────────────────────────── 평가 루프 ──────────────────────────── #
loss_meter, acc_meter = utils.AverageMeter(), utils.AverageMeter()
all_prob, all_true = [], []

with torch.no_grad():
    for batch in test_ld:
        audio = batch["audio"].to(config.DEVICE)
        y     = batch["label"].to(config.DEVICE)

        if args.modal == "ensemble":
            probs = []
            for net in students.values():
                logits, _ = net(audio)
                probs.append(torch.softmax(logits, dim=1))
            logits = torch.log(torch.stack(probs).mean(0))      # ← 평균 후 log 변환
        else:
            logits, _ = student(audio)

        loss = F.cross_entropy(logits, y)
        acc  = utils.accuracy(logits, y)

        loss_meter.update(loss.item(), y.size(0))
        acc_meter .update(acc,        y.size(0))
        all_prob .append(torch.softmax(logits, 1).cpu())
        all_true .append(y.cpu())

# ───────────────────────────── 메트릭 계산 ─────────────────────────── #
prob   = torch.cat(all_prob)
pred   = torch.argmax(prob, dim=1)
true   = torch.cat(all_true)

uar  = unweighted_accuracy(pred, true)
f1   = f1_score(true, pred, average="macro", zero_division=0)
prec = precision_score(true, pred, average="macro", zero_division=0)
rec  = recall_score(true, pred, average="macro", zero_division=0)
acc  = np.mean(pred.numpy() == true.numpy())

print(f"\n[{args.modal.upper()}] Test Loss {loss_meter.avg:.4f} | "
      f"ACC {acc:.3f}  UAR {uar:.3f}  F1 {f1:.3f}")

# ───────────────────────────── 시각화 / 저장 ─────────────────────── #
# 저장 폴더 결정
if args.modal == "ensemble":
    out_dir = Path(config.PROJECT_ROOT) / "checkpoints" / "ensemble"
else:
    out_dir = config.SAVE_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

# Confusion-Matrix
cm = confusion_matrix(true, pred)
ConfusionMatrixDisplay(cm).plot()
plt.title(f"{args.modal.upper()} – Confusion Matrix")
plt.savefig(out_dir / "confmat_test.png")
plt.close()

# 막대그래프
metrics = {"Accuracy":acc, "Precision":prec, "Recall":rec, "F1":f1}
plt.figure()
plt.bar(list(metrics.keys()), list(metrics.values()))
plt.ylim(0,1); plt.title(f"{args.modal.upper()} – Test Metrics")
plt.ylabel("Score")
plt.savefig(out_dir / "metrics_test.png")
plt.close()

print("✓ 결과 저장 →", out_dir)
# ──────────────────────────────────────────────────────────────────── #
