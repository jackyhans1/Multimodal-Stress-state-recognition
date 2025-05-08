# ───────────────────────────────────────── config.py ────────────────────────────
import os, torch
from pathlib import Path

# ─────────────────── 1) 공통 경로 (Raw 데이터 저장 위치) ──────────────────── #
DATA_ROOT = Path("/data/StressID/jihan")

AUDIO_DIR = DATA_ROOT / "audio_melspectogram"
ECG_DIR   = DATA_ROOT / "GAF_ECG"
EDA_DIR   = DATA_ROOT / "GAF_EDA"
RR_DIR    = DATA_ROOT / "GAF_RR"
VIDEO_DIR = DATA_ROOT / "video"

CSV_PATH  = DATA_ROOT.parent / "label_jihan.csv"     # /data/StressID/label_jihan.csv

# ─────────────────── 2) 학습 하이퍼파라미터(공통) ────────────────────── #
NUM_CLASSES              = 3
BATCH_SIZE               = 4
NUM_EPOCHS               = 50 
LEARNING_RATE            = 1e-3
EARLY_STOPPING_PATIENCE  = 7

# Feature dimension (백본이 출력하는 임베딩 크기)
AUDIO_FEAT_DIM  = 256
VIDEO_FEAT_DIM  = 256
PHYSIO_FEAT_DIM = 256          # ECG / EDA / RR 동일

# Knowledge-distillation 가중치
ALPHA = 1.0   # audio ←→ video
BETA  = 1.0   # audio ←→ ecg
GAMMA = 1.0   # audio ←→ eda
DELTA = 1.0   # audio ←→ rr

# 규제 관련
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4

# ─────────────────── 3) **모달별** 설정을 하나로 묶기 ─────────────────── #
#     train_teacher.py / train_student.py 실행 전에
#     set_modal("video") 처럼 한 번만 호출해 주세요.
INPUT_MODAL      : str   # 현재 Teacher가 입력받는 modal
SAVE_SUBDIR      : Path  # checkpoints/<modal>/ …
TEACHER_FEAT_DIM : int

# 프로젝트 루트(이 파일이 위치한 폴더) 기준으로 checkpoints 저장
PROJECT_ROOT = Path(__file__).resolve().parent

def set_modal(modal: str):
    """
    modal ∈ {"video","ecg","eda","rr"}
    Teacher가 입력으로 받을 모달 이름을 설정하고,
    모달별 체크포인트 폴더∙feature dim 등을 전역 변수에 기록합니다.
    """
    modal = modal.lower()
    assert modal in {"video", "ecg", "eda", "rr"}, f"Unknown modal '{modal}'"

    global INPUT_MODAL, SAVE_SUBDIR, TEACHER_FEAT_DIM

    INPUT_MODAL = modal
    SAVE_SUBDIR = PROJECT_ROOT / "checkpoints" / modal   # ← 절대경로로 변경
    SAVE_SUBDIR.mkdir(parents=True, exist_ok=True)

    TEACHER_FEAT_DIM = VIDEO_FEAT_DIM if modal == "video" else PHYSIO_FEAT_DIM

    # (선택) 모달별 맞춤 하이퍼파라미터를 여기서 조정할 수도 있습니다.
    # 예) ECG는 러닝레이트를 낮게:
    # global LEARNING_RATE
    # if modal == "ecg":
    #     LEARNING_RATE = 5e-6


# ─────────────────── 4) Device 지정 ─────────────────────────── #
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # 필요 시 변경
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────
