import os

# Base directories
AUDIO_DIR = "/data/StressID/jihan/audio_melspectogram"
ECG_DIR   = "/data/StressID/jihan/GAF_ECG"
EDA_DIR   = "/data/StressID/jihan/GAF_EDA"
RR_DIR    = "/data/StressID/jihan/GAF_RR"
VIDEO_DIR = "/data/StressID/jihan/video"

CSV_PATH  = "/data/StressID/label_jihan.csv"

# Training hyper‑parameters
NUM_CLASSES = 3
BATCH_SIZE  = 8
NUM_EPOCHS  = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 1e-4
EARLY_STOPPING_PATIENCE = 10

# Feature dimensions
AUDIO_FEAT_DIM  = 256
VIDEO_FEAT_DIM  = 256
PHYSIO_FEAT_DIM = 256

# knowledge distillation weights
ALPHA = 10.0  # audio‑video
BETA  = 10.0  # audio‑ecg
GAMMA = 10.0  # audio‑eda
DELTA = 10.0  # audio‑rr

import torch, os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
