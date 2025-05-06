import os, glob
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm

IN_DIR  = Path("/data/StressID/jihan/audio_wav")
OUT_DIR = Path("/data/StressID/jihan/audio_melspectogram")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR   = 16000
N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
TOP_DB      = 80

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

mel_extractor = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
).to(device)

amp2db = T.AmplitudeToDB(top_db=TOP_DB).to(device)

def tensor_to_img(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).cpu().numpy()  # (n_mels, T)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    t_uint8 = (t_norm * 255).astype(np.uint8)
    return Image.fromarray(t_uint8)

wav_paths = glob.glob(str(IN_DIR / "**/*.wav"), recursive=True)

with torch.no_grad():
    for wav_path in tqdm(wav_paths, desc="Converting"):
        wav_path = Path(wav_path)
        out_png  = OUT_DIR / (wav_path.stem + ".png")
        if out_png.exists():
            continue

        waveform, sr = torchaudio.load(wav_path)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.to(device)

        mel_spec = mel_extractor(waveform)  # (1, 128, T)
        logmel_spec = amp2db(mel_spec)
        img = tensor_to_img(logmel_spec)
        img.save(out_png)

print(f"[DONE] {len(wav_paths)} files processed. Output saved in: {OUT_DIR}")
