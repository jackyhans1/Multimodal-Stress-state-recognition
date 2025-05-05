import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_DIR   = pathlib.Path("/data/StressID/Physiological")
OUT_BASE  = pathlib.Path("/data/StressID")
MODALITIES = ["ECG", "EDA", "RR"]
SAMPLE_RATE = 500         # 500 Hz 원본 샘플링 주기

def load_txt(f: pathlib.Path) -> np.ndarray:
    df = pd.read_csv(f, skiprows=1, names=MODALITIES)
    return df.values.astype("float32")          # (T, 3)

def zscore(sig: np.ndarray) -> np.ndarray:
    mu, sigma = sig.mean(), sig.std()
    return (sig - mu) / (sigma + 1e-8)

def lin_resample(sig: np.ndarray, out_len: int) -> np.ndarray:
    """선형 보간으로 길이를 out_len으로 축소(또는 확대)."""
    n  = sig.shape[0]
    xp = np.linspace(0, 1, n,  endpoint=False)
    xq = np.linspace(0, 1, out_len, endpoint=False)
    return np.interp(xq, xp, sig)

def to_gaf_numpy(sig_norm: np.ndarray) -> np.ndarray:
    """정규화된 1‑D 신호 → GAF(Gramian Angular Field) 행렬."""
    sig_clipped = np.clip(sig_norm, -1, 1)
    theta = np.arccos(sig_clipped)
    return np.cos(theta[:, None] + theta[None, :]).astype("float32")

def sig_to_gaf_channel(sig_1d: np.ndarray) -> np.ndarray:
    """1‑D 신호 전체를 한 장의 GAF(uint8)로 변환."""
    sig = zscore(sig_1d)          # 표준화
    sig = np.tanh(sig / 3)        # 소프트 클리핑
    # ─ 녹음 길이(초)만큼으로 리샘플 → 이미지 한 변
    out_len = max(1, int(np.ceil(len(sig) / SAMPLE_RATE)))
    sig = lin_resample(sig, out_len)
    gaf = to_gaf_numpy(sig)
    img = 255 * (gaf - gaf.min()) / (gaf.max() - gaf.min() + 1e-8)
    return img.astype("uint8")    # (N, N)

def process_one(txt: pathlib.Path):
    subj = txt.parent.name                      
    task = txt.stem.split("_", 1)[1]            
    sig  = load_txt(txt)                        

    for c, name in enumerate(MODALITIES):       # ECG / EDA / RR
        out_dir = OUT_BASE / "jihan" / f"GAF_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        gaf = sig_to_gaf_channel(sig[:, c])
        Image.fromarray(gaf).save(
            out_dir / f"{subj}_{task}.png", optimize=True
        )

def main():
    files = sorted(RAW_DIR.glob("*/*.txt"))
    print(f"{len(files):,} files found")
    for f in tqdm(files, desc="Converting (single‑GAF)"):
        process_one(f)

if __name__ == "__main__":
    main()
