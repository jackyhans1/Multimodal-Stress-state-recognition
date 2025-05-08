import os, torch, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader
import torchvision.transforms.functional as TF
import config

# 기본 2D 이미지 전처리
_img_tf = transforms.Compose([
    transforms.ToTensor(),   # (3,H,W) float32 [0,1]
])

def _select_indices(total: int, nf: int = 16):
    step = total / nf
    return [int(i * step) for i in range(nf)]

# 이미지 또는 프레임을 패딩해 주어진 목표 크기에 맞춤
def pad_to(tensor: torch.Tensor, target_h: int, target_w: int):
    _, h, w = tensor.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    return TF.pad(tensor, [0, 0, pad_w, pad_h])  # 오른쪽, 아래로만 패딩

class StressMultimodalDataset(Dataset):
    """모달리티가 없으면 zero‑tensor로 채워 넣음 + 배치 내 최대 크기로 패딩"""
    def __init__(self, split: str, nf: int = 16):
        self.nf = nf
        df = pd.read_csv(config.CSV_PATH)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self):  return len(self.df)

    # helper: 비디오 로더 (C,T,H,W)
    def _load_video(self, path: str):
        vr   = VideoReader(path, 'video')
        meta = vr.get_metadata()['video']
        total = meta.get('frames', int(meta['fps'][0] * meta['duration'][0]))
        want  = set(_select_indices(total, self.nf))

        frames = []
        max_h, max_w = 0, 0
        for i, frm in enumerate(vr):
            if i in want:
                img = frm['data'].float() / 255.
                c, h, w = img.shape
                max_h, max_w = max(max_h, h), max(max_w, w)
                frames.append(img)
                if len(frames) == self.nf: break

        if len(frames) < self.nf:
            while len(frames) < self.nf:
                frames.append(frames[-1])

        # 다시 패딩 기준 정하기
        for f in frames:
            _, h, w = f.shape
            max_h, max_w = max(max_h, h), max(max_w, w)

        padded = [pad_to(f, max_h, max_w) for f in frames]
        return torch.stack(padded, dim=1)  # (3,T,H,W)

    # __getitem__
    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        name  = row['subject/task']
        label = torch.tensor(int(row['affect3-class']), dtype=torch.long)

        p_audio = os.path.join(config.AUDIO_DIR, f'{name}.png')
        p_video = os.path.join(config.VIDEO_DIR,  f'{name}.mp4')
        p_ecg   = os.path.join(config.ECG_DIR,   f'{name}.png')
        p_eda   = os.path.join(config.EDA_DIR,   f'{name}.png')
        p_rr    = os.path.join(config.RR_DIR,    f'{name}.png')

        # 이미지 로딩 (원본 크기 유지)
        def load_image(path):
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                return _img_tf(img)
            return None

        audio = load_image(p_audio)
        ecg   = load_image(p_ecg)
        eda   = load_image(p_eda)
        rr    = load_image(p_rr)

        # 가장 큰 H, W 계산
        hws = [x.shape[1:] for x in [audio, ecg, eda, rr] if x is not None]
        target_h = max([h for h, _ in hws]) if hws else 112
        target_w = max([w for _, w in hws]) if hws else 112

        def pad_or_zero(t):
            if t is None:
                return torch.zeros(3, target_h, target_w)
            return pad_to(t, target_h, target_w)

        audio = pad_or_zero(audio)
        ecg   = pad_or_zero(ecg)
        eda   = pad_or_zero(eda)
        rr    = pad_or_zero(rr)

        # 비디오
        if os.path.exists(p_video):
            video = self._load_video(p_video)
        else:
            video = torch.zeros(3, self.nf, target_h, target_w)

        return {
            "label": label,
            "audio": audio,
            "video": video,
            "ecg":   ecg,
            "eda":   eda,
            "rr":    rr,
        }
