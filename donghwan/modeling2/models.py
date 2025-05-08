import torch
import torch.nn as nn
import config


class Simple2DCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))                      # (B, out_dim)


class Simple3DCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(0.3),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):                                  # x : (B,C,T,H,W)
        x = self.features(x)
        return self.fc(x.flatten(1))                       # (B, out_dim)

# ─────────────────── 수정된 TeacherNet ─────────────────── #
class TeacherNet(nn.Module):
    """
    단일-모달 Teacher 모델
    (config.set_modal(...) 로 지정된 모달에 맞춰 사용)

    Parameters
    ----------
    modal : {"video","ecg","eda","rr"}
        이 Teacher 가 입력으로 받을 신호.
    out_feat_dim : int, optional
        백본이 출력할 feature 차원. 지정하지 않으면
        config 에 정의된 VIDEO_FEAT_DIM / PHYSIO_FEAT_DIM 을 사용.
    """
    def __init__(self, modal, out_feat_dim=None):
        super().__init__()

        self.modal = modal.lower()
        assert self.modal in ["video", "ecg", "eda", "rr"], \
            f"TeacherNet: unknown modal '{modal}'"

        # feature dim 자동 결정
        if out_feat_dim is None:
            out_feat_dim = (
                config.VIDEO_FEAT_DIM
                if self.modal == "video" else config.PHYSIO_FEAT_DIM
            )

        # ───── backbone 선택 ───── #
        if self.modal == "video":
            # 비디오 : 3D-CNN
            self.backbone = Simple3DCNN(out_dim=out_feat_dim, in_channels=3)
        else:
            # 생체신호(GAF 이미지) : 2D-CNN
            self.backbone = Simple2DCNN(out_dim=out_feat_dim, in_channels=3)

        # ───── classifier ───── #
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(out_feat_dim, config.NUM_CLASSES),
        )

    # ------------------------------------------------------------------ #
    def forward(self, x):
        """
        x : torch.Tensor
            · video        → (B, C, T, H, W)
            · ecg/eda/rr   → (B, C, H, W)
        """
        feat   = self.backbone(x)      # (B, out_feat_dim)
        logits = self.classifier(feat) # (B, num_classes)
        return logits, feat.detach()   # feature 는 distillation 용, gradient stop



class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_backbone = Simple2DCNN(out_dim=config.AUDIO_FEAT_DIM)
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.AUDIO_FEAT_DIM, config.NUM_CLASSES),
        )

    def forward(self, audio):
        feat   = self.audio_backbone(audio)
        logits = self.classifier(feat)
        return logits, feat
