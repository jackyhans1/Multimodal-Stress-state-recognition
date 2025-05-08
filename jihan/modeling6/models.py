import torch
import torch.nn as nn
import config

class Simple2DCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 14
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),          # 1x1
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(256, out_dim)
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)   # (B, out_dim)

class Simple3DCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, (3,3,3), padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(64, 128, (3,3,3), padding=1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(128, out_dim)
    def forward(self, x): # B x C x T x H x W
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vid_backbone = Simple3DCNN(out_dim=config.VIDEO_FEAT_DIM)
        self.ecg_backbone = Simple2DCNN(out_dim=config.PHYSIO_FEAT_DIM)
        self.eda_backbone = Simple2DCNN(out_dim=config.PHYSIO_FEAT_DIM)
        self.rr_backbone  = Simple2DCNN(out_dim=config.PHYSIO_FEAT_DIM)

        total_dim = config.VIDEO_FEAT_DIM + 3 * config.PHYSIO_FEAT_DIM
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_dim, config.NUM_CLASSES)
        )

    def forward(self, video, ecg, eda, rr):
        fv = self.vid_backbone(video)
        fe = self.ecg_backbone(ecg)
        fd = self.eda_backbone(eda)
        fr = self.rr_backbone(rr)
        feat = torch.cat([fv, fe, fd, fr], dim=1)
        logits = self.classifier(feat)
        return logits, feat.detach()

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_backbone = Simple2DCNN(out_dim=config.AUDIO_FEAT_DIM)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.AUDIO_FEAT_DIM, config.NUM_CLASSES)
        )

    def forward(self, audio):
        feat = self.audio_backbone(audio)
        logits = self.classifier(feat)
        return logits, feat
