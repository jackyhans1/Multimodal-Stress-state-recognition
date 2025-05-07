import torch, torch.nn as nn
import config

class BasicBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = nn.functional.relu(out + identity)
        return out


class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,1,1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        self.downsample = None
        if stride != (1,1,1) or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = nn.functional.relu(out + identity)
        return out

class Simple2DCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = BasicBlock2D(32, 32)
        self.layer2 = nn.Sequential(
            BasicBlock2D(32, 64, stride=2),   # ↓ H,W /2
            BasicBlock2D(64, 64)
        )
        self.layer3 = nn.Sequential(
            BasicBlock2D(64, 128, stride=2),  # ↓ /2
            BasicBlock2D(128,128)
        )
        self.layer4 = nn.Sequential(
            BasicBlock2D(128,256, stride=2),  # ↓ /2
            BasicBlock2D(256,256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

class Simple3DCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.layer1 = BasicBlock3D(32, 32)
        self.layer2 = nn.Sequential(
            BasicBlock3D(32, 64, stride=(1,2,2)),   # T same / H,W half
            BasicBlock3D(64, 64)
        )
        self.layer3 = nn.Sequential(
            BasicBlock3D(64,128, stride=(2,2,2)),   # T,H,W half
            BasicBlock3D(128,128)
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):  # (B,C,T,H,W)
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vid_backbone = Simple3DCNN(out_dim=config.VIDEO_FEAT_DIM)
        self.ecg_backbone = Simple2DCNN(out_dim=config.PHYSIO_FEAT_DIM)
        self.eda_backbone = Simple2DCNN(out_dim=config.PHYSIO_FEAT_DIM)
        self.rr_backbone  = Simple2DCNN(out_dim=config.PHYSIO_FEAT_DIM)

        feat_dim = config.VIDEO_FEAT_DIM + 3 * config.PHYSIO_FEAT_DIM
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, config.NUM_CLASSES)
        )

    def forward(self, video, ecg, eda, rr):
        fv = self.vid_backbone(video)
        fe = self.ecg_backbone(ecg)
        fd = self.eda_backbone(eda)
        fr = self.rr_backbone(rr)
        feat = torch.cat([fv, fe, fd, fr], dim=1)
        return self.classifier(feat), feat.detach()

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
        return self.classifier(feat), feat
