import torch
import torch.nn as nn
import config
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models import resnet18, ResNet18_Weights


class Pretrained3DCNN(nn.Module):
    """
    Pretrained 3D ResNet backbone for video modality.
    """
    def __init__(self, out_dim, pretrained=True):
        super().__init__()
        # Load pretrained R3D-18 weights
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        base_model = r3d_18(weights=weights)
        # Remove classification head
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # outputs (B, 512,1,1,1)
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):  # x: (B, C, T, H, W)
        x = self.features(x)            # (B, 512,1,1,1)
        x = x.flatten(1)                # (B, 512)
        return self.fc(x)               # (B, out_dim)


class Pretrained2DCNN(nn.Module):
    """
    Pretrained ResNet-18 backbone for 2D modalities (ECG, EDA, RR, Audio).
    """
    def __init__(self, out_dim, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-18 weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)
        # Remove classification head
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # (B, 512,1,1)
        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):  # x: (B, C, H, W)
        x = self.features(x)            # (B, 512,1,1)
        x = x.flatten(1)                # (B, 512)
        return self.fc(x)               # (B, out_dim)


class ModalAttentionFusion(nn.Module):
    """Multi-head self-attention fusion of modality features."""
    def __init__(self, feat_dim, num_modalities=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(dropout)
        )

    def forward(self, modal_feats):
        # modal_feats: (B, M, D)
        # Transpose to (M, B, D) for attention
        x = modal_feats.transpose(0, 1)
        # Self-attention
        attn_out, attn_weights = self.attn(x, x, x)  # (M,B,D), weights: (B, M, M)
        # Feed-forward + residual
        out = self.ffn(attn_out) + attn_out          # (M, B, D)
        # Fuse tokens by averaging across modalities
        fused = out.transpose(0,1).mean(dim=1)       # (B, D)
        return fused, attn_weights


class TeacherNet(nn.Module):
    """Teacher network using pretrained backbones and attention fusion."""
    def __init__(self):
        super().__init__()
        # Pretrained backbones
        self.vid_backbone = Pretrained3DCNN(out_dim=config.VIDEO_FEAT_DIM)
        self.ecg_backbone = Pretrained2DCNN(out_dim=config.PHYSIO_FEAT_DIM)
        self.eda_backbone = Pretrained2DCNN(out_dim=config.PHYSIO_FEAT_DIM)
        self.rr_backbone  = Pretrained2DCNN(out_dim=config.PHYSIO_FEAT_DIM)

        # Attention fusion block (feat_dim must match VIDEO_FEAT_DIM == PHYSIO_FEAT_DIM)
        self.attn_fusion = ModalAttentionFusion(
            feat_dim=config.VIDEO_FEAT_DIM,
            num_modalities=4,
            num_heads=4,
            dropout=0.1
        )
        # Classifier after fusion
        self.classifier = nn.Linear(config.VIDEO_FEAT_DIM, config.NUM_CLASSES)

    def forward(self, video, ecg, eda, rr):
        fv = self.vid_backbone(video)  # (B, D)
        fe = self.ecg_backbone(ecg)     # (B, D)
        fd = self.eda_backbone(eda)     # (B, D)
        fr = self.rr_backbone(rr)       # (B, D)
        modal_feats = torch.stack([fv, fe, fd, fr], dim=1)  # (B, M=4, D)
        fused_feat, attn_w = self.attn_fusion(modal_feats)
        logits = self.classifier(fused_feat)
        return logits, fused_feat.detach(), attn_w


class StudentNet(nn.Module):
    """Student network using pretrained 2D CNN for audio."""
    def __init__(self):
        super().__init__()
        self.audio_backbone = Pretrained2DCNN(out_dim=config.AUDIO_FEAT_DIM)
        self.classifier = nn.Linear(config.AUDIO_FEAT_DIM, config.NUM_CLASSES)

    def forward(self, audio):
        feat = self.audio_backbone(audio)
        logits = self.classifier(feat)
        return logits, feat
