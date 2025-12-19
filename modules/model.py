"""Model."""

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels, feat_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=8192):
        super().__init__()
        # keep a single hidden layer MLP; output dim can be large
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class DINOModel(nn.Module):
    def __init__(self, backbone, projector, use_predictor=False, predictor_kwargs=None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.use_predictor = use_predictor
        if use_predictor:
            pk = predictor_kwargs or {}
            # projector.mlp[-1] is Linear(hidden_dim, out_dim) so .out_features works
            proj_out = (
                projector.mlp[-1].out_features
                if hasattr(projector, "mlp")
                else getattr(projector, "out_dim", None)
            )
            self.predictor = Predictor(in_dim=proj_out, **pk)
        else:
            self.predictor = None

    def forward(self, x):
        feat = self.backbone(x)  # (B, feat_dim)
        proj = self.projector(feat)  # (B, out_dim)
        if self.use_predictor:
            pred = self.predictor(proj)
            out = F.normalize(pred, dim=1)
        else:
            out = F.normalize(proj, dim=1)
        return out
