"""Model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels, feat_dim=512, kernel_sizes=None, strides=None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [11, 9, 5]
        if strides is None:
            strides = [2, 2, 2]
        channels = [64, 128, 256]
        layers = []
        ch_in = in_channels
        for ch_out, k, s in zip(channels, kernel_sizes, strides):
            layers.extend([
                nn.Conv1d(ch_in, ch_out, kernel_size=k, stride=s, padding=k // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            ])
            ch_in = ch_out
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], feat_dim),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock1d(nn.Module):
    """Residual block for 1D convolution with optional downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ConvNetV2(nn.Module):
    """Improved 1D ConvNet with multi-scale stem, residual blocks, and attention pooling.

    Architecture:
      1. Multi-scale stem: parallel conv branches at different kernel sizes, concatenated
      2. Residual blocks: conv pairs with skip connections and stride-2 downsampling
      3. Temporal attention pooling: learnable weights over time steps

    Input: (B, C, T) -> Output: (B, feat_dim)
    Supports return_attention=True for interpretability.
    """

    def __init__(self, in_channels, feat_dim=256,
                 stem_channels=32, stem_kernel_sizes=(7, 15, 31),
                 block_channels=(128, 256), block_kernel_sizes=(9, 5),
                 attn_hidden=64):
        super().__init__()

        # Multi-scale stem: parallel branches capture features at different timescales
        self.stem_branches = nn.ModuleList()
        for k in stem_kernel_sizes:
            self.stem_branches.append(nn.Sequential(
                nn.Conv1d(in_channels, stem_channels, k, padding=k // 2),
                nn.BatchNorm1d(stem_channels),
                nn.ReLU(inplace=True),
            ))
        stem_out = stem_channels * len(stem_kernel_sizes)

        # Residual blocks with stride-2 downsampling
        self.blocks = nn.ModuleList()
        ch_in = stem_out
        for ch_out, k in zip(block_channels, block_kernel_sizes):
            self.blocks.append(ResBlock1d(ch_in, ch_out, kernel_size=k, stride=2))
            ch_in = ch_out

        self.final_channels = block_channels[-1]

        # Temporal attention pooling
        self.attn_net = nn.Sequential(
            nn.Linear(self.final_channels, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

        # Output projection
        self.head = nn.Linear(self.final_channels, feat_dim)

    def forward(self, x, return_attention=False):
        # Multi-scale stem
        branches = [branch(x) for branch in self.stem_branches]
        x = torch.cat(branches, dim=1)  # (B, stem_out, T)

        # Residual blocks
        for block in self.blocks:
            x = block(x)  # (B, C, T') progressively smaller

        # Temporal attention pooling
        x = x.transpose(1, 2)  # (B, T', C)
        attn_logits = self.attn_net(x).squeeze(-1)  # (B, T')
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, T')
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, C)

        feat = self.head(x)  # (B, feat_dim)

        if return_attention:
            return feat, attn_weights
        return feat


class ConvNet2D(nn.Module):
    def __init__(self, in_channels=1, feat_dim=256, base_channels=64):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, feat_dim),
        )

    def forward(self, x):
        return self.net(x)


class ViT1D(nn.Module):
    """1D Vision Transformer for time-series.

    Input: (B, C, T) → Output: (B, feat_dim)
    Handles variable T by padding to nearest multiple of patch_size.
    """

    def __init__(self, in_channels, feat_dim=256, patch_size=25,
                 d_model=192, n_heads=4, n_layers=4, dropout=0.0,
                 use_mask_token=False):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model

        # Patch embedding: flatten (C, patch_size) → Linear → d_model
        self.patch_embed = nn.Linear(in_channels * patch_size, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable [MASK] token for masked patch prediction
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None

        # Transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feat_dim)

    def _sinusoidal_pos_encoding(self, n_positions, d_model, device):
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(n_positions, d_model, device=device)
        position = torch.arange(0, n_positions, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (n_positions, d_model)

    def forward(self, x, mask=None, return_patch_tokens=False):
        B, C, T = x.shape

        # Pad T to multiple of patch_size if needed
        remainder = T % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = F.pad(x, (0, pad_len))
            T = T + pad_len

        n_patches = T // self.patch_size

        # Reshape: (B, C, T) → (B, n_patches, C * patch_size)
        x = x.reshape(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, C * self.patch_size)

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Replace masked patches with learnable mask token
        if mask is not None and self.mask_token is not None:
            w = mask.unsqueeze(-1).float()  # (B, n_patches, 1)
            x = x * (1.0 - w) + self.mask_token.expand(B, n_patches, -1) * w

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + n_patches, d_model)

        # Add sinusoidal positional encoding
        pos_enc = self._sinusoidal_pos_encoding(1 + n_patches, self.d_model, x.device)
        x = x + pos_enc.unsqueeze(0)

        # Transformer encoder
        x = self.encoder(x)  # (B, 1 + n_patches, d_model)

        # Extract CLS token → head
        cls_out = x[:, 0]  # (B, d_model)
        cls_out = self.norm(cls_out)
        feat = self.head(cls_out)  # (B, feat_dim)

        if return_patch_tokens:
            patch_tokens = x[:, 1:]  # (B, n_patches, d_model)
            return feat, patch_tokens
        return feat


class ViT2D(nn.Module):
    """2D Vision Transformer for time-frequency images.

    Input: (B, C, F, T) → Output: (B, feat_dim)
    C = spatial channels (e.g. 52 parcels), F = frequency bins, T = time samples.
    Patches tile the (F, T) spatial dimensions.
    """

    def __init__(self, in_channels, feat_dim=256, patch_size=(5, 25),
                 d_model=192, n_heads=4, n_layers=4, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size  # (pf, pt)
        self.d_model = d_model

        pf, pt = patch_size
        self.patch_embed = nn.Linear(in_channels * pf * pt, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, feat_dim)

    def _sinusoidal_pos_encoding(self, n_positions, d_model, device):
        pe = torch.zeros(n_positions, d_model, device=device)
        position = torch.arange(0, n_positions, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        B, C, Freq, T = x.shape
        pf, pt = self.patch_size

        # Pad Freq and T to multiples of patch_size if needed
        pad_f = (pf - Freq % pf) % pf
        pad_t = (pt - T % pt) % pt
        if pad_f > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_f))
            Freq = Freq + pad_f
            T = T + pad_t

        nf = Freq // pf
        nt = T // pt
        n_patches = nf * nt

        # (B, C, Freq, T) → (B, C, nf, pf, nt, pt) → (B, nf, nt, C, pf, pt) → (B, n_patches, C*pf*pt)
        x = x.reshape(B, C, nf, pf, nt, pt)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_patches, C * pf * pt)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        pos_enc = self._sinusoidal_pos_encoding(1 + n_patches, self.d_model, x.device)
        x = x + pos_enc.unsqueeze(0)

        x = self.encoder(x)

        cls_out = x[:, 0]
        cls_out = self.norm(cls_out)
        return self.head(cls_out)


class FilterbankNet(nn.Module):
    """Learnable filterbank backbone with attention pooling.

    A shared bank of FIR filters is applied to each channel independently,
    followed by channel mixing, temporal convolution, and attention-based
    temporal pooling that preserves temporal structure.

    Input: (B, C, T) → Output: (B, feat_dim)
    """

    def __init__(self, in_channels, feat_dim=256, n_filters=8,
                 filter_length=65, hidden_dim=128, n_queries=4):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.n_queries = n_queries

        # Shared filterbank: same filters for all channels
        self.filterbank = nn.Conv1d(1, n_filters, filter_length,
                                     padding=filter_length // 2)

        # Envelope extraction: rectify → smooth → log compress
        self.envelope_smooth = nn.Conv1d(
            in_channels * n_filters, in_channels * n_filters, 9,
            padding=4, groups=in_channels * n_filters, bias=False,
        )
        nn.init.constant_(self.envelope_smooth.weight, 1.0 / 9.0)
        self.fb_norm = nn.BatchNorm1d(in_channels * n_filters)

        # Channel mixing: spatial-spectral → hidden_dim
        self.channel_mix = nn.Sequential(
            nn.Conv1d(in_channels * n_filters, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Temporal conv with downsampling
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 9, stride=2, padding=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Multi-query attention pooling
        self.attn_queries = nn.Parameter(torch.zeros(1, n_queries, hidden_dim))
        nn.init.trunc_normal_(self.attn_queries, std=0.02)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(n_queries * hidden_dim)

        # Output head
        self.head = nn.Linear(n_queries * hidden_dim, feat_dim)

    def forward(self, x):
        B, C, T = x.shape

        # Shared filterbank: (B, C, T) → (B*C, 1, T) → (B*C, n_f, T) → (B, C*n_f, T)
        x = x.reshape(B * C, 1, T)
        x = self.filterbank(x)
        x = x.reshape(B, C * self.n_filters, T)

        # Envelope extraction: rectify → smooth → log compress
        x = x.abs()
        x = self.envelope_smooth(x)
        x = torch.log1p(x)
        x = self.fb_norm(x)

        # Channel mixing
        x = self.channel_mix(x)  # (B, hidden_dim, T)

        # Temporal conv
        x = self.temporal(x)  # (B, hidden_dim, T')

        # Multi-query attention pooling
        x = x.transpose(1, 2)  # (B, T', hidden_dim)
        keys = self.attn_key(x)  # (B, T', hidden_dim)
        attn = torch.bmm(
            self.attn_queries.expand(B, -1, -1),
            keys.transpose(1, 2)
        )  # (B, K, T')
        attn = attn / (keys.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, x)  # (B, K, hidden_dim)
        out = out.reshape(B, -1)  # (B, K * hidden_dim)
        out = self.attn_norm(out)

        return self.head(out)  # (B, feat_dim)


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

    def forward(self, x, mask=None, return_patch_tokens=False):
        # Check if backbone supports masking (ViT with mask_token)
        supports_tokens = hasattr(self.backbone, 'mask_token') and self.backbone.mask_token is not None
        if supports_tokens and (mask is not None or return_patch_tokens):
            feat, patch_tokens = self.backbone(x, mask=mask, return_patch_tokens=True)
        else:
            feat = self.backbone(x)  # (B, feat_dim)
            patch_tokens = None

        proj = self.projector(feat)  # (B, out_dim)
        if self.use_predictor:
            pred = self.predictor(proj)
            out = F.normalize(pred, dim=1)
        else:
            out = F.normalize(proj, dim=1)

        if return_patch_tokens:
            return out, patch_tokens
        return out
