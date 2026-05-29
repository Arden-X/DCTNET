from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexMorletWaveletLayer1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 65,
        omega0: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.omega0 = omega0
        self.eps = eps

        self.u_r = nn.Parameter(torch.zeros(out_channels))
        self.u_i = nn.Parameter(torch.zeros(out_channels))
        self.log_s_r = nn.Parameter(torch.zeros(out_channels))
        self.log_s_i = nn.Parameter(torch.zeros(out_channels))

        half = kernel_size // 2
        self.register_buffer("tau", torch.linspace(-half, half, steps=kernel_size), persistent=False)

    def _build_wavelet_bank(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.tau.to(device=device, dtype=dtype).unsqueeze(0)
        s_r = torch.exp(self.log_s_r).unsqueeze(1).to(dtype=dtype) + self.eps
        s_i = torch.exp(self.log_s_i).unsqueeze(1).to(dtype=dtype) + self.eps
        u_r = self.u_r.unsqueeze(1).to(dtype=dtype)
        u_i = self.u_i.unsqueeze(1).to(dtype=dtype)

        tau_r = (tau - u_r) / s_r
        tau_i = (tau - u_i) / s_i
        psi_r = (1.0 / torch.sqrt(s_r)) * torch.exp(-0.5 * tau_r.square()) * torch.cos(self.omega0 * tau_r)
        psi_i = (1.0 / torch.sqrt(s_i)) * torch.exp(-0.5 * tau_i.square()) * torch.cos(self.omega0 * tau_i)

        psi_r = psi_r.unsqueeze(1).expand(-1, self.in_channels, -1).contiguous()
        psi_i = psi_i.unsqueeze(1).expand(-1, self.in_channels, -1).contiguous()
        return psi_r, psi_i

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_r.shape != x_i.shape:
            raise ValueError(f"x_r and x_i shape mismatch: {x_r.shape} vs {x_i.shape}")
        psi_r, psi_i = self._build_wavelet_bank(x_r.device, x_r.dtype)
        pad = self.kernel_size // 2
        y_r = F.conv1d(x_r, psi_r, padding=pad) - F.conv1d(x_i, psi_i, padding=pad)
        y_i = F.conv1d(x_r, psi_i, padding=pad) + F.conv1d(x_i, psi_r, padding=pad)
        return y_r, y_i


class ComplexMorletQuadratureLayer1d(nn.Module):
    """Morlet front-end with cosine/sine quadrature kernels for stricter CV-Morlet tests."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 65,
        omega0: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.omega0 = omega0
        self.eps = eps

        self.u = nn.Parameter(torch.zeros(out_channels))
        self.log_s = nn.Parameter(torch.zeros(out_channels))
        half = kernel_size // 2
        self.register_buffer("tau", torch.linspace(-half, half, steps=kernel_size), persistent=False)

    def _build_wavelet_bank(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = self.tau.to(device=device, dtype=dtype).unsqueeze(0)
        scale = torch.exp(self.log_s).unsqueeze(1).to(dtype=dtype) + self.eps
        shift = self.u.unsqueeze(1).to(dtype=dtype)
        z = (tau - shift) / scale
        envelope = (1.0 / torch.sqrt(scale)) * torch.exp(-0.5 * z.square())
        psi_r = envelope * torch.cos(self.omega0 * z)
        psi_i = envelope * torch.sin(self.omega0 * z)
        psi_r = psi_r.unsqueeze(1).expand(-1, self.in_channels, -1).contiguous()
        psi_i = psi_i.unsqueeze(1).expand(-1, self.in_channels, -1).contiguous()
        return psi_r, psi_i

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_r.shape != x_i.shape:
            raise ValueError(f"x_r and x_i shape mismatch: {x_r.shape} vs {x_i.shape}")
        psi_r, psi_i = self._build_wavelet_bank(x_r.device, x_r.dtype)
        pad = self.kernel_size // 2
        y_r = F.conv1d(x_r, psi_r, padding=pad) - F.conv1d(x_i, psi_i, padding=pad)
        y_i = F.conv1d(x_r, psi_i, padding=pad) + F.conv1d(x_i, psi_r, padding=pad)
        return y_r, y_i


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight_r = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_i = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias_r = nn.Parameter(torch.zeros(out_channels))
        self.bias_i = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_i, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.in_channels * self.kernel_size)
        nn.init.uniform_(self.bias_r, -bound, bound)
        nn.init.uniform_(self.bias_i, -bound, bound)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_r = F.conv1d(x_r, self.weight_r, self.bias_r, padding=self.padding) - F.conv1d(
            x_i, self.weight_i, None, padding=self.padding
        )
        y_i = F.conv1d(x_r, self.weight_i, self.bias_i, padding=self.padding) + F.conv1d(
            x_i, self.weight_r, None, padding=self.padding
        )
        return y_r, y_i


class ComplexReLU(nn.Module):
    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return F.relu(x_r), F.relu(x_i)


class ComplexMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            F.max_pool1d(x_r, self.kernel_size, self.stride),
            F.max_pool1d(x_i, self.kernel_size, self.stride),
        )


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_r = self.real(x_r) - self.imag(x_i)
        y_i = self.real(x_i) + self.imag(x_r)
        return y_r, y_i


class CVWavLeNet1D(nn.Module):
    def __init__(self, num_classes: int, wavelet_channels: int = 16, omega0: float = 5.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.wavelet = ComplexMorletWaveletLayer1d(1, wavelet_channels, kernel_size=65, omega0=omega0)
        self.act1 = ComplexReLU()
        self.pool1 = ComplexMaxPool1d(2, 2)
        self.conv2 = ComplexConv1d(wavelet_channels, 32, kernel_size=5, padding=2)
        self.act2 = ComplexReLU()
        self.pool2 = ComplexMaxPool1d(2, 2)
        self.conv3 = ComplexConv1d(32, 64, kernel_size=3, padding=1)
        self.act3 = ComplexReLU()
        self.pool3 = ComplexMaxPool1d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected [B,2,N], got {x.shape}")
        x_r = x[:, 0:1, :]
        x_i = x[:, 1:2, :]
        x_r, x_i = self.wavelet(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        x_r, x_i = self.pool1(x_r, x_i)
        x_r, x_i = self.conv2(x_r, x_i)
        x_r, x_i = self.act2(x_r, x_i)
        x_r, x_i = self.pool2(x_r, x_i)
        x_r, x_i = self.conv3(x_r, x_i)
        x_r, x_i = self.act3(x_r, x_i)
        x_r, x_i = self.pool3(x_r, x_i)
        x = torch.cat([x_r, x_i], dim=1)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


class CVMLeNet(nn.Module):
    """Stricter CV-Morlet LeNet-style detector for raw I/Q input."""

    def __init__(self, num_classes: int, input_len: int = 820, wavelet_channels: int = 16, omega0: float = 5.0) -> None:
        super().__init__()
        self.wavelet = ComplexMorletQuadratureLayer1d(1, wavelet_channels, kernel_size=65, omega0=omega0)
        self.act1 = ComplexReLU()
        self.pool1 = ComplexMaxPool1d(2, 2)
        self.conv2 = ComplexConv1d(wavelet_channels, 32, kernel_size=5, padding=2)
        self.act2 = ComplexReLU()
        self.pool2 = ComplexMaxPool1d(2, 2)
        self.conv3 = ComplexConv1d(32, 64, kernel_size=5, padding=2)
        self.act3 = ComplexReLU()
        self.pool3 = ComplexMaxPool1d(2, 2)

        pooled_len = input_len // 8
        flat_dim = 64 * pooled_len
        self.fc1 = ComplexLinear(flat_dim, 256)
        self.fc2 = ComplexLinear(256, 84)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(84 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected [B,2,N], got {x.shape}")
        x_r = x[:, 0:1, :]
        x_i = x[:, 1:2, :]
        x_r, x_i = self.wavelet(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        x_r, x_i = self.pool1(x_r, x_i)
        x_r, x_i = self.conv2(x_r, x_i)
        x_r, x_i = self.act2(x_r, x_i)
        x_r, x_i = self.pool2(x_r, x_i)
        x_r, x_i = self.conv3(x_r, x_i)
        x_r, x_i = self.act3(x_r, x_i)
        x_r, x_i = self.pool3(x_r, x_i)
        x_r = x_r.flatten(1)
        x_i = x_i.flatten(1)
        x_r, x_i = self.fc1(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        x_r = self.dropout(x_r)
        x_i = self.dropout(x_i)
        x_r, x_i = self.fc2(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        feat = torch.cat([x_r, x_i], dim=1)
        return self.head(feat)


class CVMLeNetMTL(nn.Module):
    """CVM-LeNet with an added synchronization heatmap head."""

    def __init__(
        self,
        num_classes: int,
        input_len: int = 820,
        loc_len: int = 493,
        wavelet_channels: int = 16,
        omega0: float = 5.0,
    ) -> None:
        super().__init__()
        self.loc_len = loc_len
        self.wavelet = ComplexMorletQuadratureLayer1d(1, wavelet_channels, kernel_size=65, omega0=omega0)
        self.act1 = ComplexReLU()
        self.pool1 = ComplexMaxPool1d(2, 2)
        self.conv2 = ComplexConv1d(wavelet_channels, 32, kernel_size=5, padding=2)
        self.act2 = ComplexReLU()
        self.pool2 = ComplexMaxPool1d(2, 2)
        self.conv3 = ComplexConv1d(32, 64, kernel_size=5, padding=2)
        self.act3 = ComplexReLU()
        self.pool3 = ComplexMaxPool1d(2, 2)

        pooled_len = input_len // 8
        flat_dim = 64 * pooled_len
        self.fc1 = ComplexLinear(flat_dim, 256)
        self.fc2 = ComplexLinear(256, 84)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(84 * 2, num_classes)
        self.loc_head = nn.Sequential(
            nn.Conv1d(64 * 2, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, return_loc: bool = False):
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected [B,2,N], got {x.shape}")
        x_r = x[:, 0:1, :]
        x_i = x[:, 1:2, :]
        x_r, x_i = self.wavelet(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        x_r, x_i = self.pool1(x_r, x_i)
        x_r, x_i = self.conv2(x_r, x_i)
        x_r, x_i = self.act2(x_r, x_i)
        x_r, x_i = self.pool2(x_r, x_i)
        x_r, x_i = self.conv3(x_r, x_i)
        x_r, x_i = self.act3(x_r, x_i)

        loc_feat = torch.cat([x_r, x_i], dim=1)
        loc_feat = F.interpolate(loc_feat, size=self.loc_len, mode="linear", align_corners=False)
        loc_logits = self.loc_head(loc_feat).squeeze(1)

        x_r, x_i = self.pool3(x_r, x_i)
        x_r = x_r.flatten(1)
        x_i = x_i.flatten(1)
        x_r, x_i = self.fc1(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        x_r = self.dropout(x_r)
        x_i = self.dropout(x_i)
        x_r, x_i = self.fc2(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        feat = torch.cat([x_r, x_i], dim=1)
        cls_logits = self.head(feat)
        if return_loc:
            return cls_logits, loc_logits
        return cls_logits


class RealIQCNN1D(nn.Module):
    """Small real-valued 1D CNN baseline for I/Q input."""

    def __init__(self, num_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, 9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, 7, padding=3),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 2, 5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


class CorrCNN1D(nn.Module):
    """Small A_corr baseline."""

    def __init__(self, num_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, 5, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


class _CorrContextBlock(nn.Module):
    def __init__(
        self,
        seq_len: int = 493,
        hidden: int = 32,
        lstm_hidden: int = 64,
        attn_num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=21, padding=10),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.feat_dim = lstm_hidden * 2
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.feat_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=attn_num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        self.norm1 = nn.LayerNorm(self.feat_dim)
        self.norm2 = nn.LayerNorm(self.feat_dim)

    def forward(self, x_corr: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x_corr).transpose(1, 2)
        feat, _ = self.bilstm(feat)
        feat = feat + self.pos_embed[:, : feat.size(1), :]
        attn_out = self.attn(feat, feat, feat, need_weights=False)[0]
        feat = self.norm1(feat + attn_out)
        feat = self.norm2(feat + self.ffn(feat))
        return feat


class CCNetClassifier(nn.Module):
    """CC-Net classifier adapter: correlation context backbone + pooled classifier."""

    def __init__(self, num_classes: int, seq_len: int = 493, hidden: int = 32) -> None:
        super().__init__()
        self.context = _CorrContextBlock(seq_len=seq_len, hidden=hidden)
        self.classifier = nn.Sequential(
            nn.Linear(self.context.feat_dim, self.context.feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.context.feat_dim // 2, num_classes),
        )

    def forward(self, x_corr: torch.Tensor) -> torch.Tensor:
        feat = self.context(x_corr)
        return self.classifier(feat.mean(dim=1))


class CCMTLClassifier(nn.Module):
    """CC-MTL-Net adapter with classification and synchronization heads."""

    def __init__(self, num_classes: int, seq_len: int = 493, hidden: int = 32) -> None:
        super().__init__()
        self.context = _CorrContextBlock(seq_len=seq_len, hidden=hidden)
        dim = self.context.feat_dim
        self.cls_branch = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.loc_branch = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.cls_head = nn.Linear(dim, num_classes)
        self.loc_head = nn.Linear(dim, 1)

    def forward(self, x_corr: torch.Tensor, return_loc: bool = False):
        feat = self.context(x_corr)
        loc_logits = self.loc_head(self.loc_branch(feat)).squeeze(-1)
        cls_logits = self.cls_head(self.cls_branch(feat.mean(dim=1)))
        if return_loc:
            return cls_logits, loc_logits
        return cls_logits


class CCMtlSNetClassifier(nn.Module):
    """CC-CV-MTL-SNet adapter matching the shared/private-branch idea of CCNETS-MTL."""

    def __init__(self, num_classes: int, seq_len: int = 493, hidden: int = 32) -> None:
        super().__init__()
        self.context = _CorrContextBlock(seq_len=seq_len, hidden=hidden)
        dim = self.context.feat_dim
        self.cls_private = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.loc_private = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.loc_to_cls = nn.Linear(dim, dim)
        self.cls_to_loc_gate = nn.Linear(dim, dim)
        self.cls_head = nn.Linear(dim, num_classes)
        self.loc_head = nn.Linear(dim, 1)

    def forward(self, x_corr: torch.Tensor, return_loc: bool = False):
        feat = self.context(x_corr)
        cls_feat = self.cls_private(feat)
        loc_feat = self.loc_private(feat)
        pooled_loc = loc_feat.mean(dim=1)
        pooled_cls = cls_feat.mean(dim=1) + self.loc_to_cls(pooled_loc)
        gate = torch.sigmoid(self.cls_to_loc_gate(pooled_cls)).unsqueeze(1)
        loc_feat = loc_feat * (1.0 + gate)
        cls_logits = self.cls_head(pooled_cls)
        loc_logits = self.loc_head(loc_feat).squeeze(-1)
        if return_loc:
            return cls_logits, loc_logits
        return cls_logits


class _WaveletFeatureAlign(nn.Module):
    def __init__(self, seq_len: int = 493, hidden: int = 32, wavelet_ch: int = 16) -> None:
        super().__init__()
        self.wavelet = ComplexMorletWaveletLayer1d(1, wavelet_ch, kernel_size=65, omega0=5.0)
        self.act1 = ComplexReLU()
        self.conv2 = ComplexConv1d(wavelet_ch, hidden, kernel_size=5, padding=2)
        self.act2 = ComplexReLU()
        self.conv3 = ComplexConv1d(hidden, hidden, kernel_size=3, padding=1)
        self.act3 = ComplexReLU()
        self.align = nn.AdaptiveAvgPool1d(seq_len)
        self.out_dim = hidden * 2

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x_r = iq[:, 0:1, :]
        x_i = iq[:, 1:2, :]
        x_r, x_i = self.wavelet(x_r, x_i)
        x_r, x_i = self.act1(x_r, x_i)
        x_r, x_i = self.conv2(x_r, x_i)
        x_r, x_i = self.act2(x_r, x_i)
        x_r, x_i = self.conv3(x_r, x_i)
        x_r, x_i = self.act3(x_r, x_i)
        feat = torch.cat([x_r, x_i], dim=1)
        return self.align(feat).transpose(1, 2)


class CCCVMTLClassifier(nn.Module):
    """CC-CV-MTL-Net adapter: corr context plus complex-wavelet I/Q branch."""

    def __init__(self, num_classes: int, seq_len: int = 493, hidden: int = 32) -> None:
        super().__init__()
        self.corr_context = _CorrContextBlock(seq_len=seq_len, hidden=hidden)
        self.wavelet_branch = _WaveletFeatureAlign(seq_len=seq_len, hidden=hidden)
        dim = self.corr_context.feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(dim + self.wavelet_branch.out_dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(dim, num_classes)
        self.loc_head = nn.Linear(dim, 1)

    def forward(self, x_corr: torch.Tensor, iq: torch.Tensor, return_loc: bool = False):
        corr_feat = self.corr_context(x_corr)
        wav_feat = self.wavelet_branch(iq)
        feat = self.fuse(torch.cat([corr_feat, wav_feat], dim=-1))
        cls_logits = self.cls_head(feat.mean(dim=1))
        loc_logits = self.loc_head(feat).squeeze(-1)
        if return_loc:
            return cls_logits, loc_logits
        return cls_logits


class CCCVGNetClassifier(nn.Module):
    """CC-CV-MTL-GNet adapter: gated injection of wavelet features into corr features."""

    def __init__(self, num_classes: int, seq_len: int = 493, hidden: int = 32) -> None:
        super().__init__()
        self.corr_context = _CorrContextBlock(seq_len=seq_len, hidden=hidden)
        self.wavelet_branch = _WaveletFeatureAlign(seq_len=seq_len, hidden=hidden)
        dim = self.corr_context.feat_dim
        self.wav_proj = nn.Linear(self.wavelet_branch.out_dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.cls_head = nn.Linear(dim, num_classes)
        self.loc_head = nn.Linear(dim, 1)

    def forward(self, x_corr: torch.Tensor, iq: torch.Tensor, return_loc: bool = False):
        corr_feat = self.corr_context(x_corr)
        wav_feat = self.wav_proj(self.wavelet_branch(iq))
        gate = self.gate(torch.cat([corr_feat, wav_feat], dim=-1))
        feat = self.norm(corr_feat + gate * wav_feat)
        cls_logits = self.cls_head(feat.mean(dim=1))
        loc_logits = self.loc_head(feat).squeeze(-1)
        if return_loc:
            return cls_logits, loc_logits
        return cls_logits


class _TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _MTDCTBranch(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, depth: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.shallow = nn.Sequential(
            *[
                _TCNResidualBlock(
                    hidden,
                    kernel_size=5,
                    dilation=2**idx,
                    dropout=dropout,
                )
                for idx in range(depth)
            ]
        )
        self.down = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            _TCNResidualBlock(hidden, kernel_size=5, dilation=1, dropout=dropout),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            _TCNResidualBlock(hidden * 2, kernel_size=5, dilation=1, dropout=dropout),
            _TCNResidualBlock(hidden * 2, kernel_size=5, dilation=2, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shallow = self.shallow(self.stem(x))
        deep = self.down(shallow)
        return shallow, deep


class MTDCTNet(nn.Module):
    """Multi-task dual-branch correlation-aware temporal network.

    The model keeps the localization axis fully convolutional. If I/Q and
    correlation inputs have different lengths, the I/Q features are aligned to
    the correlation axis so loc_logits stay in the correlation-domain index.
    """

    def __init__(
        self,
        num_classes: int,
        use_iq: bool = True,
        use_corr: bool = True,
        corr_channels: int = 1,
        hidden: int = 64,
        tcn_depth: int = 4,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not use_iq and not use_corr:
            raise ValueError("At least one branch must be enabled.")
        self.use_iq = use_iq
        self.use_corr = use_corr

        if use_iq:
            self.iq_branch = _MTDCTBranch(2, hidden=hidden, depth=tcn_depth, dropout=dropout)
        if use_corr:
            self.corr_branch = _MTDCTBranch(corr_channels, hidden=hidden, depth=tcn_depth, dropout=dropout)

        n_branches = int(use_iq) + int(use_corr)
        self.shallow_fuse = nn.Sequential(
            nn.Conv1d(hidden * n_branches, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.deep_fuse = nn.Sequential(
            nn.Conv1d(hidden * 2 * n_branches, hidden * 2, kernel_size=1),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden * 2,
            nhead=transformer_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden * 3, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            _TCNResidualBlock(hidden, kernel_size=5, dilation=1, dropout=dropout),
            nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
        )
        self.loc_head = nn.Conv1d(hidden // 2, 1, kernel_size=1)

    @staticmethod
    def _sinusoidal_positional_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        even_dim = (dim + 1) // 2
        div = torch.exp(
            torch.arange(even_dim, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(even_dim - 1, 1))
        )
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div[: pe[:, 0::2].size(1)])
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        return pe.unsqueeze(0)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == ref.size(-1):
            return x
        return F.interpolate(x, size=ref.size(-1), mode="linear", align_corners=False)

    def _encode(self, x_corr: torch.Tensor | None, iq: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        shallow_feats: list[torch.Tensor] = []
        deep_feats: list[torch.Tensor] = []
        ref_shallow: torch.Tensor | None = None
        ref_deep: torch.Tensor | None = None

        if self.use_corr:
            if x_corr is None:
                raise ValueError("x_corr is required when use_corr=True")
            corr_shallow, corr_deep = self.corr_branch(x_corr)
            ref_shallow, ref_deep = corr_shallow, corr_deep
            shallow_feats.append(corr_shallow)
            deep_feats.append(corr_deep)

        if self.use_iq:
            if iq is None:
                raise ValueError("iq is required when use_iq=True")
            iq_shallow, iq_deep = self.iq_branch(iq)
            if ref_shallow is None:
                ref_shallow, ref_deep = iq_shallow, iq_deep
            else:
                iq_shallow = self._resize_like(iq_shallow, ref_shallow)
                iq_deep = self._resize_like(iq_deep, ref_deep)
            shallow_feats.append(iq_shallow)
            deep_feats.append(iq_deep)

        shallow = self.shallow_fuse(torch.cat(shallow_feats, dim=1))
        deep = self.deep_fuse(torch.cat(deep_feats, dim=1))
        return shallow, deep

    def forward(self, x_corr: torch.Tensor | None = None, iq: torch.Tensor | None = None, return_loc: bool = False):
        shallow, deep = self._encode(x_corr=x_corr, iq=iq)
        tokens = deep.transpose(1, 2)
        tokens = tokens + self._sinusoidal_positional_encoding(
            length=tokens.size(1),
            dim=tokens.size(2),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        context = self.context(tokens).transpose(1, 2)
        pooled = context.mean(dim=-1)
        cls_logits = self.cls_head(pooled)

        if not return_loc:
            return cls_logits

        context_up = F.interpolate(context, size=shallow.size(-1), mode="linear", align_corners=False)
        loc_feat = self.decoder(torch.cat([context_up, shallow], dim=1))
        loc_logits = self.loc_head(loc_feat).squeeze(1)
        return cls_logits, loc_logits


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name in {"CVWavLeNet1D", "CV-NET"}:
        return CVWavLeNet1D(num_classes=num_classes)
    if model_name == "CVM-LeNet":
        return CVMLeNet(num_classes=num_classes)
    if model_name == "CV-MTL-Net":
        return CVMLeNetMTL(num_classes=num_classes)
    if model_name == "RealIQCNN1D":
        return RealIQCNN1D(num_classes=num_classes)
    if model_name == "CorrCNN1D":
        return CorrCNN1D(num_classes=num_classes)
    if model_name == "CC-Net":
        return CCNetClassifier(num_classes=num_classes)
    if model_name == "CC-MTL-Net":
        return CCMTLClassifier(num_classes=num_classes)
    if model_name == "CC-CV-MTL-Net":
        return CCCVMTLClassifier(num_classes=num_classes)
    if model_name == "CC-CV-MTL-SNet":
        return CCMtlSNetClassifier(num_classes=num_classes)
    if model_name == "CC-CV-MTL-GNet":
        return CCCVGNetClassifier(num_classes=num_classes)
    if model_name == "MT-DCTNet-IQ":
        return MTDCTNet(num_classes=num_classes, use_iq=True, use_corr=False)
    if model_name == "MT-DCTNet-Corr":
        return MTDCTNet(num_classes=num_classes, use_iq=False, use_corr=True)
    if model_name in {"MT-DCTNet-Dual", "MT-DCTNet"}:
        return MTDCTNet(num_classes=num_classes, use_iq=True, use_corr=True)
    raise ValueError(f"Unknown model_name: {model_name}")


def model_feature_keys(model_name: str) -> tuple[str, ...]:
    if model_name in {"CVWavLeNet1D", "CV-NET", "CVM-LeNet", "CV-MTL-Net", "RealIQCNN1D"}:
        return ("I_obs", "Q_obs")
    if model_name == "MT-DCTNet-IQ":
        return ("I_obs", "Q_obs")
    if model_name in {"CorrCNN1D", "CC-Net", "CC-MTL-Net", "CC-CV-MTL-SNet"}:
        return ("A_corr",)
    if model_name == "MT-DCTNet-Corr":
        return ("A_corr",)
    if model_name in {"CC-CV-MTL-Net", "CC-CV-MTL-GNet"}:
        return ("A_corr", "I_obs", "Q_obs")
    if model_name in {"MT-DCTNet-Dual", "MT-DCTNet"}:
        return ("A_corr", "I_obs", "Q_obs")
    raise ValueError(f"Unknown model_name: {model_name}")
