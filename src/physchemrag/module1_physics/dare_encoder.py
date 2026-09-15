"""DARE 推理与域对齐所需的最小光谱编码器定义。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """前向恒等、反向乘以负域权重。"""

    @staticmethod
    def forward(ctx, features: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert features.ndim == 2, f"特征应为 [B,D]，实际为 {tuple(features.shape)}"
        return GradientReversalFunction.apply(features, self.alpha)


class DARESpectralEncoder(nn.Module):
    """与历史 checkpoint 严格同构的 1D CNN 光谱编码器。"""

    def __init__(self, feature_dim: int = 512, num_classes: int = 15) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.InstanceNorm1d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(128, affine=True),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8),
            nn.Flatten(),
            nn.Linear(128 * 8, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.task_predictor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(
        self, spectra: torch.Tensor, alpha: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert spectra.ndim == 3 and spectra.shape[1:] == (1, 1800), (
            f"光谱应为 [B,1,1800]，实际为 {tuple(spectra.shape)}"
        )
        # spectra: [B,1,1800] -> features: [B,512]
        features = self.feature_extractor(spectra)
        task_logits = self.task_predictor(features)
        self.grl.alpha = float(alpha)
        domain_logits = self.domain_discriminator(self.grl(features))
        return task_logits, features, domain_logits


# 保留历史类名，确保已有 checkpoint 和训练代码无需迁移。
DARE_SpectralEncoder = DARESpectralEncoder

