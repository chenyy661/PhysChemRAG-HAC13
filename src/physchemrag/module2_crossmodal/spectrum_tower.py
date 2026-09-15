"""仅保留 HAC13 流程使用的 DARE 光谱塔。"""

from __future__ import annotations

import torch
import torch.nn as nn

from physchemrag.module1_physics.dare_encoder import DARESpectralEncoder


class DARESpectrumTower(nn.Module):
    def __init__(self, pretrained_encoder_path: str, final_dim: int = 512) -> None:
        super().__init__()
        self.encoder = DARESpectralEncoder(feature_dim=final_dim, num_classes=15)
        checkpoint = torch.load(
            pretrained_encoder_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "encoder_state_dict", "model_state_dict"):
                if isinstance(checkpoint.get(key), dict):
                    state_dict = checkpoint[key]
                    break
        try:
            self.encoder.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"DARE 光谱塔无法加载权重: {pretrained_encoder_path}"
            ) from exc
        print(f"DARESpectrumTower [OK] {pretrained_encoder_path}")
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.projection_head = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, final_dim),
        )

    def forward(
        self, spectra: torch.Tensor, batch_anchors: object | None = None
    ) -> torch.Tensor:
        assert spectra.ndim == 3 and spectra.shape[1:] == (1, 1800), (
            f"光谱应为 [B,1,1800]，实际为 {tuple(spectra.shape)}"
        )
        with torch.no_grad():
            _, features, _ = self.encoder(spectra)
        # features: [B,512] -> embeddings: [B,512]
        return self.projection_head(features)

