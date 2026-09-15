"""构造红外光谱多视图和峰 token。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    centered = vector - vector.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return centered / scale


def _coerce_valid_mask(spectrum: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    """将 valid_mask 规范为 [B,L]；缺失时视为所有波数均有效。"""
    if valid_mask is None:
        return torch.ones(
            (spectrum.shape[0], spectrum.shape[-1]),
            dtype=torch.bool,
            device=spectrum.device,
        )
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=spectrum.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 2 or mask.shape != (spectrum.shape[0], spectrum.shape[-1]):
        raise ValueError(
            "valid_mask 应为 [L]、[B,L] 或 [B,1,L]，"
            f"实际为 {tuple(mask.shape)}，光谱为 {tuple(spectrum.shape)}"
        )
    if not bool(mask.any(dim=1).all()):
        raise ValueError("每条光谱至少需要一个有效波数点")
    return mask


def prepare_masked_spectrum(
    spectrum: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用相邻有效点线性填补未测量位置，避免零填充制造伪峰。"""
    if spectrum.ndim == 1:
        values = spectrum.view(1, 1, -1)
    elif spectrum.ndim == 2:
        values = spectrum.unsqueeze(1)
    elif spectrum.ndim == 3 and spectrum.shape[1] == 1:
        values = spectrum
    else:
        raise ValueError(f"spectrum 应为 [L]、[B,L] 或 [B,1,L]，实际为 {tuple(spectrum.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError("光谱包含非有限值")
    values = values.float()
    mask = _coerce_valid_mask(values, valid_mask)
    filled = values.clone()
    # 仅在存在缺测点时执行逐条插值；L=1800，开销相对于编码器可忽略。
    for batch_index in range(values.shape[0]):
        valid_positions = torch.where(mask[batch_index])[0]
        if valid_positions.numel() == values.shape[-1]:
            continue
        valid_values = values[batch_index, 0, valid_positions]
        first = int(valid_positions[0])
        last = int(valid_positions[-1])
        filled[batch_index, 0, :first] = valid_values[0]
        filled[batch_index, 0, last + 1 :] = valid_values[-1]
        if last > first + 1:
            gap_positions = torch.arange(first + 1, last, device=values.device)
            gap_mask = ~mask[batch_index, gap_positions]
            if bool(gap_mask.any()):
                gap_positions = gap_positions[gap_mask]
                right = torch.searchsorted(valid_positions, gap_positions).clamp(
                    min=1, max=valid_positions.numel() - 1
                )
                left = right - 1
                left_pos = valid_positions[left].float()
                right_pos = valid_positions[right].float()
                alpha = (gap_positions.float() - left_pos) / (right_pos - left_pos).clamp_min(1.0)
                interpolated = valid_values[left] * (1.0 - alpha) + valid_values[right] * alpha
                filled[batch_index, 0, gap_positions] = interpolated
    return filled, mask


def build_spectral_views(
    spectrum: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 [B,5,L]：原始、基线校正、一阶导数、二阶导数和峰掩码。"""
    if spectrum.ndim == 1:
        spectrum = spectrum.view(1, 1, -1)
    elif spectrum.ndim == 2:
        spectrum = spectrum.unsqueeze(1)
    if spectrum.ndim != 3 or spectrum.shape[1] != 1:
        raise ValueError(f"spectrum 应为 [B,1,L]、[B,L] 或 [L]，实际为 {tuple(spectrum.shape)}")
    if not torch.isfinite(spectrum).all():
        raise ValueError("光谱包含非有限值")
    raw, mask = prepare_masked_spectrum(spectrum, valid_mask)
    length = raw.shape[-1]
    window = max(9, min(101, length // 8 * 2 + 1))
    baseline = F.avg_pool1d(raw, kernel_size=window, stride=1, padding=window // 2)
    baseline = baseline[..., :length]
    corrected = raw - baseline
    first = F.pad(corrected[..., 1:] - corrected[..., :-1], (0, 1))
    second = F.pad(first[..., 1:] - first[..., :-1], (0, 1))
    smooth = F.avg_pool1d(corrected, kernel_size=5, stride=1, padding=2)
    # 使用局部最大池化构造无监督峰掩码；峰阈值按每条谱的 robust scale 自适应。
    peak_window = max(5, min(31, length // 40 * 2 + 1))
    local_max = F.max_pool1d(smooth, kernel_size=peak_window, stride=1, padding=peak_window // 2)
    scale = corrected.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    peaks = ((smooth >= local_max - 1.0e-6) & (smooth > 1.5 * scale)).float()
    peaks = peaks * mask.unsqueeze(1).to(dtype=peaks.dtype)
    views = torch.cat([_normalize(raw), _normalize(corrected), _normalize(first), _normalize(second), peaks], dim=1)
    if views.ndim != 3 or views.shape[1] != 5 or views.shape[-1] != length:
        raise ValueError(f"光谱视图形状异常: {tuple(views.shape)}")
    return views


def extract_peak_tokens(
    spectrum_views: torch.Tensor,
    max_peaks: int = 64,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 [B,max_peaks,4]：归一化位置、高度、宽度和局部面积。"""
    if spectrum_views.ndim != 3 or spectrum_views.shape[1] < 2:
        raise ValueError(f"spectrum_views 应为 [B,V,L]，实际为 {tuple(spectrum_views.shape)}")
    corrected = spectrum_views[:, 1]
    mask = spectrum_views[:, 4] if spectrum_views.shape[1] > 4 else torch.zeros_like(corrected)
    batch_size, length = corrected.shape
    valid = _coerce_valid_mask(corrected.unsqueeze(1), valid_mask)
    tokens = corrected.new_zeros((batch_size, max_peaks, 4))
    for batch_index in range(batch_size):
        positions = torch.where((mask[batch_index] > 0.5) & valid[batch_index])[0]
        if positions.numel() == 0:
            valid_positions = torch.where(valid[batch_index])[0]
            values = corrected[batch_index, valid_positions].abs()
            positions = valid_positions[torch.topk(values, min(max_peaks, values.numel())).indices]
        scores = corrected[batch_index, positions].abs()
        order = torch.argsort(scores, descending=True)[:max_peaks]
        positions = positions[order]
        values = corrected[batch_index, positions]
        count = positions.numel()
        tokens[batch_index, :count, 0] = positions.float() / max(length - 1, 1)
        tokens[batch_index, :count, 1] = values
        tokens[batch_index, :count, 2] = 1.0 / max(length, 1)
        tokens[batch_index, :count, 3] = values.abs()
    if not torch.isfinite(tokens).all():
        raise ValueError("峰 token 包含非有限值")
    return tokens


__all__ = ["prepare_masked_spectrum", "build_spectral_views", "extract_peak_tokens"]
