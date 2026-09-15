"""前向光谱与实验 IR 的无泄漏候选级特征。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _assert_spectra(predicted: torch.Tensor, target: torch.Tensor) -> None:
    """检查预测谱和目标谱均为 [N,L]，且长度一致。"""
    if predicted.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"光谱必须为二维 [N,L]，实际为 {tuple(predicted.shape)} 与 {tuple(target.shape)}"
        )
    if predicted.shape != target.shape:
        raise ValueError(
            f"预测谱与目标谱形状不一致: {tuple(predicted.shape)} != {tuple(target.shape)}"
        )
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("光谱包含 NaN 或 Inf")


def _positive_mass(spectrum: torch.Tensor) -> torch.Tensor:
    """将吸收谱转换为非负质量分布；输入输出形状均为 [N,L]。"""
    mass = spectrum.float().clamp_min(0.0)
    return mass / mass.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def _derivative(spectrum: torch.Tensor) -> torch.Tensor:
    """计算一阶差分；[N,L] -> [N,L]，首点用零填充。"""
    diff = spectrum[:, 1:] - spectrum[:, :-1]
    return F.pad(diff, (1, 0))


def _soft_peak_profile(spectrum: torch.Tensor, kernel_size: int = 9):
    """提取平滑背景上的软峰质量，输入输出形状为 [N,L]。"""
    # 该函数不依赖离散 find_peaks，适合大候选池批量计算。
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size 必须为大于等于 3 的奇数")
    local = F.avg_pool1d(
        spectrum.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).squeeze(1)
    return (spectrum - local).clamp_min(0.0)


def compare_forward_spectra(
    predicted: torch.Tensor,
    target: torch.Tensor,
    region_count: int = 3,
) -> torch.Tensor:
    """计算前向预测谱与实验谱的多指标特征。

    返回顺序固定为：
    cosine、derivative_cosine、emd_score、soft_peak_cosine、region_cosine、
    pearson、predicted_mass_entropy、target_mass_entropy。
    返回形状为 [N, 8]，不会使用真值结构或候选输入位置。
    """
    _assert_spectra(predicted, target)
    if region_count < 1:
        raise ValueError("region_count 必须大于 0")

    # predicted/target: [N,L] -> l2-normalized [N,L]
    pred_centered = predicted - predicted.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    cosine = F.cosine_similarity(predicted, target, dim=1)
    derivative_cosine = F.cosine_similarity(
        _derivative(predicted), _derivative(target), dim=1
    )
    pearson = F.cosine_similarity(pred_centered, target_centered, dim=1)

    # 非负质量的 CDF 差近似一维 Sinkhorn/EMD；值越大表示越相似。
    pred_mass = _positive_mass(predicted)
    target_mass = _positive_mass(target)
    pred_cdf = torch.cumsum(pred_mass, dim=1)
    target_cdf = torch.cumsum(target_mass, dim=1)
    emd = torch.mean(torch.abs(pred_cdf - target_cdf), dim=1)
    emd_score = (1.0 - 2.0 * emd).clamp(-1.0, 1.0)

    pred_peak = _soft_peak_profile(pred_mass)
    target_peak = _soft_peak_profile(target_mass)
    soft_peak_cosine = F.cosine_similarity(pred_peak, target_peak, dim=1)

    region_scores = []
    length = predicted.shape[1]
    for index in range(region_count):
        start = (index * length) // region_count
        end = ((index + 1) * length) // region_count
        if end <= start:
            continue
        region_scores.append(
            F.cosine_similarity(predicted[:, start:end], target[:, start:end], dim=1)
        )
    region_cosine = torch.stack(region_scores, dim=1).mean(dim=1)

    entropy_pred = -(pred_mass * pred_mass.clamp_min(1.0e-8).log()).sum(dim=1)
    entropy_target = -(target_mass * target_mass.clamp_min(1.0e-8).log()).sum(dim=1)
    return torch.stack(
        [
            cosine,
            derivative_cosine,
            emd_score,
            soft_peak_cosine,
            region_cosine,
            pearson,
            entropy_pred,
            entropy_target,
        ],
        dim=1,
    )


def fixed_forward_score(features: torch.Tensor) -> torch.Tensor:
    """计算不读取标签的固定前向基线分数，输入为 [N,8]。"""
    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError(f"features 必须为 [N,8]，实际为 {tuple(features.shape)}")
    # 熵差是辅助稳定项，不让其压过峰位和整体谱形匹配。
    entropy_agreement = 1.0 - torch.tanh(
        torch.abs(features[:, 6] - features[:, 7]) / 8.0
    )
    return (
        0.30 * features[:, 0]
        + 0.18 * features[:, 1]
        + 0.18 * features[:, 2]
        + 0.20 * features[:, 3]
        + 0.10 * features[:, 4]
        + 0.04 * features[:, 5]
        + 0.05 * entropy_agreement
    )


__all__ = ["compare_forward_spectra", "fixed_forward_score"]
