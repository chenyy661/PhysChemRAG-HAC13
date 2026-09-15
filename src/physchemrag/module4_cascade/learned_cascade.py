"""无 PubChem 返回顺序依赖的光谱-结构级联模型。

本模块提供两部分：
1. DARE 光谱塔与分子图塔组成的对比学习双塔；
2. 对候选集合进行置换不变建模的 Set Transformer 精排器。

PubChem 只负责候选生成；任何 API 返回名次都不会进入模型输入或最终分数。
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from physchemrag.shared.chemical_domain import FORMULA_ELEMENTS

from physchemrag.module2_crossmodal.graph_tower import GraphTower
from physchemrag.module2_crossmodal.spectrum_tower import DARESpectrumTower


class SpectrumGraphContrastiveModel(nn.Module):
    """DARE 光谱塔与 GINE 分子图塔的双塔对齐模型。"""

    def __init__(
        self,
        dare_weights: str,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,
        embed_dim: int = 512,
        graph_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if embed_dim < 32 or node_in_dim < 1 or edge_in_dim < 1:
            raise ValueError("对比学习模型维度不合法")
        self.embed_dim = int(embed_dim)
        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.dare_weights = str(dare_weights)
        self.spectrum_tower = DARESpectrumTower(
            pretrained_encoder_path=self.dare_weights,
            final_dim=self.embed_dim,
        )
        self.graph_tower = GraphTower(
            node_in_dim=self.node_in_dim,
            edge_in_dim=self.edge_in_dim,
            hidden_dim=graph_hidden_dim,
            final_dim=self.embed_dim,
        )
        # logit_scale 以 log 参数化，避免温度在训练中变成非正数。
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_spectrum(self, spectra: torch.Tensor) -> torch.Tensor:
        """编码光谱；输入 [B,1,1800]，输出归一化向量 [B,D]。"""
        if spectra.ndim != 3 or tuple(spectra.shape[1:]) != (1, 1800):
            raise ValueError(f"spectra 应为 [B,1,1800]，实际为 {tuple(spectra.shape)}")
        return F.normalize(self.spectrum_tower(spectra), p=2, dim=-1)

    def encode_graph(self, graph_batch) -> torch.Tensor:
        """编码 PyG Batch 图，输出归一化向量 [B,D]。"""
        if not hasattr(graph_batch, "batch"):
            raise ValueError("graph_batch 缺少 batch 属性")
        output = self.graph_tower(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )
        return F.normalize(output, p=2, dim=-1)

    def forward(self, graph_batch, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回光谱和分子图嵌入，二者形状均为 [B,D]。"""
        z_spec = self.encode_spectrum(spectra)
        z_mol = self.encode_graph(graph_batch)
        if z_spec.shape != z_mol.shape:
            raise ValueError(f"双塔批次形状不一致: {tuple(z_spec.shape)} != {tuple(z_mol.shape)}")
        return z_spec, z_mol


def symmetric_infonce_loss(
    z_spec: torch.Tensor,
    z_mol: torch.Tensor,
    logit_scale: torch.Tensor,
    formula_ids: Iterable[str] | None = None,
    structure_ids: Iterable[str] | None = None,
    hard_negative_weight: float = 0.20,
    hard_negative_margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """对称 InfoNCE，并对同分子式负样本增加轻量硬负样本约束。"""
    if z_spec.ndim != 2 or z_mol.ndim != 2 or z_spec.shape != z_mol.shape:
        raise ValueError("z_spec 与 z_mol 必须为同形状 [B,D] 张量")
    batch_size = z_spec.shape[0]
    if batch_size < 2:
        raise ValueError("对比学习批次至少需要两个样本")
    if not torch.isfinite(z_spec).all() or not torch.isfinite(z_mol).all():
        raise FloatingPointError("对比学习嵌入包含非有限值")

    scale = logit_scale.exp().clamp(1.0, 100.0)
    # z_spec/z_mol: [B,D] -> FP32 logits: [B,B]，避免 AMP 下大矩阵乘法的数值溢出。
    logits = scale.float() * (z_spec.float() @ z_mol.float().transpose(0, 1))
    labels = torch.arange(batch_size, device=logits.device)
    positive_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    if structure_ids is not None:
        structures = [str(value) for value in structure_ids]
        if len(structures) != batch_size:
            raise ValueError("structure_ids 数量与对比学习批次不一致")
        # positive_mask: [B,B]；相同非立体连接结构均为正样本，避免假负样本。
        positive_mask = torch.tensor(
            [[structures[i] == structures[j] for j in range(batch_size)] for i in range(batch_size)],
            dtype=torch.bool,
            device=logits.device,
        )
    if structure_ids is None:
        loss_spec = F.cross_entropy(logits, labels)
        loss_mol = F.cross_entropy(logits.transpose(0, 1), labels)
    else:
        negative_infinity = torch.finfo(logits.dtype).min
        positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
        # logits/positive_mask: [B,B]；分别沿候选图轴和查询谱轴聚合多正样本概率质量。
        loss_spec = -(
            torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
        ).mean()
        loss_mol = -(
            torch.logsumexp(positive_logits, dim=0) - torch.logsumexp(logits, dim=0)
        ).mean()
    loss = 0.5 * (loss_spec + loss_mol)

    hard_loss = logits.new_zeros(())
    hard_count = 0
    if formula_ids is not None:
        formulas = [str(value) for value in formula_ids]
        if len(formulas) != batch_size:
            raise ValueError("formula_ids 数量与对比学习批次不一致")
        same_formula = torch.tensor(
            [[formulas[i] == formulas[j] for j in range(batch_size)] for i in range(batch_size)],
            dtype=torch.bool,
            device=logits.device,
        )
        # 同分子式且连接结构不同才是困难负样本。
        same_formula &= ~positive_mask
        if same_formula.any():
            # 每个查询可有多个同结构正样本；用正样本集合的最大 logit 作为 margin 参照。
            positive = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min).amax(dim=1, keepdim=True)
            positive = positive.expand_as(logits)
            hard_margin = F.relu(hard_negative_margin - positive + logits)
            hard_loss = hard_margin[same_formula].mean()
            hard_count = int(same_formula.sum().item())
            loss = loss + float(hard_negative_weight) * hard_loss
    return loss, {
        "infonce": float((0.5 * (loss_spec + loss_mol)).detach().item()),
        "hard_negative": float(hard_loss.detach().item()),
        "hard_count": float(hard_count),
        "positive_pair_count": float(positive_mask.sum().item()),
        "temperature": float((1.0 / scale).detach().item()),
    }


class SetTransformerListwiseRanker(nn.Module):
    """集合感知的候选精排器，不读取候选输入顺序。"""

    def __init__(
        self,
        embedding_dim: int = 512,
        candidate_feature_dim: int = 14,
        d_model: int = 256,
        set_stats_dim: int = 5,
        max_gate: float = 0.45,
        residual_scale: float = 0.80,
    ) -> None:
        super().__init__()
        if min(embedding_dim, candidate_feature_dim, d_model, set_stats_dim) < 1:
            raise ValueError("集合精排器维度必须为正")
        self.embedding_dim = int(embedding_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.d_model = int(d_model)
        self.set_stats_dim = int(set_stats_dim)
        self.max_gate = float(max_gate)
        self.residual_scale = float(residual_scale)

        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.candidate_feature_dim),
            nn.Linear(self.embedding_dim + self.candidate_feature_dim, self.d_model),
            nn.GELU(),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=4 * self.d_model,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.cross_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=8,
            dropout=0.10,
            batch_first=True,
        )
        interaction_dim = 4 * self.d_model
        self.residual_head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(self.d_model, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.set_stats_dim),
            nn.Linear(self.embedding_dim + self.set_stats_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # 初始阶段退化为双塔粗检索分数，防止精排器随机破坏粗检索结果。
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        candidate_features: torch.Tensor,
        query_embedding: torch.Tensor,
        set_stats: torch.Tensor,
        coarse_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入候选 [N,D]/[N,F]、查询 [D]，返回分数 [N]、门控和残差。"""
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate_embeddings 应为 [N,D]")
        if candidate_features.ndim != 2 or candidate_features.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("candidate_features 与候选数不一致")
        if candidate_embeddings.shape[1] != self.embedding_dim:
            raise ValueError("candidate_embeddings 维度与 checkpoint 不一致")
        if candidate_features.shape[1] != self.candidate_feature_dim:
            raise ValueError("candidate_features 维度与 checkpoint 不一致")
        if query_embedding.ndim != 1 or query_embedding.shape[0] != self.embedding_dim:
            raise ValueError("query_embedding 应为 [D]")
        if set_stats.ndim != 1 or set_stats.shape[0] != self.set_stats_dim:
            raise ValueError("set_stats 维度异常")
        if coarse_scores.ndim != 1 or coarse_scores.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("coarse_scores 应为 [N]")

        # 候选: [N,D]+[N,F] -> [1,N,d_model]，集合编码不依赖输入顺序。
        candidate_tokens = self.candidate_proj(
            torch.cat([candidate_embeddings.float(), candidate_features.float()], dim=-1)
        ).unsqueeze(0)
        candidate_context = self.set_encoder(candidate_tokens).squeeze(0)
        query_token = self.query_proj(query_embedding.float()).view(1, 1, self.d_model)
        query_tokens = query_token.expand(1, 1, self.d_model)
        attended, _ = self.cross_attention(
            query_tokens,
            candidate_context.unsqueeze(0),
            candidate_context.unsqueeze(0),
            need_weights=False,
        )
        query_context = attended.squeeze(0).expand_as(candidate_context)
        interaction = torch.cat(
            [candidate_context, query_context, torch.abs(candidate_context - query_context), candidate_context * query_context],
            dim=-1,
        )
        residual = torch.tanh(self.residual_head(interaction).squeeze(-1))
        gate_input = torch.cat([query_embedding.float(), set_stats.float()], dim=0).view(1, -1)
        gate = self.max_gate * torch.sigmoid(self.gate_head(gate_input).squeeze(-1)).squeeze(0)
        scores = coarse_scores.float() + self.residual_scale * gate * residual
        return scores, gate, residual


def listwise_set_loss(
    scores: torch.Tensor,
    positive_index: int,
    coarse_scores: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    hard_negative_k: int = 8,
    temperature: float = 0.20,
    margin: float = 0.05,
    residual_penalty: float = 0.01,
    gate_penalty: float = 0.002,
) -> torch.Tensor:
    """候选集合 listwise 损失，硬负样本由双塔粗分数确定。"""
    if scores.ndim != 1 or scores.numel() < 2:
        raise ValueError("scores 应为至少含两个候选的一维张量")
    if not 0 <= int(positive_index) < scores.numel():
        raise ValueError("positive_index 超出候选范围")
    target = torch.tensor([int(positive_index)], dtype=torch.long, device=scores.device)
    listwise = F.cross_entropy((scores / temperature).unsqueeze(0), target)
    positive = scores[int(positive_index)]
    negative_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    negative_mask[int(positive_index)] = False
    hard_candidates = torch.where(negative_mask)[0]
    if hard_candidates.numel():
        k = min(int(hard_negative_k), int(hard_candidates.numel()))
        hard_order = torch.argsort(coarse_scores[hard_candidates], descending=True)[:k]
        hard_idx = hard_candidates[hard_order]
        hard_margin = F.relu(float(margin) - positive + scores[hard_idx]).mean()
    else:
        hard_margin = scores.new_zeros(())
    return (
        listwise
        + hard_margin
        + residual_penalty * residual.square().mean()
        + gate_penalty * gate.square()
    )


class FormulaConditionedSetTransformerRanker(nn.Module):
    """将已知分子式作为查询条件的无顺序集合精排器。

    分子式只通过元素计数向量调节查询表示和门控，不使用 PubChem CID
    返回顺序、候选原始位置或 teacher 分数。候选集合编码不含位置编码，
    因而对候选置换保持不变。
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        candidate_feature_dim: int = 6,
        formula_dim: int = len(FORMULA_ELEMENTS),
        d_model: int = 256,
        set_stats_dim: int = 5,
        max_gate: float = 0.45,
        residual_scale: float = 0.80,
    ) -> None:
        super().__init__()
        if min(embedding_dim, candidate_feature_dim, formula_dim, d_model, set_stats_dim) < 1:
            raise ValueError("分子式集合精排器维度必须为正")
        self.embedding_dim = int(embedding_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.formula_dim = int(formula_dim)
        self.d_model = int(d_model)
        self.set_stats_dim = int(set_stats_dim)
        self.max_gate = float(max_gate)
        self.residual_scale = float(residual_scale)
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.candidate_feature_dim),
            nn.Linear(self.embedding_dim + self.candidate_feature_dim, self.d_model),
            nn.GELU(),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.formula_dim),
            nn.Linear(self.embedding_dim + self.formula_dim, self.d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=4 * self.d_model,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.cross_attention = nn.MultiheadAttention(
            self.d_model,
            num_heads=8,
            dropout=0.10,
            batch_first=True,
        )
        interaction_dim = 4 * self.d_model
        self.residual_head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(self.d_model, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.formula_dim + self.set_stats_dim),
            nn.Linear(self.embedding_dim + self.formula_dim + self.set_stats_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # 零初始化残差使模型起点等价于双塔粗排，避免随机精排破坏召回结果。
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        candidate_features: torch.Tensor,
        query_embedding: torch.Tensor,
        formula_features: torch.Tensor,
        set_stats: torch.Tensor,
        coarse_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入候选 [N,D]/[N,F] 与查询 [D]/[F]，返回分数 [N]。"""
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate_embeddings 应为 [N,D]")
        if candidate_features.ndim != 2 or candidate_features.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("candidate_features 与候选数不一致")
        if candidate_embeddings.shape[1] != self.embedding_dim:
            raise ValueError("candidate_embeddings 维度与模型配置不一致")
        if candidate_features.shape[1] != self.candidate_feature_dim:
            raise ValueError("candidate_features 维度与模型配置不一致")
        if query_embedding.ndim != 1 or query_embedding.shape[0] != self.embedding_dim:
            raise ValueError("query_embedding 应为 [D]")
        if formula_features.ndim != 1 or formula_features.shape[0] != self.formula_dim:
            raise ValueError("formula_features 应为 [F]")
        if set_stats.ndim != 1 or set_stats.shape[0] != self.set_stats_dim:
            raise ValueError("set_stats 维度异常")
        if coarse_scores.ndim != 1 or coarse_scores.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("coarse_scores 应为 [N]")

        # 候选: [N,D]+[N,F] -> [1,N,d_model]，没有位置编码。
        candidate_tokens = self.candidate_proj(
            torch.cat([candidate_embeddings.float(), candidate_features.float()], dim=-1)
        ).unsqueeze(0)
        candidate_context = self.set_encoder(candidate_tokens).squeeze(0)
        query_input = torch.cat([query_embedding.float(), formula_features.float()], dim=0)
        query_token = self.query_proj(query_input).view(1, 1, self.d_model)
        attended, _ = self.cross_attention(
            query_token,
            candidate_context.unsqueeze(0),
            candidate_context.unsqueeze(0),
            need_weights=False,
        )
        query_context = attended.squeeze(0).expand_as(candidate_context)
        interaction = torch.cat(
            [candidate_context, query_context, torch.abs(candidate_context - query_context), candidate_context * query_context],
            dim=-1,
        )
        residual = torch.tanh(self.residual_head(interaction).squeeze(-1))
        gate_input = torch.cat([query_embedding.float(), formula_features.float(), set_stats.float()], dim=0).view(1, -1)
        gate = self.max_gate * torch.sigmoid(self.gate_head(gate_input).squeeze(-1)).squeeze(0)
        scores = coarse_scores.float() + self.residual_scale * gate * residual
        return scores, gate, residual


class CandidateIndependentFormulaRanker(nn.Module):
    """候选独立的分子式条件精排器。

    每个候选只与查询光谱和已知分子式交互，不读取候选集合上下文，
    因而单候选分数不会随同池候选的增删而改变。候选维度为 [N,D]，
    结构特征为 [N,F]，输出分数为 [N]。
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        candidate_feature_dim: int = 8,
        formula_dim: int = len(FORMULA_ELEMENTS),
        d_model: int = 256,
        set_stats_dim: int = 5,
        max_gate: float = 0.45,
        residual_scale: float = 0.80,
    ) -> None:
        super().__init__()
        if min(embedding_dim, candidate_feature_dim, formula_dim, d_model, set_stats_dim) < 1:
            raise ValueError("候选独立精排器维度必须为正")
        self.embedding_dim = int(embedding_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.formula_dim = int(formula_dim)
        self.d_model = int(d_model)
        self.set_stats_dim = int(set_stats_dim)
        self.max_gate = float(max_gate)
        self.residual_scale = float(residual_scale)
        self.candidate_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.candidate_feature_dim),
            nn.Linear(self.embedding_dim + self.candidate_feature_dim, self.d_model),
            nn.GELU(),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.formula_dim),
            nn.Linear(self.embedding_dim + self.formula_dim, self.d_model),
            nn.GELU(),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(4 * self.d_model),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(self.d_model, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim + self.formula_dim),
            nn.Linear(self.embedding_dim + self.formula_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # 零初始化残差使训练起点严格退化为粗排分数。
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.constant_(self.gate_head[-1].bias, -2.0)

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        candidate_features: torch.Tensor,
        query_embedding: torch.Tensor,
        formula_features: torch.Tensor,
        set_stats: torch.Tensor,
        coarse_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入候选 [N,D]/[N,F] 与查询 [D]/[F]，返回分数 [N]。"""
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate_embeddings 应为 [N,D]")
        if candidate_features.ndim != 2 or candidate_features.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("candidate_features 与候选数不一致")
        if candidate_embeddings.shape[1] != self.embedding_dim:
            raise ValueError("candidate_embeddings 维度与模型配置不一致")
        if candidate_features.shape[1] != self.candidate_feature_dim:
            raise ValueError("candidate_features 维度与模型配置不一致")
        if query_embedding.ndim != 1 or query_embedding.shape[0] != self.embedding_dim:
            raise ValueError("query_embedding 应为 [D]")
        if formula_features.ndim != 1 or formula_features.shape[0] != self.formula_dim:
            raise ValueError("formula_features 应为 [F]")
        if set_stats.ndim != 1 or set_stats.shape[0] != self.set_stats_dim:
            raise ValueError("set_stats 维度异常")
        if coarse_scores.ndim != 1 or coarse_scores.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("coarse_scores 应为 [N]")
        # candidate_tokens: [N,D+F] -> [N,d_model]；每行独立计算。
        candidate_tokens = self.candidate_proj(
            torch.cat([candidate_embeddings.float(), candidate_features.float()], dim=-1)
        )
        query_input = torch.cat([query_embedding.float(), formula_features.float()], dim=0)
        query_token = self.query_proj(query_input).unsqueeze(0).expand(candidate_tokens.shape[0], -1)
        interaction = torch.cat(
            [
                candidate_tokens,
                query_token,
                torch.abs(candidate_tokens - query_token),
                candidate_tokens * query_token,
            ],
            dim=-1,
        )
        residual = torch.tanh(self.score_head(interaction).squeeze(-1))
        # 门控只依赖查询和已知分子式；不能使用候选集合统计，否则分数会随池组成改变。
        gate_input = torch.cat([query_embedding.float(), formula_features.float()], dim=0).unsqueeze(0)
        gate = self.max_gate * torch.sigmoid(self.gate_head(gate_input).squeeze(-1)).squeeze(0)
        scores = coarse_scores.float() + self.residual_scale * gate * residual
        return scores, gate, residual


__all__ = [
    "SpectrumGraphContrastiveModel",
    "symmetric_infonce_loss",
    "SetTransformerListwiseRanker",
    "FormulaConditionedSetTransformerRanker",
    "CandidateIndependentFormulaRanker",
    "listwise_set_loss",
]
