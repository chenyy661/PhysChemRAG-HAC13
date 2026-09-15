"""IR + 已知分子式分支的条件化双塔检索适配器。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from physchemrag.shared.chemical_domain import (
    FORMULA_ELEMENTS,
    require_formula_domain,
)
from physchemrag.shared.molecular_formula import formula_counts


FORMULA_DIM = len(FORMULA_ELEMENTS)


def formula_vector(formula: str) -> torch.Tensor:
    """将闭域分子式转换为稳定的 10 维条件向量。"""
    require_formula_domain(formula)
    counts = formula_counts(formula)
    total = max(sum(counts.values()), 1)
    values = []
    for element in FORMULA_ELEMENTS:
        count = counts.get(element, 0)
        values.append(float(math.log1p(count) / math.log1p(total)))
    vector = torch.tensor(values, dtype=torch.float32)
    if vector.shape != (FORMULA_DIM,) or not torch.isfinite(vector).all():
        raise ValueError(f"分子式向量异常: {formula}")
    return vector


class FormulaConditionedDualAdapter(nn.Module):
    """在冻结双塔向量上学习分子式条件化的残差变换。

    适配后仍输出单个归一化向量，因此可用于分子式分片索引、精确内积检索
    或小型 FAISS 索引；不会引入 Cross-Encoder 的逐候选高成本。
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        formula_dim: int = FORMULA_DIM,
        hidden_dim: int = 256,
        residual_scale: float = 0.35,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or formula_dim < 1 or hidden_dim < 32:
            raise ValueError("embedding_dim、formula_dim 必须为正，hidden_dim 至少为 32")
        self.embedding_dim = embedding_dim
        self.formula_dim = formula_dim
        self.residual_scale = float(residual_scale)

        self.formula_encoder = nn.Sequential(
            nn.Linear(formula_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        adapter_input_dim = embedding_dim + 64
        self.query_adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.molecule_adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.query_gate = nn.Linear(adapter_input_dim, 1)
        self.molecule_gate = nn.Linear(adapter_input_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        # 适配器从严格恒等映射开始，避免随机初始化在训练前破坏双塔排序。
        nn.init.zeros_(self.query_adapter[-1].weight)
        nn.init.zeros_(self.query_adapter[-1].bias)
        nn.init.zeros_(self.molecule_adapter[-1].weight)
        nn.init.zeros_(self.molecule_adapter[-1].bias)
        nn.init.constant_(self.query_gate.bias, -2.0)
        nn.init.constant_(self.molecule_gate.bias, -2.0)

    def encode_query(
        self, query_embedding: torch.Tensor, formula_features: torch.Tensor
    ) -> torch.Tensor:
        """编码查询；输入和输出形状均为 [batch_size, embedding_dim]。"""
        if query_embedding.ndim != 2:
            raise ValueError(f"query_embedding 应为 [B,D]，实际为 {tuple(query_embedding.shape)}")
        if formula_features.ndim != 2:
            raise ValueError(f"formula_features 应为 [B,F]，实际为 {tuple(formula_features.shape)}")
        if query_embedding.shape[0] != formula_features.shape[0]:
            raise ValueError("查询与分子式特征 batch 维度不一致")
        if query_embedding.shape[1] != self.embedding_dim or formula_features.shape[1] != self.formula_dim:
            raise ValueError("查询或分子式特征维度与模型配置不一致")
        formula = self.formula_encoder(formula_features)
        # conditioned: [B,D] + [B,64] -> [B,D+64]
        conditioned = torch.cat([query_embedding, formula], dim=-1)
        delta = self.query_adapter(conditioned)
        gate = torch.sigmoid(self.query_gate(conditioned))
        return F.normalize(query_embedding + self.residual_scale * gate * delta, p=2, dim=-1)

    def encode_molecule(
        self, molecule_embedding: torch.Tensor, formula_features: torch.Tensor
    ) -> torch.Tensor:
        """编码候选；支持 [B,D] 或 [B,N,D] 输入。"""
        if molecule_embedding.ndim not in (2, 3):
            raise ValueError("molecule_embedding 应为 [B,D] 或 [B,N,D]")
        if formula_features.ndim != 2 or formula_features.shape[0] != molecule_embedding.shape[0]:
            raise ValueError("候选与分子式特征 batch 维度不一致")
        if molecule_embedding.shape[-1] != self.embedding_dim or formula_features.shape[1] != self.formula_dim:
            raise ValueError("候选或分子式特征维度与模型配置不一致")
        formula = self.formula_encoder(formula_features)
        if molecule_embedding.ndim == 2:
            conditioned = torch.cat([molecule_embedding, formula], dim=-1)
        else:
            candidate_count = molecule_embedding.shape[1]
            # formula: [B,64] -> [B,N,64]
            formula = formula.unsqueeze(1).expand(-1, candidate_count, -1)
            conditioned = torch.cat([molecule_embedding, formula], dim=-1)
        delta = self.molecule_adapter(conditioned)
        gate = torch.sigmoid(self.molecule_gate(conditioned))
        return F.normalize(molecule_embedding + self.residual_scale * gate * delta, p=2, dim=-1)

    def forward(
        self,
        query_embedding: torch.Tensor,
        molecule_embedding: torch.Tensor,
        formula_features: torch.Tensor,
    ) -> torch.Tensor:
        """返回候选内积 logits；输出形状为 [batch_size, candidate_count]。"""
        if molecule_embedding.ndim != 3:
            raise ValueError("forward 的 molecule_embedding 必须为 [B,N,D]")
        query = self.encode_query(query_embedding, formula_features)
        molecule = self.encode_molecule(molecule_embedding, formula_features)
        # query: [B,D] -> [B,1,D]，与 molecule [B,N,D] 做候选内积。
        cosine = (query.unsqueeze(1) * molecule).sum(dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return cosine * scale


__all__ = [
    "FORMULA_DIM",
    "FORMULA_ELEMENTS",
    "FormulaConditionedDualAdapter",
    "formula_vector",
]
