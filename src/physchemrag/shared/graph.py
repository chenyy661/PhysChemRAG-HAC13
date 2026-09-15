"""分子图构造的唯一公共入口。"""

from __future__ import annotations

import torch
from torch_geometric.utils.smiles import from_smiles


def graph_data(smiles: str):
    data = from_smiles(smiles)
    if data is None or data.x is None:
        return None
    # x: [N,F_atom]；edge_attr: [E,F_bond]，GraphTower 使用浮点特征。
    data.x = data.x.float()
    data.edge_attr = (
        data.edge_attr.float()
        if data.edge_attr is not None
        else torch.zeros((0, 3), dtype=torch.float32)
    )
    return data

