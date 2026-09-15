import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool

class MolecularGraphEncoder(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=256, num_layers=5, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # 初始特征嵌入层
        self.atom_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.bond_encoder = nn.Linear(edge_in_dim, hidden_dim)

        # 虚拟节点 (Virtual Node) 嵌入初始化
        self.virtualnode_embedding = nn.Embedding(1, hidden_dim)

        # 虚拟节点的 MLP 更新网络 (逐层更新)
        self.vn_mlp = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.vn_mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ))

        # GINE 卷积层与批归一化
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):

            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )

            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        # ==========================================
        # 将图结构特征强转为 Float，消除 Dtype 冲突
        # ==========================================
        x = x.float()
        edge_attr = edge_attr.float()

        # 将原始特征映射到隐空间
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        # 初始化虚拟节点 (对于 batch 中的每个图，生成一个对应的 VN 特征)
        num_graphs = batch.max().item() + 1
        virtualnode_feat = self.virtualnode_embedding(
            torch.zeros(num_graphs, dtype=torch.long, device=x.device)
        )

        for i in range(self.num_layers):
            # 虚拟节点广播: VN 特征加到对应的真实原子特征上
            x = x + virtualnode_feat[batch]

            # 图卷积消息传递
            x_residual = x
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)


            x = x + x_residual


            if i < self.num_layers - 1:
                vn_update = global_add_pool(x_residual, batch)
                virtualnode_feat = virtualnode_feat + self.vn_mlp[i](virtualnode_feat + vn_update)
                virtualnode_feat = F.dropout(virtualnode_feat, p=self.dropout, training=self.training)

        # 全局平均池化，输出最终的分子图深层表征
        mol_representation = global_mean_pool(x, batch)
        return mol_representation


class GraphTower(nn.Module):
    """
    完整的模块二图塔 (Graph Tower)
    封装了 GINE 骨干网络、预训练头和用于 FAISS 的对齐投影头
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=256, final_dim=512, num_fgs=15):
        super().__init__()

        # 核心骨干网络
        self.encoder = MolecularGraphEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim
        )

        # 投影头 (Projection Head): 映射到与光谱对齐的 512 维空间
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, final_dim)
        )

        # 预训练头: 用于预测官能团或分子性质，增强表征提取能力
        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_fgs)
        )

    def forward(self, x, edge_index, edge_attr, batch, return_pretrain=False):
        # 获取图级表征 [B, hidden_dim]
        mol_embeds = self.encoder(x, edge_index, edge_attr, batch)

        # 映射到对齐空间 [B, final_dim]
        z_mol = self.projection_head(mol_embeds)

        # L2 归一化 (跨模态 InfoNCE 检索标配)
        z_mol = F.normalize(z_mol, p=2, dim=-1)

        # 如果需要辅助预训练损失，则返回预训练头的输出
        if return_pretrain:
            fg_logits = self.pretrain_head(mol_embeds)
            return z_mol, fg_logits

        return z_mol

# ================= 测试代码 =================
if __name__ == '__main__':
    from physchemrag.shared.dataset import PhysChemRADataset
    from torch_geometric.loader import DataLoader
    from physchemrag.config import DATA_DIR


    dataset = PhysChemRADataset(root=str(DATA_DIR))

    if len(dataset) > 0:
        sample = dataset[0]
        node_dim = sample.x.shape[1]
        edge_dim = sample.edge_attr.shape[1]

        print(f"检测到输入维度: 原子特征={node_dim}, 化学键特征={edge_dim}")

        # 实例化 Graph Tower
        tower = GraphTower(node_in_dim=node_dim, edge_in_dim=edge_dim, final_dim=512)

        # 用 DataLoader 取出一个 Batch 测试
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        batch_data = next(iter(loader))

        # 前向传播 (含可选的预训练输出)
        z_mol, fg_logits = tower(
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
            return_pretrain=True
        )


        print(f"输出维度: z_mol={z_mol.shape}, fg_logits={fg_logits.shape}")
