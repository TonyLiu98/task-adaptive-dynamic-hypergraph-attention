"""Task-adaptive dynamic hypergraph attention model for C-MAPSS FD001."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax


class DynamicHypergraphBuilder(nn.Module):
    """
    Build a task-level hypergraph from shared node and hyperedge embeddings.
    Each sensor channel is a node; the topology is shared across samples.
    """

    def __init__(self, sensor_num, h_model, hyper_num, k, device):
        super().__init__()
        self.sensor_num = sensor_num  # 节点数等于传感器个数
        self.hyperedge_num = hyper_num  # 超边的个数
        self.k = k  # 每个节点连接的top-k超边
        self.alpha_prob = 1.0  # 相似度放大系数，用于激活softmax更突出大值控制超边的学习稳定性
        self.alpha_weight = 1.0  # 放大系数控制信息权重
        self.ac = torch.nn.Tanh()

        '''
        初始化节点和超边的 embedding（每个传感器、每个超边都有一个可学习的 d_model 维嵌入向量）。
        作用： 给节点和超边引入“身份特征”，让模型可以学习到节点/超边在超图中的角色。
        '''
        self.node_embed = nn.Embedding(sensor_num, h_model).to(device)
        self.hyper_embed = nn.Embedding(hyper_num, h_model).to(device)

    def forward(self, batch_size, device):
        hyperedge_indices = []

        for b in range(batch_size):
            node_emb = self.node_embed(torch.arange(self.sensor_num).to(device))  # [sensor_num, d_model]
            hyper_emb = self.hyper_embed(torch.arange(self.hyperedge_num).to(device))  # [hyperedge_num, d_model]

            sim = torch.mm(node_emb, hyper_emb.t())  # [sensor_num, hyper_num]

            # sim1 = F.normalize(sim, dim=1)
            sim1 = F.relu(self.alpha_prob * sim)
            sim2 = F.relu(self.alpha_weight * sim)

            prob_for_topk = F.softmax(sim1, dim=1)
            # Top-k selects discrete indices; selected softmax scores provide differentiable weights.
            _, topk_idx = prob_for_topk.topk(self.k, dim=1)

            connect_mask = torch.zeros_like(prob_for_topk, dtype=torch.bool)
            connect_mask.scatter_(1, topk_idx, True)  # shape: [sensor_num, hyper_num]

            # 用于卷积加权的边权 softmax（可能与连接用的 softmax 不同）
            # prob_for_weight = F.softmax(self.alpha_weight * sim, dim=1)
            prob_for_weight = F.softmax(sim2, dim=1)
            # prob_for_weight = sim2
            # 根据连接掩码提取边索引和权重
            node_idx, edge_idx = connect_mask.nonzero(as_tuple=True)  # [num_edges]
            edge_index = torch.stack([node_idx, edge_idx], dim=0)  # shape: [2, num_edges]
            edge_weight = prob_for_weight[node_idx, edge_idx]  # shape: [num_edges]
            # 保存该样本的超图结构
            hyperedge_indices.append((edge_index, edge_weight))
        return hyperedge_indices


class Multi_headHGC(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, dropout, device):
        super().__init__(aggr='add')  # 聚合方式设为加和
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.device = device

        # 一次性计算 Q, K, V
        self.qkv_proj = nn.Linear(in_dim, 3 * out_dim).to(device)
        self.q_proj = nn.Linear(in_dim, out_dim).to(device)
        self.k_proj = nn.Linear(in_dim, out_dim).to(device)
        self.v_proj = nn.Linear(in_dim, out_dim).to(device)

        # 每个注意力头的可学习参数；包含节点到超边和超边到节点两次传播的两组注意力权重矩阵参数
        self.attn2edge = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim)).to(device)
        nn.init.xavier_uniform_(self.attn2edge)

        self.attn2node = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim)).to(device)
        nn.init.xavier_uniform_(self.attn2node)

        # 二次传播时Q2，K2，V2
        self.fck2 = torch.nn.Linear(self.head_dim, self.head_dim).to(device)
        self.fcv2 = torch.nn.Linear(self.head_dim, self.head_dim).to(device)

        # 输出映射
        self.out_proj = nn.Linear(out_dim, out_dim).to(device)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)

        # QKV projection 并 reshape 为多头形式
        # qkv = self.qkv_proj(x)  # [Node, 3 * out_dim]
        # qkv = qkv.view(num_nodes, 3, self.num_heads, self.head_dim)  # [Node, 3, Head, out_dim]
        # Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [Node, Head, out_dim]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 展平：变为 [Head*Node, out_dim]
        Q = Q.transpose(0, 1).reshape(-1, self.head_dim)
        K = K.transpose(0, 1).reshape(-1, self.head_dim)
        V = V.transpose(0, 1).reshape(-1, self.head_dim)

        # 复制边：让每个头都能使用相同 edge_index，但 index + 偏移
        edge_index1 = edge_index.repeat(1, self.num_heads)  # [2, E*H]
        num_edges_per_head = edge_index.size(1)  # ← 原始 E
        head_offset = torch.arange(self.num_heads, device=x.device).repeat_interleave(num_edges_per_head) * num_nodes
        edge_index1 = edge_index1 + head_offset.unsqueeze(0)  # 广播偏移每个头

        if edge_weight is not None:
            ew1 = edge_weight.repeat(self.num_heads)  # [E*H]
        else:
            ew1 = None

        # 第一次传播: 节点 -> 超边
        edge_deg = degree(edge_index1[1], self.num_heads * num_nodes, dtype=Q.dtype)
        norm1 = 1.0 / edge_deg[edge_index1[1]]
        attn2edge = self.attn2edge.repeat_interleave(num_edges_per_head, dim=0)  # [E*H, 2D]
        out = self.propagate(edge_index1, Q=Q, K=K, V=V, edge_weight=ew1,
                             attn_weight=attn2edge, norm=norm1, size=(self.num_heads * num_nodes,) * 2)

        # 将输出特征从 [H*N, D] 映射为[N, H, D]
        out_hyper = out.view(self.num_heads, num_nodes, self.head_dim)

        out_hyper = self.dropout(out_hyper)

        out1 = out_hyper
        out1 = out1.view(self.num_heads, num_nodes, self.head_dim).permute(1, 0, 2).reshape(num_nodes, -1)

        # 第二次传播: 超边 -> 节点
        attn2node = self.attn2node.repeat_interleave(num_edges_per_head, dim=0)  # [E*H, 2D]
        Q2 = out_hyper.transpose(0, 1).reshape(-1, self.head_dim)  # flatten to [H*N, D]
        K2 = self.fck2(Q2)  # 此处可加入新线性层或复用
        V2 = self.fcv2(Q2)
        edge_index_2 = edge_index[[1, 0], :].repeat(1, self.num_heads)
        edge_index_2 = edge_index_2 + head_offset.unsqueeze(0)

        edge_deg2 = degree(edge_index_2[1], self.num_heads * num_nodes, dtype=Q2.dtype)
        norm2 = 1.0 / edge_deg2[edge_index_2[1]]
        if edge_weight is not None:
            ew2 = ew1  # [E*H]
        else:
            ew2 = None

        out_final = self.propagate(edge_index_2, Q=Q2, K=K2, V=V2, attn_weight=attn2node, edge_weight=ew2,
                                   norm=norm2, size=(self.num_heads * num_nodes,) * 2)
        out_final = out_final.view(self.num_heads, num_nodes, self.head_dim).permute(1, 0, 2).reshape(num_nodes, -1)
        out_final = self.out_proj(out_final)
        out_final = self.dropout(out_final)

        return out1, out_final

    def message(self, Q_j, K_j, V_j, attn_weight, index, size_i, norm, edge_weight):
        z = torch.cat((Q_j, K_j), dim=-1)  # 拼接Q与K，dim=[E*H, 2* head_dim]
        score = (z * attn_weight).sum(dim=-1) / (self.head_dim ** 0.5)  # [E*H]
        alpha = softmax(score, index, num_nodes=size_i)  # 按每个节点聚合
        msg = alpha.view(-1, 1) * V_j

        if norm is not None:
            msg = msg * norm.view(-1, 1)
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)  # 关键：把可导边权乘进去
        return msg


class HyperGraphNet(nn.Module):
    """
    主模型：超图构建 + 超图卷积 + RUL预测
    """

    def __init__(self, sensor_num, seq_len, d_model, h_model, hyperedge_num, k, head, dropout, device):
        super().__init__()
        self.seq_len = seq_len
        self.sensor_num = sensor_num
        self.d_model = d_model
        self.edge_num = hyperedge_num
        self.device = device
        self.k = k
        self.head = head
        self.h_model = h_model

        self.fc1 = nn.Linear(self.seq_len, self.d_model).to(device)
        self.hyper_builder = DynamicHypergraphBuilder(sensor_num=self.sensor_num, h_model=self.h_model,
                                                      hyper_num=self.edge_num, k=self.k, device=self.device)
        self.trans_gnn = Multi_headHGC(in_dim=self.d_model, out_dim=self.d_model, num_heads=self.head, dropout=dropout, device=self.device)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(4 * d_model, d_model).to(self.device)
        )

        self.fc2 = nn.Linear(self.d_model, 1).to(self.device)
        self.fc3 = nn.Linear(self.sensor_num, 1).to(self.device)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.ac1 = torch.nn.Sigmoid()
        self.ac2 = torch.nn.ReLU()

    def forward(self, x, return_structure=False):
        """
        x: [batch_size, seq_len, sensor_num]
        """
        batch_size = x.size(0)

        # Step1: 从时间维度聚合成传感器特征
        x = x.permute(0, 2, 1)  # x = [batch_size, sensor_num, seq_len]
        node_feats = self.fc1(x)  # x = [batch_size, sensor_num, d_model]
        batch_out1 = torch.Tensor().to(self.device)
        batch_out2 = torch.Tensor().to(self.device)
        # Step2: 每个样本单独跑超图卷积
        hyperedge_indices = self.hyper_builder(batch_size, self.device)

        for b in range(batch_size):
            node_feat_b = node_feats[b].to(self.device)  # [sensor_num, d_model]
            edge_index_b, edge_weight_b = hyperedge_indices[b]
            edge_index_b = edge_index_b.to(self.device)
            edge_weight_b = edge_weight_b.to(self.device)

            gnn_out1, gnn_out2 = self.trans_gnn(node_feat_b, edge_index_b,
                                                edge_weight=edge_weight_b)  # [sensor_num, d_model]

            # fnn_out = gnn_out1 + gnn_out2
            out1 = gnn_out1.unsqueeze(0)
            out2 = gnn_out2.unsqueeze(0)
            batch_out1 = torch.cat((batch_out1, out1), dim=0)
            batch_out2 = torch.cat((batch_out2, out2), dim=0)

        # Step3: 预测 RUL

        ffn_out = self.ffn(batch_out1 + batch_out2)
        y = self.fc2(ffn_out)  # y = [batch_size, sensor_num]
        y = y.squeeze(-1)
        pred = self.fc3(y)  # y = [batch_size, 1]

        if return_structure:
            return pred, hyperedge_indices
        else:
            return pred
