import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GOGATLayer(nn.Module):
    """
    X: [N, Fin]   —— N个GO节点的输入特征
    A: [N, N]     —— 0/1或bool的邻接矩阵（非零=有边）
    输出: [N, H*Fout]
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)                 # 训练时对注意力权重做dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope)     # GAT常用LeakyReLU

        # 1) 线性投影 W: [Fin, H*Fout]，一次性得到所有head的通道
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # 2) 每个head的注意力向量 a_src, a_dst
        #    形状: [H, Fout, 1]，与TF版本等价（左/右分量相加）
        self.a_src = nn.Parameter(torch.Tensor(num_heads, out_dim, 1))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, out_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)   # 与glorot一致
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    @torch.no_grad()
    def _ensure_bool_mask(self, A: torch.Tensor) -> torch.Tensor:
        # 把0/1矩阵统一转成bool，便于后续where/masked_fill
        if A.dtype != torch.bool:
            return A != 0
        return A

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: [N, Fin]
        A: [N, N]
        return: [N, H*Fout]
        """
        N = X.size(0)
        device = X.device                                  # （小细节：这行变量未使用，可以删）
        A_mask = self._ensure_bool_mask(A)                 # [N, N] bool

        # ---- 1) 线性投影并拆head ----
        XW = self.W(X)                                     # [N, H*F]
        XW = XW.view(N, self.num_heads, self.out_dim)      # [N, H, F]
        XW = XW.transpose(0, 1)                            # [H, N, F] —— 按head分组

        # ---- 2) 注意力打分（加性注意力，左右分量相加）----
        f1 = torch.matmul(XW, self.a_src)                  # [H, N, 1] = Wh_i · a_src
        f2 = torch.matmul(XW, self.a_dst)                  # [H, N, 1] = Wh_j · a_dst
        e  = f1 + f2.transpose(1, 2)                       # [H, N, N] 逐对(i,j)相加
        e  = self.leaky_relu(e)                            # 激活

        # ---- 3) 掩码：非邻接置 -inf（确保该处softmax→0）----
        neg_inf = torch.finfo(e.dtype).min                 # 根据dtype选择极小值
        mask = A_mask.unsqueeze(0).expand(self.num_heads, -1, -1)   # [H, N, N]
        e = torch.where(mask, e, torch.full_like(e, neg_inf))       # 非邻接= -inf

        # ⚠️ 注意：若某一行全False（该节点无出边），这一行全为 -inf，
        # softmax会产生NaN。务必保证 A 对角线为1（自环）或行内至少一个True。

        # ---- 4) 归一化 + dropout ----
        attn = torch.softmax(e, dim=-1)                    # [H, N, N] 行内归一化
        attn = self.dropout(attn)                          # 训练期dropout注意力权重（不重归一）

        # ---- 5) 聚合邻居特征 & 合并heads ----
        h_prime = torch.matmul(attn, XW)                   # [H, N, F] = ∑_j α_{ij} * Wh_j
        h_prime = h_prime.transpose(0, 1).contiguous()     # [N, H, F]
        h_prime = h_prime.view(N, self.num_heads * self.out_dim)     # [N, H*F]
        h_prime = F.elu(h_prime)                           # GAT常用ELU
        return h_prime

from typing import Optional, Tuple

class GOEmbedding(nn.Module):
    def __init__(self, label_size: int, emb_dim: int,
                 init_weight: Optional[torch.Tensor] = None,
                 trainable: bool = True):
        super().__init__()
        if init_weight is None:
            self.emb = nn.Embedding(label_size, emb_dim)
        else:
            self.emb = nn.Embedding.from_pretrained(init_weight, freeze=not trainable)
        self.label_size = label_size
        self.emb_dim = emb_dim

    def forward(self, term_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        term_ids: [C] 或 [B,C] 的 LongTensor；若 None，则自动生成 [0..label_size-1]
        return:
            term_vec: [C, emb_dim] 或 [B, C, emb_dim]
            term_ids: 实际使用的 ids（便于对齐邻接矩阵等）
        """
        if term_ids is None:
            device = next(self.parameters()).device
            term_ids = torch.arange(self.label_size, dtype=torch.long, device=device)
        term_vec = self.emb(term_ids)

        return term_vec

class Cross_attention_interpro(nn.Module):
    """
    q: [B, D]    —— InterPro 向量（每个样本一条）
    k: [N, D]    —— GO embedding matrix（术语数 N=T）
    v: [N, D]    —— 通常和 k 相同（这里就是 go embedding）
    返回: [B, D]
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.2, name: str = "cross_attention"):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.d_head     = hidden_dim // num_heads

        # W_q, W_k, W_v（等价于 tf.get_variable 初始化的矩阵）
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Xavier/Glorot 初始化（与 tf.glorot_uniform_initializer 一致）
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: [B, D], k: [N, D], v: [N, D]
        """
        B, D = q.shape
        N, Dk = k.shape
        Nv, Dv = v.shape
        assert D == self.hidden_dim and Dk == D and Dv == D and N == Nv, "维度不匹配"

        # 线性映射
        q_proj = self.W_q(q)   # [B, D]
        k_proj = self.W_k(k)   # [N, D]
        v_proj = self.W_v(v)   # [N, D]

        H, dh = self.num_heads, self.d_head

        # 多头拆分（Head-first），完全对应你的 TF 版：
        # q: [B,D] -> [H,B,1,dh]
        qh = q_proj.view(B, H, dh).permute(1, 0, 2).unsqueeze(2).contiguous()  # [H, B, 1, dh]

        # k: [N,D] -> [H,1,N,dh]
        kh = k_proj.view(N, H, dh).permute(1, 0, 2).unsqueeze(1).contiguous()  # [H, 1, N, dh]
        vh = v_proj.view(N, H, dh).permute(1, 0, 2).unsqueeze(1).contiguous()  # [H, 1, N, dh]

        # 注意力分数：逐 head、逐样本，对所有 GO 术语做点积
        # score: [H, B, N]
        score = (qh * kh).sum(dim=-1) / math.sqrt(dh)

        # softmax + dropout（只在 model.train() 下生效）
        weights = F.softmax(score, dim=-1)     # [H, B, N]
        weights = self.attn_drop(weights)      # [H, B, N]

        # context = sum_n alpha * v   → [H, B, dh]
        context = (weights.unsqueeze(-1) * vh).sum(dim=2)                       # [H, B, dh]

        # 合并多头 → [B, D]
        out = context.permute(1, 0, 2).contiguous().view(B, D)                  # [B, D]

        return out

class GCN(nn.Module):

    def __init__(self, in_dim1, in_dim2, in_dim3, embed_dim, hidden_dim):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(0.3)

        self.liner1 = nn.Linear(in_dim2, embed_dim)
        self.liner2 = nn.Linear(embed_dim + in_dim3, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv1 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)

    def forward(self, g, h1, h2, h3):


        h2 = self.liner1(h2)
        h = torch.cat((h2, h3), -1)
        h = self.liner2(h)

        pre = h
        h = self.bn1(h)
        h = pre + self.dropout(F.relu(self.conv1(g, h)))  # , edge_weight=ew

        pre = h
        h = self.bn2(h)
        h = pre + self.dropout(F.relu(self.conv2(g, h)))

        with g.local_scope():
            g.ndata['output'] = h
            readout = dgl.mean_nodes(g, "output")
            return readout

class combine_inter_model(nn.Module):

    def __init__(self, graph_size1, graph_size2, graph_size3, interpro_size, embedding_hid, graph_hid, label_num, head):

        super(combine_inter_model, self).__init__()

        self.GNN = GCN(graph_size1, graph_size2, graph_size3, embedding_hid, graph_hid)
        self.go_embeddings = GOEmbedding(label_num, graph_hid)
        self.GAT = GOGATLayer(graph_hid, graph_hid // head, head)
        self.liner1 = nn.Linear(interpro_size, graph_hid)
        self.liner2 = nn.Linear(graph_hid*2, graph_hid)

        self.classify = nn.Sequential(
            nn.Linear(graph_hid, graph_hid*2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(graph_hid*2, label_num)
        )
        self.dropout = nn.Dropout(0.2)

        self.classify_interpro = nn.Linear(graph_hid, label_num)
        self.ca_interpro = Cross_attention_interpro(graph_hid, head, 0.2)
        self.ln1 = nn.LayerNorm(graph_hid*2)
        self.bn1 = nn.BatchNorm1d(graph_hid)

    def forward_without_interpro(self, graph, graph_h1, graph_h2, graph_h3, interpro_h, gog):

        go_emb = self.go_embeddings()
        go_emb_matrix = self.GAT(go_emb, gog)

        graph_feature = self.GNN(graph, graph_h1, graph_h2, graph_h3)

        combine_h = self.bn1(graph_feature)

        combine_h = combine_h + self.dropout(self.ca_interpro(combine_h, go_emb_matrix, go_emb_matrix))


        return self.classify(combine_h)

    def forward_all(self, graph, graph_h1, graph_h2, graph_h3, interpro_h, gog):

        go_emb = self.go_embeddings()
        go_emb_matrix = self.GAT(go_emb, gog)

        graph_feature = self.GNN(graph, graph_h1, graph_h2, graph_h3)
        interpro_h = self.dropout(F.relu(self.liner1(interpro_h)))

        combine_h = torch.cat((graph_feature, interpro_h), -1)
        combine_h = self.ln1(combine_h)
        combine_h = self.dropout(F.relu(self.liner2(combine_h)))

        combine_h = combine_h + self.dropout(self.ca_interpro(combine_h, go_emb_matrix, go_emb_matrix))

        return self.classify(combine_h)

    def forward(self, graph, graph_h1, graph_h2, graph_h3, interpro_h, gog):

        #go_emb = self.go_embeddings()
        #go_emb_matrix = self.GAT(go_emb, gog)

        graph_feature = self.GNN(graph, graph_h1, graph_h2, graph_h3)
        #interpro_h = self.dropout(F.relu(self.liner1(interpro_h)))

        #combine_h = torch.cat((graph_feature, interpro_h), -1)
        combine_h = self.bn1(graph_feature)
        #combine_h = self.dropout(F.relu(self.liner2(combine_h)))

        #combine_h = combine_h + self.dropout(self.ca_interpro(combine_h, go_emb_matrix, go_emb_matrix))

        return self.classify(combine_h)



