import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


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

'''
class TermAwarePooling(nn.Module):
    """
    GO -> Protein 的术语感知池化（term-aware graph readout）
    - go_mat:   [T, H]   T个GO术语的向量（可来自你的GO-GNN编码）
    - node_h:   [N, H]   batch后所有图的节点特征（来自你的GNN）
    - batch_ids:[N]      每个节点属于第几号图（0..B-1）
    - num_graphs: int    B（本batch图的数量）

    返回:
    - ctx:   [B, T, H]   每个样本×每个术语的图级向量（term-specific）
    - attn:  [B, T, N]   （可解释）样本b、术语t 对 N个节点的注意力权重
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1,
                 temperature: float = 1.0, proj_out: bool = True):
        super().__init__()
        assert dim % heads == 0, "hidden dim 必须能被 heads 整除"
        self.h = heads
        self.dk = dim // heads
        self.temperature = float(temperature)

        # 线性映射到 Q/K/V 空间
        self.Wq = nn.Linear(dim, dim, bias=False)  # GO -> Q
        self.Wk = nn.Linear(dim, dim, bias=False)  # node -> K
        self.Wv = nn.Linear(dim, dim, bias=False)  # node -> V

        # 多头拼接后的输出投影
        self.proj_out = proj_out
        if proj_out:
            self.proj = nn.Linear(dim, dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self,
                go_mat: torch.Tensor,        # [T, H]
                node_h: torch.Tensor,        # [N, H]
                batch_ids: torch.Tensor,     # [N] in {0..B-1}
                num_graphs: int):

        T, H = go_mat.shape
        N, Hn = node_h.shape
        B = int(num_graphs)
        assert H == Hn, f"维度不一致: go_mat:{H} vs node_h:{Hn}"
        assert batch_ids.shape[0] == N, "batch_ids 长度应与节点数 N 相等"

        # —— 安全检查：避免出现空图（否则 masked softmax 全是 -inf）——
        counts = torch.bincount(batch_ids, minlength=B)
        if (counts == 0).any():
            raise ValueError("TermAwarePooling: 检测到空图（某个样本没有节点），请在构建batch时过滤。")

        # 1) 线性映射到 Q/K/V，并拆多头
        Q = self.Wq(go_mat).contiguous()   # [T, H]
        K = self.Wk(node_h).contiguous()   # [N, H]
        V = self.Wv(node_h).contiguous()   # [N, H]

        # 2) reshape -> [h, *, dk]
        #   注意 permute 后 .contiguous()，保证后续 view 的内存布局安全
        Q = Q.view(T, self.h, self.dk).permute(1, 0, 2).contiguous()  # [h, T, dk]
        K = K.view(N, self.h, self.dk).permute(1, 0, 2).contiguous()  # [h, N, dk]
        V = V.view(N, self.h, self.dk).permute(1, 0, 2).contiguous()  # [h, N, dk]

        # 3) 计算注意力打分：每个术语对所有节点
        #    scores[h, t, n] = <Q[h,t], K[h,n]> / (sqrt(dk) * temperature)
        scores = torch.einsum('htd,hnd->htn', Q, K)
        scores = scores / (math.sqrt(self.dk) * self.temperature)      # [h, T, N]

        # 4) —— 按图 masked softmax ——（避免不同图之间“串味”）
        #    mask[b, n] = True 表示第 b 个图的节点 n
        device = scores.device
        mask = F.one_hot(batch_ids, num_classes=B).T.to(device=device, dtype=torch.bool)  # [B, N]

        # 扩成 [h, T, B, N]，对每个图做 softmax
        scores4 = scores.unsqueeze(2).expand(-1, -1, B, -1)            # [h, T, B, N]
        neg_inf = torch.finfo(scores4.dtype).min                       # 半精度安全的 -inf 近似
        scores4 = scores4.masked_fill(~mask.unsqueeze(0).unsqueeze(1), neg_inf)

        attn4 = torch.softmax(scores4, dim=-1)                         # [h, T, B, N]
        attn4 = self.drop(attn4)                                       # Dropout 在权重上

        # 5) 用注意力对 V 做加权汇聚 -> term-specific 上下文
        #    ctx4[h, t, b, dk] = sum_n attn4[h, t, b, n] * V[h, n, dk]
        ctx4 = torch.einsum('htbn,hnd->htbd', attn4, V)                # [h, T, B, dk]

        # 6) 还原成 [B, T, H]，可选输出投影
        ctx = ctx4.permute(2, 1, 0, 3).contiguous().view(B, T, H)      # [B, T, H]
        if self.proj_out:
            ctx = self.proj(ctx)                                       # [B, T, H]

        return ctx
'''

from typing import Optional, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TermAwarePoolingLite(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int = 4,
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 proj_out: bool = True,
                 chunk_t: Optional[int] = None):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dk = dim // heads
        self.temperature = float(temperature)
        self.chunk_t = chunk_t

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

        self.proj_out = proj_out
        if proj_out:
            self.proj = nn.Linear(dim, dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def _split_heads(self, X: torch.Tensor):  # [L,H] -> [h,L,dk]
        L, H = X.shape
        X = X.view(L, self.h, self.dk).permute(1, 0, 2).contiguous()
        return X

    def forward(self,
                go_mat: torch.Tensor,                 # [T,H]
                node_h: torch.Tensor,                 # [N,H]
                num_nodes_per_graph: Union[torch.Tensor, List[int]]):  # 长度 B
        device = node_h.device
        if not torch.is_tensor(num_nodes_per_graph):
            num_nodes_per_graph = torch.tensor(num_nodes_per_graph, device=device, dtype=torch.long)
        B = int(num_nodes_per_graph.numel())

        T, H = go_mat.shape
        N, Hn = node_h.shape
        assert H == Hn, f"H mismatch: {H} vs {Hn}"
        assert num_nodes_per_graph.sum().item() == N, "sum(num_nodes_per_graph) != N"

        # 预先映射 Q/K/V
        Q_all = self._split_heads(self.Wq(go_mat).contiguous())         # [h,T,dk]
        K_all = self.Wk(node_h).contiguous()                            # [N,H]
        V_all = self.Wv(node_h).contiguous()                            # [N,H]

        ctx_list = []
        start = 0
        for b in range(B):
            n_b = int(num_nodes_per_graph[b].item())
            end = start + n_b
            Kb = K_all[start:end, :]                                    # [n_b,H]
            Vb = V_all[start:end, :]
            start = end

            Kb_h = self._split_heads(Kb)                                # [h,n_b,dk]
            Vb_h = self._split_heads(Vb)                                # [h,n_b,dk]

            if self.chunk_t is None or self.chunk_t >= T:
                scores_b = torch.matmul(Q_all, Kb_h.transpose(-2, -1))  # [h,T,n_b]
                scores_b = scores_b / (math.sqrt(self.dk) * self.temperature)
                attn_b = torch.softmax(scores_b, dim=-1)                # [h,T,n_b]
                attn_b = self.drop(attn_b)
                ctx_b = torch.matmul(attn_b, Vb_h)                      # [h,T,dk]
            else:
                chunks = []
                for t0 in range(0, T, self.chunk_t):
                    t1 = min(t0 + self.chunk_t, T)
                    Q_ch = Q_all[:, t0:t1, :]                           # [h,tC,dk]
                    scores_ch = torch.matmul(Q_ch, Kb_h.transpose(-2, -1))  # [h,tC,n_b]
                    scores_ch = scores_ch / (math.sqrt(self.dk) * self.temperature)
                    attn_ch = torch.softmax(scores_ch, dim=-1)          # [h,tC,n_b]
                    attn_ch = self.drop(attn_ch)
                    ctx_ch = torch.matmul(attn_ch, Vb_h)                # [h,tC,dk]
                    chunks.append(ctx_ch)
                ctx_b = torch.cat(chunks, dim=1)                        # [h,T,dk]

            ctx_b = ctx_b.permute(1, 0, 2).contiguous().view(T, H)      # [T,H]
            if self.proj_out:
                ctx_b = self.proj(ctx_b)                                # [T,H]
            ctx_list.append(ctx_b)

        ctx = torch.stack(ctx_list, dim=0)                               # [B,T,H]
        return ctx


class GCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, head):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(0.3)

        self.liner1 = nn.Linear(in_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)


        self.conv1 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)

        self.term_pool = TermAwarePoolingLite(hidden_dim, head, 0.2)

    def forward(self, g, h, go_mat):
        with g.local_scope():
            g.ndata['h'] = h
            init_avg_h = dgl.mean_nodes(g, 'h')

        h = self.liner1(h)

        pre = h
        h = self.bn1(h)
        h = pre + self.dropout(F.relu(self.conv1(g, h)))  # , edge_weight=ew

        pre = h
        h = self.bn2(h)
        h = pre + self.dropout(F.relu(self.conv2(g, h)))

        node_x = h

        with g.local_scope():
            g.ndata['output'] = node_x
            readout = dgl.sum_nodes(g, "output")

        num_nodes_per_graph = g.batch_num_nodes().tolist()

        ctx = self.term_pool(go_mat, node_x, num_nodes_per_graph)

        return readout, init_avg_h, ctx

class combine_inter_model(nn.Module):

    def __init__(self, graph_size, graph_hid, label_num, head):

        super(combine_inter_model, self).__init__()

        self.GNN = GCN(graph_size, graph_hid, head)
        self.go_embeddings = GOEmbedding(label_num, graph_hid)
        self.GAT = GOGATLayer(graph_hid, graph_hid//head, head)

        self.gate_drop = nn.Dropout(0.2)

        self.classify1 = nn.Sequential(
            nn.BatchNorm1d(graph_hid * 2),
            nn.Linear(graph_hid * 2, graph_hid * 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear((graph_hid * 2), label_num)
        )

        self.classify2 = nn.Sequential(
            #nn.LayerNorm(graph_hid),
            nn.Linear(graph_hid, graph_hid),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(graph_hid, 1)
        )

        self.classify = nn.Sequential(
            nn.Linear(graph_hid*3, graph_hid),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(graph_hid, 1)
        )

        self.fuse_alpha = nn.Parameter(torch.zeros(label_num))
        self.bn1 = nn.BatchNorm1d(graph_hid)
        self.bn2 = nn.BatchNorm1d(graph_hid)
        self.ctx_bn = nn.BatchNorm1d(graph_hid)

        self.sample_proj = nn.Sequential(
            nn.BatchNorm1d(graph_hid * 2),
            nn.Linear(graph_hid * 2, graph_hid),
            nn.Dropout(0.2)
        )

        self.fuse_gate = nn.Sequential(
            nn.Linear(2 * graph_hid, graph_hid),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(graph_hid, 1)
        )

    def forward(self, graph, graph_h, gog):

        go_emb = self.go_embeddings()

        go_emb_matrix = self.GAT(go_emb, gog)

        graph_readout, init_avg_h, ctx = self.GNN(graph, graph_h, go_emb_matrix)
        fea1 = torch.cat((init_avg_h, graph_readout), 1)

        ctx = ctx.permute(0, 2, 1).contiguous()  # [B, H, T]
        ctx = self.ctx_bn(ctx)  # BN1d(H) over [B, T]
        ctx = ctx.permute(0, 2, 1).contiguous()

        logits1 = self.classify1(fea1)
        logits2 = self.classify2(ctx).squeeze(-1)


        alpha = torch.sigmoid(self.fuse_alpha).unsqueeze(0)  # [1, T]
        logits = alpha * logits1 + (1.0 - alpha) * logits2

        return logits

    def forward_back1(self, graph, graph_h, gog):

        go_emb = self.go_embeddings()

        go_emb_matrix = self.GAT(go_emb, gog)

        graph_readout, init_avg_h, ctx = self.GNN(graph, graph_h, go_emb_matrix)
        graph_readout = self.bn1(graph_readout)
        init_avg_h = self.bn2(init_avg_h)

        B, T, H = ctx.shape

        gr_a = graph_readout.unsqueeze(1).expand(B, T, H)  # [B, T, H]
        gr_b = init_avg_h.unsqueeze(1).expand(B, T, H)  # [B, T, H]

        # 4) 合并三者：ctx（term-specific）+ readout（sample-global）+ init（sample-baseline）
        feat = torch.cat([ctx, gr_a, gr_b], dim=-1)

        '''

        feat  = feat.permute(0, 2, 1).contiguous()  # [B, H, T]
        feat  = self.ctx_bn(feat)  # BN1d(H) over [B, T]
        feat  = feat.permute(0, 2, 1).contiguous()
        '''

        logits = self.classify(feat).squeeze(-1)

        return logits

    def forward_back2(self, graph, graph_h, gog):
        go_emb = self.go_embeddings()

        go_emb_matrix = self.GAT(go_emb, gog)

        graph_readout, init_avg_h, ctx = self.GNN(graph, graph_h, go_emb_matrix)
        fea1 = torch.cat((init_avg_h, graph_readout), 1)

        logits1 = self.classify1(fea1)
        logits2 = self.classify2(ctx).squeeze(-1)

        s = self.sample_proj(fea1)

        B, T, H = ctx.shape
        s_b = s.unsqueeze(1).expand(B, T, H)
        gate = torch.sigmoid(self.fuse_gate(torch.cat([ctx, s_b], dim=-1))).squeeze(-1)  # [B,T]
        logits = gate * logits2 + (1.0 - gate) * logits1  # [B,T]
        return logits








