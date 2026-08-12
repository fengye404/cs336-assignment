import torch

# Linear 实际上就是内部维护了一个可训练的权重 tensor（cs336 作业中不包含 bias 偏置）
# 输入 tensor 经过权重 tensor 向量计算后，输出一个新的 tensor
# 输入和输出的 tensor 的 shape 可以不一样，也就是构造时的 in_features、out_features
# 因此可训练的权重 tensor 的 shape 就是 (out_features, in_features)
# 为什么不是 (in_features, out_features)? 因为 linear forward 的公式是 y=x@W.T，而不是 y=W@x，因为 pytorch 中向量是 row-vector 的
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 用 troch empty 申请出一个 tensor，devie、dtype 直接透传
        # in_features 输入维度
        # out_features 输出维度
        self.W = torch.nn.Parameter(
            torch.empty(
                (out_features, in_features),
                device = device,
                dtype = dtype
            )
        )
        
        # 用 assignment1 里面给出的初始化公式进行初始化
        # W ~ 𝒩(μ = 0, σ² = 2 / (d_in + d_out))，截断到 [-3σ, 3σ]
        
        # 先计算标准差
        std = (2 / (in_features+out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.W,
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std
        )
    
    # linear 实际上就是一个简单的线性变换，在 pytorch 中表达为 y=x@W.T，「.T」 表示转置
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


# embedding 实际上就是把一个 tensor 中的每个元素都替换为 vector
# 因此 embedding 中需要维护一个可训练的映射表
# forward 就是执行映射
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        # num_embeddings 词表大小
        # embedding_dim 每个 token 的维度
        self.embedding = torch.nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device = device,
                dtype = dtype
            )
        )
        
        # 用 assignment1 里面给出的初始化公式进行初始化
        # W ~ 𝒩(μ = 0, σ² = 1，截断到 [-3, 3]
        torch.nn.init.trunc_normal_(
            self.embedding,
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
        
    # embedding 实际上就是一个查表操作
    # forward 的输入 tensor shape 为 (batch, seq_len)
    # 里面执行的逻辑就是对每个 batch 的每个 seq 元素，执行一下查表操作
    
    # 怎么理解这个 []？
    # 实际上 python 中 A[B] 等价于 A.__getitem__(B)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]