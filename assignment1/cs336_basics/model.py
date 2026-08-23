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

# RMSNorm 公式：output = (x / sqrt(mean(x²) + eps)) × weight
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        # d_model 模型 token 维度；weight 用于最后做缩放，维度要和 d_model 一致
        # eps 用于防止分母为 0
        
        self.weight = torch.nn.Parameter(
            torch.ones(
                d_model,
                device = device,
                dtype = dtype
            )
        )
        self.eps = eps
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 按照 pdf 的要求，入参 x 需要先转到 32 位
        in_dtype = x.dtype
        x_float32 = x.to(torch.float32)
        
        # 应用公式
        # 注意 mean 只需要对最后维度求平均，keepdim 表示需要保留维度，这里 mean 后的 shape 就是 (batch, seq_len, 1)
        rms = torch.sqrt(
            torch.mean(
                torch.square(x_float32), 
                dim = -1,
                keepdim = True
            ) + self.eps
        )
        # 注意这里是 * weight 而不是 @，因为是逐元素相乘缩放
        # 这里计算需要注意一下 shape
        # x_floagt32(batch, seq_len, d_model)，rms(batch, seq_len, 1)
        # x_floagt32/rms 这里会用到 pytorch 里面的广播机制
        # 广播：两个 shape 不完全一样的 tensor 做逐元素运算时，PyTorch 会在尺寸为 1 的维度上，自动“重复使用”那个值，让 shape 对齐。
        result = (x_float32 / rms) * self.weight
        
        # 返回前转回原来的 dtype
        return result.to(in_dtype) 
    
    
# silu 公式：SiLU(x) = x × sigmoid(x)
# sigmoid(x) = 1 / (1 + e⁻ˣ)
def silu(in_features: torch.Tensor):
    return in_features * torch.sigmoid(in_features)

# SwiGLU 相比于通常的 FNN，实际上就是加了一层 gate 门控分支
# FNN：x -> Linear W1（通常维度扩大 4 倍） -> ReLU -> Linear W2（缩小四倍） -> 输出
#
# SwiGLU：
# x (d_model) -> Linear W1 (d_model -> d_ff) -> SiLU ─┐
#                                                   × -> Linear W2 (d_ff → d_model) → 输出
# x (d_model) -> Linear W3 (d_model -> d_ff)  ────────┘
# 公式：SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) = 𝑊2(SiLU(𝑊1𝑥) ⊙ 𝑊3𝑥)   ⊙表示逐元素相乘
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # w1、w3 负责升维
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)
        
        # w2 负责降维
        self.w2 = Linear(d_ff,d_model,device,dtype)
        
    def forward(self, x: torch.Tensor):
        a = silu(self.w1(x)) 
        b = self.w3(x)
        # 注意这里是是逐元素相乘
        return self.w2(a * b)
    

# 传统 Transformer 的绝对位置编码：
#
# token_ids = [101, 205, 307]            # token 在词表中的 ID
#              │    │    │
#              │    │    └── token_embedding[307] = e₂
#              │    └─────── token_embedding[205] = e₁
#              └──────────── token_embedding[101] = e₀
#
# token 向量： [e₀,      e₁,      e₂]
# 位置下标：    [ 0,       1,       2]
#                │        │        │
#                │        │        └── position_embedding[2] = p₂
#                │        └─────────── position_embedding[1] = p₁
#                └──────────────────── position_embedding[0] = p₀
#
# 位置向量： [p₀,      p₁,      p₂]
#                │        │        │
#                └────────逐元素相加────────┘
#                             ↓
# Transformer 输入：
# x = [e₀ + p₀,  e₁ + p₁,  e₂ + p₂]
#
# ==================== 分界线 ==========================
#
# 而 RoPE 作用在在 Attention 的 q、k 向量上
# 它会对每个 token 的 q、k 向量进行旋转，通过旋转来把位置信息编码到 q、k 中
# 例如对于某个 token X[n]，RoPE 会对它的 q、k 向量进行如下操作：
# 1、将 q、k 向量分为两两一组（q、k 向量的 shape 等于每个 token 的维度，现在的 transformer 里面 token 维度都是偶数）
#    例如 q 为 [q0, q1, q2, q3, q4, q5]，
#    会被分为 [(q0,q1), (q2,q3), (q4,q5)]
#
# 2、每一对维度都看作一个二维向量，并分别旋转
#    每一对的旋转角度 = token 的位置 n × 该维度对自己的旋转频率
#
#    例如：
#    (q0,q1) 按 n × f1 旋转
#    (q2,q3) 按 n × f2 旋转
#    (q4,q5) 按 n × f3 旋转
#    f_k = 1 / theta^((2k - 2) / d_k)
#
#    不同维度对的频率 f1、f2、f3 不同，
#    因此同一个 q/k 向量中的不同维度对，旋转角度通常也不同
#
# 3、q、k 的对应维度对，会按相同的位置 n 和相同频率旋转
#    得到旋转后的 q_rotated、k_rotated；它们的 shape 不变
#
#    q: [q0, q1, q2, q3, q4, q5]
#          ↓    ↓    ↓
#    q_rotated: [q0', q1', q2', q3', q4', q5']
#
# 4、之后 Attention 使用旋转后的 q、k 计算点积
#    q_rotated @ k_rotated.T
#
#    因此不同 token 的 q、k 点积不仅和内容有关，
#    还会受到两者相对位置距离的影响
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # theta：RoPE 的频率基数
        
        pass
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass