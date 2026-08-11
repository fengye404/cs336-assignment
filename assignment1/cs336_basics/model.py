import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super.__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 用 troch empty 申请出一个 tensor，devie、dtype 直接透传
        self.weight = torch.nn.Parameter(
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
            self.weight,
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std
        )
    
    # linear 实际上就是一个简单的线性变换，在 pytorch 中表达为 y=x@W.T，「.T」 表示转置
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T