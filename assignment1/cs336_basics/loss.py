import torch

def cross_entropy(
    logits: torch.Tensor , targets: torch.Tensor
) -> torch.Tensor:
    """
    输入：
        logits (Float[Tensor, "batch_size, seq, vocab_size"])
            Transformer 输出的 logits
        targets (Int[Tensor, "batch_size"])
            每次预测对应的正确 token 编号

    输出：
        所有预测的平均 loss
    """
    
    # 按照作业的要求，先获取每个 batch 里面最大的 logit 分数
    max = torch.max(logits, dim=-1, keepdim=True).values
    
    # 减去最大项
    logits = logits - max
    
    """
    用 target 取出对应的 token，shape 变化：
    targets                 (batch_size, seq)
        ↓ unsqueeze(-1)
    索引张量                 (batch_size, seq, 1)
        ↓ gather
    取出的正确 token 分数     (batch_size, seq, 1)
        ↓ squeeze(-1)
    target_logits           (batch_size, seq)
    """
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    
    """
    参考文章里面的公式，每个位置的 loss = log(sum(exp(logits)) - target_logits
    计算 loss，shape 变化：
    logits（已减去最大值）      (batch_size, seq, vocab_size)
        ↓ exp
    每个分数的指数值            (batch_size, seq, vocab_size)
        ↓ sum(dim=-1)
    每个位置的指数和            (batch_size, seq)
        ↓ log
    每个位置的对数值            (batch_size, seq)
        ↓ 减去 target_logits    (batch_size, seq)
    每个位置的 loss            (batch_size, seq)
        ↓ mean()
    所有位置的平均 loss         ()，标量 Tensor
    """
    loss = torch.log(torch.sum(torch.exp(logits), dim=-1)) - target_logits
    
    return torch.mean(loss)