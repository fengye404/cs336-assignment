import pytest
import torch

from cs336_basics.model import MultiheadSelfAttention


@pytest.mark.parametrize('batch_size', [2, 4])
def test_rope_batched_positions_match_separate_sequences(batch_size):
    torch.manual_seed(42)
    attention = MultiheadSelfAttention(16, 4, theta=10000, max_seq_len=32)
    x = torch.randn(batch_size, 5, 16)
    # 不同文本的位置不同，防止 batch 和 head 恰好同大时静默错位。
    positions = torch.arange(5).expand(batch_size, -1) + torch.arange(batch_size)[:, None]
    expected = torch.cat([
        attention(x[i:i+1], positions[i]) for i in range(batch_size)
    ])
    torch.testing.assert_close(attention(x, positions), expected)
    shared = torch.arange(5)
    torch.testing.assert_close(attention(x), attention(x, shared))
    torch.testing.assert_close(attention(x, positions), attention(x, positions[:, None, :]))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
def test_attention_preserves_requested_dtype(dtype):
    attention = MultiheadSelfAttention(16, 4, theta=10000, max_seq_len=16, dtype=dtype)
    x = torch.randn(2, 5, 16, dtype=dtype)
    assert all(p.dtype == dtype for p in attention.parameters())
    output = attention(x)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in attention.parameters())


def test_attention_constructor_places_rope_buffers_on_device():
    # meta 可在没有 CUDA 的机器上检查构造函数的设备传递。
    attention = MultiheadSelfAttention(16, 4, theta=10000, max_seq_len=16, device='meta')
    assert all(p.device.type == 'meta' for p in attention.parameters())
    assert all(b.device.type == 'meta' for b in attention.buffers())
