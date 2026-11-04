import torch
import torch.nn.functional as F

from model import local_attention


def test_local_attention_matches_a_dense_window_mask():
    torch.manual_seed(0)
    B, H, T, D, W = 3, 4, 150, 24, 64
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True) for _ in range(3))
    valid = torch.arange(T)[None] < torch.tensor([150, 97, 10])[:, None]
    t = torch.arange(T)
    dense = ((t[None] <= t[:, None]) & (t[None] > t[:, None] - W))[None, None] & valid[:, None, None, :]
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=dense)
    got = local_attention(q, k, v, valid, W, 0.0)
    keep = valid[:, None, :, None]  # padded frames have no keys in a window and are never used
    assert torch.allclose(got * keep, want * keep, atol=1e-10)
    g_want = torch.autograd.grad((want * keep).sum(), (q, k, v))
    g_got = torch.autograd.grad((got * keep).sum(), (q, k, v))
    assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(g_got, g_want))
