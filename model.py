"""Kuyil's acoustic model, rebuilt from PiVo's report and the tensor shapes in its released checkpoint.

Text encoder: token + voice embeddings, then blocks alternating grouped-query attention (RMS norm, rotary
positions, q/k norm, gated-SiLU feed-forward) and dilated convolution (1, 2, 4, ...).
Durations: a width-64 predictor of log(1 + frames) per token.
Decoder: each token's encoding repeated for its frames, plus PiVo's 4 within-phone timing signals, through
causal attention blocks that look back `window` frames (rotary positions in the first only); predicts the
20 FARGAN features, normalized. At PiVo's sizes (the defaults) it has about 585k parameters.
Inside details PiVo didn't publish (norm placement in conv blocks, rotary base) are our choices.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 128
    voices: int = 2
    width: int = 96
    ff_width: int = 192
    heads: int = 4
    kv_heads: int = 2
    text_layers: int = 6
    decoder_layers: int = 2
    kernel: int = 5
    duration_width: int = 64
    feature_dim: int = 20
    window: int = 64
    dropout: float = 0.1
    max_token_frames: int = 200


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotary(x, pos):
    half = x.shape[-1] // 2
    freq = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos[:, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


def local_attention(q, k, v, valid, window, p):
    """Causal attention over the last `window` frames, the same as a dense (T, T) mask but in blocks of `window`
    frames that each see themselves and the block before: memory and work grow with T, not T squared."""
    B, H, T, D = q.shape
    n, pad = -(-T // window), -T % window
    blocks = lambda x: F.pad(x, (0, 0, 0, pad)).view(B, H, n, window, D).transpose(1, 2)  # (B, n, H, W, D)
    two = lambda x: torch.cat([F.pad(x, (0, 0) * (x.dim() - 2) + (1, 0))[:, :-1], x], -2)  # previous block, then this
    q, k, v = blocks(q), two(blocks(k)), two(blocks(v))
    ok = two(F.pad(valid, (0, pad)).view(B, n, window)[..., None]).squeeze(-1)  # (B, n, 2W) key frames that exist
    i, j = torch.arange(window, device=q.device)[:, None], torch.arange(2 * window, device=q.device)
    mask = (j > i) & (j <= i + window) & ok[:, :, None, None, :]  # key in (t - window, t]
    y = F.scaled_dot_product_attention(*(x.reshape(B * n, *x.shape[2:]) for x in (q, k, v)),
                                       attn_mask=mask.reshape(B * n, 1, window, 2 * window), dropout_p=p)
    return y.reshape(B, n, H, window, D).transpose(1, 2).reshape(B, H, n * window, D)[:, :, :T]


class AttentionBlock(nn.Module):
    def __init__(self, c: Config, use_rotary=True, window=0):
        super().__init__()
        self.h, self.kv, self.hd, self.rot, self.p = c.heads, c.kv_heads, c.width // c.heads, use_rotary, c.dropout
        self.window = window  # >0: causal local attention, and forward() takes the (B, T) frame mask
        self.attention = nn.Module()
        a = self.attention
        a.q = nn.Linear(c.width, c.width, bias=False)
        a.k = nn.Linear(c.width, c.kv_heads * self.hd, bias=False)
        a.v = nn.Linear(c.width, c.kv_heads * self.hd, bias=False)
        a.out = nn.Linear(c.width, c.width, bias=False)
        a.q_norm, a.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)
        self.norm1, self.norm2 = RMSNorm(c.width), RMSNorm(c.width)
        self.gate = nn.Linear(c.width, c.ff_width, bias=False)
        self.up = nn.Linear(c.width, c.ff_width, bias=False)
        self.down = nn.Linear(c.ff_width, c.width, bias=False)

    def forward(self, x, mask):
        a, (B, T, W) = self.attention, x.shape
        h = self.norm1(x)
        q = a.q_norm(a.q(h).view(B, T, self.h, self.hd)).transpose(1, 2)
        k = a.k_norm(a.k(h).view(B, T, self.kv, self.hd)).transpose(1, 2)
        v = a.v(h).view(B, T, self.kv, self.hd).transpose(1, 2)
        if self.rot:
            pos = torch.arange(T, device=x.device)
            q, k = rotary(q, pos), rotary(k, pos)
        k, v = k.repeat_interleave(self.h // self.kv, 1), v.repeat_interleave(self.h // self.kv, 1)
        p = self.p if self.training else 0.0
        y = (local_attention(q, k, v, mask, self.window, p) if self.window
             else F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p))
        x = x + F.dropout(a.out(y.transpose(1, 2).reshape(B, T, W)), self.p, self.training)
        h = self.norm2(x)
        return x + F.dropout(self.down(F.silu(self.gate(h)) * self.up(h)), self.p, self.training)


class ConvBlock(nn.Module):
    def __init__(self, c: Config, dilation):
        super().__init__()
        self.norm1, self.norm2, self.p = RMSNorm(c.width), RMSNorm(c.width), c.dropout
        self.conv = nn.Conv1d(c.width, c.width, c.kernel, padding=dilation * (c.kernel // 2), dilation=dilation, bias=False)

    def forward(self, x, valid):
        h = (self.norm1(x) * valid[..., None]).transpose(1, 2)
        return x + F.dropout(self.norm2(F.silu(self.conv(h).transpose(1, 2))), self.p, self.training)


class Durations(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        w, self.p = c.duration_width, c.dropout
        self.in_proj = nn.Linear(c.width, w)
        self.layers = nn.ModuleList(nn.Sequential(nn.Conv1d(w, w, 3, padding=1, groups=w), nn.Conv1d(w, w, 1)) for _ in range(2))
        self.norms = nn.ModuleList(RMSNorm(w) for _ in range(2))
        self.out = nn.Linear(w, 1)

    def forward(self, x, valid):
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            r = layer((h * valid[..., None]).transpose(1, 2)).transpose(1, 2)
            h = norm(h + F.dropout(F.silu(r), self.p, self.training))
        return self.out(h).squeeze(-1)


class Kuyil(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.register_buffer("feature_mean", torch.zeros(c.feature_dim))
        self.register_buffer("feature_std", torch.ones(c.feature_dim))
        self.tokens = nn.Embedding(c.vocab, c.width)
        self.voices = nn.Embedding(c.voices, c.width)
        self.text_blocks = nn.ModuleList(AttentionBlock(c) if i % 2 == 0 else ConvBlock(c, 2 ** (i // 2))
                                         for i in range(c.text_layers))
        self.text_norm = RMSNorm(c.width)
        self.durations = Durations(c)
        self.phase_projection = nn.Linear(4, c.width, bias=False)
        self.decoder = nn.ModuleList(AttentionBlock(c, use_rotary=i == 0, window=c.window) for i in range(c.decoder_layers))
        self.feature_norm = RMSNorm(c.width)
        self.feature_head = nn.Linear(c.width, c.feature_dim)

    def encode(self, tokens, voice, lengths):
        valid = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
        mask = valid[:, None, None, :]
        x = self.tokens(tokens) + self.voices(voice)[:, None]
        for b in self.text_blocks:
            x = b(x, mask) if isinstance(b, AttentionBlock) else b(x, valid)
        x = self.text_norm(x) * valid[..., None]
        return x, self.durations(x.detach(), valid), valid

    def decode(self, encoded, durations):
        """encoded (B, N, W), integer durations (B, N) -> normalized features (B, F, 20) and frame mask."""
        ends = durations.cumsum(1)
        frames = ends[:, -1]
        t = torch.arange(int(frames.max()), device=encoded.device)
        idx = torch.searchsorted(ends, t.expand(len(ends), -1).contiguous(), right=True).clamp(max=ends.shape[1] - 1)
        ex = torch.gather(encoded, 1, idx[..., None].expand(-1, -1, encoded.shape[-1]))  # each frame's token
        count = torch.gather(durations, 1, idx).float().clamp(min=1)
        frac = (t - torch.gather(ends - durations, 1, idx) + 0.5) / count
        # PiVo's timing signals (pivo/phase.py): progress, its sine and cosine, scaled log duration
        ph = torch.stack((2 * frac - 1, torch.sin(math.pi * frac), torch.cos(math.pi * frac),
                          torch.log1p(count) / math.log1p(self.c.max_token_frames)), -1)
        valid = t[None] < frames[:, None]
        x = (ex + self.phase_projection(ph)) * valid[..., None]
        for b in self.decoder:
            x = b(x, valid)
        return self.feature_head(self.feature_norm(x)), valid

    @torch.no_grad()
    def synthesize(self, tokens, voice):
        """One utterance: token ids (N,) and a voice id -> FARGAN features (F, 20), PiVo's inference rules."""
        self.eval()
        enc, logd, _ = self.encode(tokens[None], torch.tensor([voice], device=tokens.device),
                                   torch.tensor([len(tokens)], device=tokens.device))
        d = torch.expm1(logd[0]).round().clamp(0, self.c.max_token_frames).long()
        d[tokens >= 10] = d[tokens >= 10].clamp(min=1)  # every spoken phone gets a frame (ids 0-9 are special)
        d[tokens < 3] = 0  # PAD, BOS, EOS
        feats, _ = self.decode(enc, d[None])
        return feats[0] * self.feature_std + self.feature_mean

    def config(self):
        return asdict(self.c)
