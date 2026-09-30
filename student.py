"""Compact gated-attention GPT for the MP1 next-token task."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019). No mean-centering, no bias."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_rope(head_dim, max_len, base=10000.0):
    """Precompute cos/sin tables for Rotary Position Embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x, cos, sin):
    """Apply RoPE to [batch, heads, seq_len, head_dim]."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLUMLP(nn.Module):
    """Gated MLP: SiLU(gate) * up -> down  (Shazeer 2020, LLaMA)."""
    def __init__(self, width, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim, bias=False)
        self.up   = nn.Linear(width, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width, heads, mlp_hidden, residual_scale=1.0):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.residual_scale = residual_scale

        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv  = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width, bias=False)
        self.attn_gate = nn.Linear(width, heads)
        self.mlp  = SwiGLUMLP(width, mlp_hidden)

    def forward(self, x, rope_cos, rope_sin):
        batch, length, width = x.shape
        h = self.norm1(x)
        q, k, v = (self.qkv(h)
                    .view(batch, length, 3, self.heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4))
        q = _apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = _apply_rope(self.k_norm(k), rope_cos, rope_sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        gate = torch.sigmoid(self.attn_gate(h)).transpose(1, 2).unsqueeze(-1)
        attn = attn * gate
        attn_out = self.proj(attn.transpose(1, 2).reshape(batch, length, width))
        x = x + attn_out * self.residual_scale
        x = x + self.mlp(self.norm2(x)) * self.residual_scale
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        heads = config['heads']
        depth = config.get('depth', 4)

        mlp_hidden = ((width * 8 // 3 + 7) // 8) * 8
        residual_scale = 1.0 / math.sqrt(depth)

        self.token = nn.Embedding(config['vocab'], width)

        head_dim = width // heads
        cos, sin = _precompute_rope(head_dim, self.context)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        self.blocks = nn.ModuleList(
            [Block(width, heads, mlp_hidden, residual_scale) for _ in range(depth)]
        )
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)

        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.constant_(block.attn_gate.bias, 2.0)
        self.head.weight = self.token.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities."""
        return F.log_softmax(self(ids).float(), dim=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config):
    improved = dict(config)
    improved['width'] = 160
    improved['heads'] = 5
    improved['depth'] = 6
    return StudentGPT(improved)
