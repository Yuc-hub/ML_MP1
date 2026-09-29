"""LLaMA-style GPT: RoPE + RMSNorm + SwiGLU, deeper (6 layers vs baseline 4).

Diagnosis: the baseline uses absolute position embeddings, LayerNorm, and a
plain GELU MLP.  Modern LLM research (LLaMA, Llama-2) shows that swapping
these for RoPE, RMSNorm, and SwiGLU consistently lowers loss at matched
parameter counts.  The parameter savings from removing the learned position
embedding (~33K) plus a slight SwiGLU hidden-dim adjustment let us add two
extra transformer blocks (4 -> 6) within budget.

Ablation plan (to isolate each mechanism):
  A) baseline                        -> ~2.10 BPB  (reference)
  B) baseline + RoPE only            -> measure
  C) baseline + RMSNorm only         -> measure
  D) baseline + SwiGLU only          -> measure
  E) full (RoPE+RMSNorm+SwiGLU, 6L) -> target < 2.10
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019).
    Simpler than LayerNorm: no mean-centering, no bias, just scale."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_rope(head_dim, max_len, base=10000.0):
    """Precompute cosine/sine tables for Rotary Position Embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len)
    angles = torch.outer(t, freqs)          # [max_len, head_dim/2]
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x, cos, sin):
    """Apply RoPE to query or key tensor [batch, heads, seq_len, head_dim]."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)   # [1, 1, seq, d/2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLUMLP(nn.Module):
    """Gated MLP with SiLU activation (Shazeer, 2020).
    Three projections (gate, up, down) instead of two, but hidden_dim is
    reduced to ~8/3 * width so total params stay comparable to a 4x GELU MLP."""
    def __init__(self, width, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim, bias=False)
        self.up   = nn.Linear(width, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, width, heads, mlp_hidden):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv  = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.mlp  = SwiGLUMLP(width, mlp_hidden)

    def forward(self, x, rope_cos, rope_sin):
        batch, length, width = x.shape
        # --- self-attention with RoPE ---
        h = self.norm1(x)
        q, k, v = (self.qkv(h)
                    .view(batch, length, 3, self.heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4))           # 3 x [B, H, T, D]
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, length, width))
        # --- feed-forward ---
        x = x + self.mlp(self.norm2(x))
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

        # SwiGLU hidden dim: 8/3 * width, rounded up to multiple of 8
        mlp_hidden = ((width * 8 // 3 + 7) // 8) * 8    # 128 -> 344

        # Token embedding (no position embedding -- RoPE handles positions)
        self.token = nn.Embedding(config['vocab'], width)

        # Precompute RoPE cos/sin tables as non-learned buffers
        head_dim = width // heads
        cos, sin = _precompute_rope(head_dim, self.context)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(width, heads, mlp_hidden) for _ in range(depth)]
        )
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)

        # Initialize and tie embeddings
        self.apply(self._init_weights)
        self.head.weight = self.token.weight      # weight tying

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
    improved['depth'] = 6       # deeper: 4 -> 6 blocks
    return StudentGPT(improved)
