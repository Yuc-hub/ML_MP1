"""LLaMA-style GPT with modern techniques from LLaMA 2/3, Gemma 2, DeepSeek.

Improvements over baseline (GPT-2 style, width=128, depth=4, ~1.09M params):
  1. RoPE           – rotary position encoding (LLaMA)
  2. RMSNorm        – simpler norm, no mean-centering (LLaMA)
  3. SwiGLU         – gated MLP with SiLU activation (LLaMA/PaLM)
  4. Width 160      – wider representations
  5. Depth 6        – deeper network
  6. Residual scale – 1/sqrt(depth) per residual branch (DeepSeek-V2)
  7. QK-Norm        – RMSNorm on Q,K before attention (Gemma 2 / DeepSeek-V2)
  8. Dropout 0.1    – regularization on attn/MLP/embedding
  9. GQA            – grouped query attention: 5 Q heads, 1 KV head (LLaMA 2/3)
  10. Wider MLP     – reinvest GQA param savings into MLP capacity

Run: python train.py --steps 2400 --run-dir runs/test06
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

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
    def __init__(self, width, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim, bias=False)
        self.up   = nn.Linear(width, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, width, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# Transformer block with GQA (Grouped Query Attention, LLaMA 2/3)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, width, n_heads, n_kv_heads, mlp_hidden,
                 residual_scale=1.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = width // n_heads
        self.kv_group_size = n_heads // n_kv_heads  # Q heads per KV head
        self.residual_scale = residual_scale

        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)

        # GQA: separate Q and KV projections (LLaMA 2/3)
        self.q_proj  = nn.Linear(width, n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(width, 2 * n_kv_heads * self.head_dim, bias=False)

        # QK-Norm (Gemma 2 / DeepSeek-V2)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.proj = nn.Linear(width, width, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.mlp = SwiGLUMLP(width, mlp_hidden, dropout)

    def forward(self, x, rope_cos, rope_sin):
        batch, length, width = x.shape
        h = self.norm1(x)

        # Q: [batch, n_heads, length, head_dim]
        q = (self.q_proj(h)
             .view(batch, length, self.n_heads, self.head_dim)
             .transpose(1, 2))
        # KV: [batch, n_kv_heads, length, head_dim] each
        kv = (self.kv_proj(h)
              .view(batch, length, 2, self.n_kv_heads, self.head_dim)
              .permute(2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        # QK-Norm + RoPE
        q = _apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = _apply_rope(self.k_norm(k), rope_cos, rope_sin)

        # Expand KV heads to match Q heads (GQA broadcast)
        if self.kv_group_size > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.kv_group_size, -1, -1)
            k = k.reshape(batch, self.n_heads, length, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.kv_group_size, -1, -1)
            v = v.reshape(batch, self.n_heads, length, self.head_dim)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = attn.transpose(1, 2).reshape(batch, length, width)
        x = x + self.attn_drop(self.proj(out)) * self.residual_scale
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
        n_heads = config['heads']
        n_kv_heads = config.get('kv_heads', n_heads)
        depth = config.get('depth', 4)
        dropout = config.get('dropout', 0.0)

        # SwiGLU hidden dim, rounded to multiple of 8
        mlp_ratio = config.get('mlp_ratio', 8 / 3)
        mlp_hidden = ((int(width * mlp_ratio) + 7) // 8) * 8

        # Residual scaling: 1/sqrt(depth)
        residual_scale = 1.0 / math.sqrt(depth)

        # Embedding + dropout
        self.token = nn.Embedding(config['vocab'], width)
        self.embed_drop = nn.Dropout(dropout)

        # RoPE cos/sin tables
        head_dim = width // n_heads
        cos, sin = _precompute_rope(head_dim, self.context)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(width, n_heads, n_kv_heads, mlp_hidden, residual_scale, dropout)
            for _ in range(depth)
        ])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)

        # Initialize and tie embeddings
        self.apply(self._init_weights)
        self.head.weight = self.token.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.embed_drop(self.token(ids))
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
    improved['width']     = 160   # wider:  128 -> 160
    improved['heads']     = 5     # Q heads: 5  (head_dim = 32)
    improved['kv_heads']  = 1     # KV heads: 1  (GQA, LLaMA 2/3 style)
    improved['depth']     = 6     # deeper: 4 -> 6
    improved['dropout']   = 0.1   # regularization
    improved['mlp_ratio'] = 3.5   # wider MLP (reinvest GQA param savings)
    return StudentGPT(improved)
