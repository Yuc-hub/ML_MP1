"""LLaMA/Gemma2/PaLM-style GPT with modern techniques.

Improvements over baseline (GPT-2 style, width=128, depth=4, ~1.09M params):
  1. RoPE           – rotary position encoding (LLaMA)
  2. RMSNorm        – simpler normalization (LLaMA)
  3. SwiGLU         – gated MLP (LLaMA / PaLM)
  4. Width 160      – wider representations
  5. Depth 6        – deeper network
  6. QK-Norm        – normalize Q,K before attention (Gemma 2 / DeepSeek-V2)
  7. Embedding ×√d  – scale embeddings to match residual stream (Transformer / GPT-3)
  8. Scaled init    – residual branches init std /= √(2·depth) (GPT-2 / GPT-NeoX)
  9. Parallel block – attention & MLP in parallel (PaLM / GPT-J)

Ablation results:
  A) baseline                                  -> 2.10 BPB
  B) RoPE+RMSNorm+SwiGLU, depth=6, width=128  -> 1.85 BPB
  C) + width=160, residual scaling             -> 1.80 BPB
  D) + QK-Norm, EmbedScale, ScaledInit, PaLM   -> measure
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
    """Gated MLP: SiLU(gate) * up -> down  (Shazeer 2020, used in LLaMA)."""
    def __init__(self, width, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim, bias=False)
        self.up   = nn.Linear(width, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Parallel Transformer block (PaLM / GPT-J style)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Parallel attention + MLP block with QK-Norm.

    PaLM style:  x = x + Attn(norm(x)) + MLP(norm(x))
    Instead of:  x = x + MLP(norm(x + Attn(norm(x))))

    Benefits: single norm, better gradient flow, slightly faster (parallelism).
    """
    def __init__(self, width, heads, mlp_hidden):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        # Single pre-norm for both branches (PaLM style)
        self.norm = RMSNorm(width)
        # Attention
        self.qkv  = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)   # QK-Norm (Gemma 2 / DeepSeek-V2)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width, bias=False)
        # MLP
        self.mlp = SwiGLUMLP(width, mlp_hidden)

    def forward(self, x, rope_cos, rope_sin):
        batch, length, width = x.shape
        h = self.norm(x)
        # --- attention branch ---
        q, k, v = (self.qkv(h)
                    .view(batch, length, 3, self.heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4))
        q = _apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = _apply_rope(self.k_norm(k), rope_cos, rope_sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = self.proj(attn.transpose(1, 2).reshape(batch, length, width))
        # --- MLP branch (parallel, shares the same norm input) ---
        mlp_out = self.mlp(h)
        # --- combine ---
        return x + attn_out + mlp_out


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
        self.embed_scale = math.sqrt(width)     # Embedding scaling (GPT-3)

        # SwiGLU hidden dim: 8/3 * width, rounded to multiple of 8
        mlp_hidden = ((width * 8 // 3 + 7) // 8) * 8

        # Token embedding (RoPE handles positions, no pos embedding needed)
        self.token = nn.Embedding(config['vocab'], width)

        # RoPE cos/sin tables
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

        # --- Initialization ---
        self.apply(self._init_weights)
        # Scaled init: residual-contributing projections get smaller init
        # to prevent residual stream from growing with depth (GPT-2 / GPT-NeoX)
        residual_std = 0.02 / math.sqrt(2 * depth)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, std=residual_std)
        # Weight tying
        self.head.weight = self.token.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) * self.embed_scale     # scale embeddings by √width
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
    improved['width'] = 160     # wider:  128 -> 160
    improved['heads'] = 5       # heads:  4 -> 5  (head_dim = 32)
    improved['depth'] = 6       # deeper: 4 -> 6 blocks
    return StudentGPT(improved)
