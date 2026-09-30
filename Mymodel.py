"""Width-160 GPT with SwiGLU, RMSNorm, RoPE, and gated attention."""
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        # The baseline MLP has about 8 * width**2 weights. SwiGLU uses
        # three projections, so 8 * width / 3 keeps its weight count similar.
        hidden = (8 * width) // 3
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).type_as(x) * self.weight


def precompute_rope(head_dim, context, base=10000.0):
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context, dtype=torch.float32)
    angles = torch.outer(positions, frequencies)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    seq_len = x.shape[2]
    cos = cos[:seq_len].to(device=x.device, dtype=x.dtype)[None, None, :, :]
    sin = sin[:seq_len].to(device=x.device, dtype=x.dtype)[None, None, :, :]
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin,
                      second * cos + first * sin), dim=-1)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        head_dim = width // heads
        self.norm1 = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attn_gate = nn.Linear(width, heads)
        self.proj = nn.Linear(width, width)
        self.mlp = SwiGLU(width)

    def forward(self, x, rope_cos, rope_sin):
        batch, length, width = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        gate = torch.sigmoid(self.attn_gate(h)).transpose(1, 2).unsqueeze(-1)
        attended = attended * gate
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        rope_cos, rope_sin = precompute_rope(width // config['heads'], self.context)
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)
        self.blocks = nn.ModuleList(
            [Block(width, config['heads']) for _ in range(config['depth'])]
        )
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.constant_(block.attn_gate.bias, 2.0)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
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
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    model_config = dict(config)
    model_config['width'] = 160
    model_config['depth'] = 6
    model_config['heads'] = 5
    return GPT(model_config)
