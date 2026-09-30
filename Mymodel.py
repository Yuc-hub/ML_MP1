"""Width-160 causal LM; experiment notes are kept here for run tracking.

# try01: SwiGLU
# try02: width: 128 -> 160
# try03: depth: 4 -> 6
# try04: RMSNorm 替换 LayerNorm
# try05: RoPE 替换绝对位置嵌入
# try06: 增加 QK-RMSNorm 和门控
# try07: 实现 8Q/4KV + 保留现有 QK-RMSNorm 和门控
# try08: 10 Q / 5 KV
# try09: Value Residual：后续层复用第 1 层的 Value；每头系数零初始化、可学习

以上为实验记录；具体运行结果请与对应 run 目录中的指标一并记录。
"""
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


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = SwiGLU(width)

    def forward(self, x, first_value=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        # Value Residual: later layers can reuse the first layer's value stream.
        # A zero-initialized per-head coefficient keeps the starting model
        # exactly equivalent to the original attention and lets training learn
        # whether (and how strongly) the earlier features are useful.
        if first_value is not None:
            v = v + self.value_residual * first_value
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x)), v

    def add_value_residual(self, heads):
        # One learned mixing weight per attention head, initialized to zero so
        # the feature path is introduced gradually instead of perturbing the
        # baseline at initialization.
        self.value_residual = nn.Parameter(torch.zeros(1, heads, 1, 1))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            [Block(width, config['heads']) for _ in range(config['depth'])]
        )
        # Keep the first block as the source of an optional long-range value
        # shortcut; block 0 itself has no shortcut, while deeper blocks learn
        # independent per-head mixing strengths.
        for block in self.blocks[1:]:
            block.add_value_residual(config['heads'])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.pos(positions)
        first_value = None
        for index, block in enumerate(self.blocks):
            x, value = block(x, first_value if index > 0 else None)
            if index == 0:
                # Reuse the first layer's projected value tensor as a direct
                # attention-value source in all subsequent layers.
                first_value = value
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    model_config = dict(config)
    model_config['width'] = 160
    return GPT(model_config)
