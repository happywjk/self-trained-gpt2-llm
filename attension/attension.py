import torch.nn as nn
import torch.nn.functional as F


class AttentionBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError("AttentionBase.forward must be implemented by subclasses")


class DenseAttention(AttentionBase):
    def __init__(self, config):
        super().__init__(config)
        self.proj_qkv = nn.Linear(config.hiddensize, 3 * config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.hiddensize)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        assert self.config.hiddensize % self.config.num_heads == 0
        qkv = self.proj_qkv(x)  # B T 3C
        q, k, v = qkv.split(self.config.hiddensize, dim=2)  # B T C
        q = q.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        k = k.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # B nh T hd
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)  # B T C
        out = self.out_proj(attention)  # B T C
        out = self.dropout(out)
        return out
