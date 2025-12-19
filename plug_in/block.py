from typing import Type

import torch.nn as nn

# NOTE: keep imports relative so we always pick up the local implementations
from plug_in.attension import AttentionBase
from plug_in.ffn import DenseFF, FeedForwardBase


class Block(nn.Module):
    def __init__(
        self,
        config,
        attention_impl: Type[AttentionBase],
        ff_impl: Type[FeedForwardBase] = DenseFF,
        device_mesh=None,
    ):
        super().__init__()
        self.attn = attention_impl(config)
        self.ln1 = nn.LayerNorm(config.hiddensize)
        self.ff = ff_impl(config, device_mesh=device_mesh)
        self.ln2 = nn.LayerNorm(config.hiddensize)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
