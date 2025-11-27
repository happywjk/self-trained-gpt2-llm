import torch.nn as nn


class FeedForwardBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError("FeedForwardBase.forward must be implemented by subclasses")


class DenseFF(FeedForwardBase):
    def __init__(self, config):
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(config.hiddensize, 4 * config.hiddensize),
            nn.GELU(),
            nn.Linear(4 * config.hiddensize, config.hiddensize),
            nn.Dropout(config.dropout),
        )
        self.net[2].NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.net(x)
