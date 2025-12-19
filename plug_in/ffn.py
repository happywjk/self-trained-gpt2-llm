import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


class FeedForwardBase(nn.Module):
    def __init__(self, config, device_mesh=None):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh

    def forward(self, x):
        raise NotImplementedError("FeedForwardBase.forward must be implemented by subclasses")


class DenseFF(FeedForwardBase):
    def __init__(self, config, device_mesh=None):
        super().__init__(config, device_mesh=device_mesh)
        self.net = nn.Sequential(
            nn.Linear(config.hiddensize, 4 * config.hiddensize),
            nn.GELU(),
            nn.Linear(4 * config.hiddensize, config.hiddensize),
            nn.Dropout(config.dropout),
        )
        self.net[2].NANOGPT_SCALE_INIT = 1
        if device_mesh is not None:
            # Parallelize MLP: first Linear col-wise, second Linear row-wise.
            self.net = parallelize_module(
                self.net,
                device_mesh,
                {
                    "0": ColwiseParallel(),
                    "2": RowwiseParallel(),
                },
            )

    def forward(self, x):
        return self.net(x)
