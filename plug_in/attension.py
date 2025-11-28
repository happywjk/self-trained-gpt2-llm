import torch.nn as nn
import torch.nn.functional as F
import torch

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
        # print("using dense attension now!")
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

class LinearAttention(AttentionBase):
    def __init__(self, config, eps: float = 1e-4):
        super().__init__(config)
        self.proj_qkv = nn.Linear(config.hiddensize, 3 * config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.hiddensize)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)
        self.eps = eps

    def _phi(self, x):
        # Simple positive feature map to avoid negative values.
        return F.relu(x) + self.eps

    def forward(self, x):
        B, T, C = x.shape
        assert self.config.hiddensize % self.config.num_heads == 0
        qkv = self.proj_qkv(x)  # B T 3C
        q, k, v = qkv.split(self.config.hiddensize, dim=2)  # B T C
        q = q.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        k = k.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)

        phi_q = self._phi(q)
        phi_k = self._phi(k)

        # Compute linear attention approximation: phi(q) @ (phi(k)^T v)
        kv = torch.matmul(phi_k.transpose(-2, -1), v)  # B nh hd hd
        attention = torch.matmul(phi_q, kv)  # B nh T hd

        attention = attention.transpose(1, 2).contiguous().view(B, T, C)  # B T C
        out = self.out_proj(attention)  # B T C
        out = self.dropout(out)
        return out

class DeltaAttention(AttentionBase):
    def __init__(self, config, threshold: float = 1e-6):
        super().__init__(config)
        self.proj_qkv = nn.Linear(config.hiddensize, 3 * config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.hiddensize)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)
        self.threshold = threshold

    def forward(self, x):
        B, T, C = x.shape
        assert self.config.hiddensize % self.config.num_heads == 0
        qkv = self.proj_qkv(x)  # B T 3C
        q, k, v = qkv.split(self.config.hiddensize, dim=2)  # B T C
        q = q.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        k = k.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)

        effective_k = torch.empty_like(k)
        effective_v = torch.empty_like(v)
        eff_k = [k[:, :, 0]]
        eff_v = [v[:, :, 0]]
        prev_k, prev_v = eff_k[0], eff_v[0]
        for t in range(1, T):
            k_t, v_t = k[:, :, t], v[:, :, t]
            delta = (k_t - prev_k).pow(2).sum(dim=-1, keepdim=True)
            mask = delta > self.threshold
            prev_k = torch.where(mask, k_t, prev_k)
            prev_v = torch.where(mask, v_t, prev_v)
            eff_k.append(prev_k)
            eff_v.append(prev_v)
        effective_k = torch.stack(eff_k, dim=2)
        effective_v = torch.stack(eff_v, dim=2)

        attention = F.scaled_dot_product_attention(q, effective_k, effective_v, is_causal=True)  # B nh T hd
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)  # B T C
        out = self.out_proj(attention)  # B T C
        out = self.dropout(out)
        return out