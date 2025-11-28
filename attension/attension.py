import torch
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

class DenseAttention_as_tutorial(AttentionBase):
    def __init__(self, config):
        super().__init__(config)
        self.proj_qkv = nn.Linear(config.hiddensize, 3 * config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.hiddensize)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        assert self.config.hiddensize % self.config.num_heads == 0

        # 1) 线性层得到 qkv
        qkv = self.proj_qkv(x)              # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)       # 各自 [B, T, C]

        # 2) 拆成多头
        nh = self.config.num_heads
        hd = C // nh
        q = q.view(B, T, nh, hd).transpose(1, 2)   # [B, nh, T, hd]
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)

        # -------- FlashAttention 核心部分（纯 PyTorch）--------

        block = 128
        out = torch.zeros_like(q)                 # [B, nh, T, hd]
        m = torch.full((B, nh, T, 1), -1e9, device=x.device)
        l = torch.zeros((B, nh, T, 1), device=x.device)

        for ks in range(0, T, block):
            ke = min(ks + block, T)

            k_blk = k[:, :, ks:ke, :]             # [B, nh, BK, hd]
            v_blk = v[:, :, ks:ke, :]             # [B, nh, BK, hd]

            # 1) 局部 q @ kᵀ，只算 T × BK 的一小块
            score = torch.matmul(q, k_blk.transpose(-2, -1))  # [B, nh, T, BK]

            # 2) causal mask（只遮住未来的 token）
            q_idx = torch.arange(T, device=x.device).view(1, 1, T, 1)
            k_idx = torch.arange(ks, ke, device=x.device).view(1, 1, 1, -1)
            score = score.masked_fill(q_idx < k_idx, -1e9)

            # 3) block 内最大值（稳定 softmax 用）
            blk_max = score.max(-1, keepdim=True).values
            new_m = torch.maximum(m, blk_max)

            # 4) 旧的 softmax 状态重缩放
            l = l * torch.exp(m - new_m)
            out = out * torch.exp(m - new_m)

            # 5) 当前块的 softmax 部分
            exp_score = torch.exp(score - new_m)
            l = l + exp_score.sum(-1, keepdim=True)

            # 6) 累加 v 的加权和
            out = out + torch.matmul(exp_score, v_blk)

            m = new_m

        # 7) 除以最终 softmax 分母
        attention = out / l                    # [B, nh, T, hd]

        # -------- 恢复到原始流程 --------

        attention = attention.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attention)
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


class GatedDeltaAttention(AttentionBase):
    def __init__(self, config, threshold: float = 1e-6):
        super().__init__(config)
        self.proj_qkv = nn.Linear(config.hiddensize, 3 * config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.hiddensize)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)
        self.threshold = threshold
        self.prev_kv_state = None
        # Gate produces a scalar per token to blend previous and new KV.
        self.gate_proj = nn.Linear(config.hiddensize, 1)

    def forward(self, x):
        B, T, C = x.shape
        assert self.config.hiddensize % self.config.num_heads == 0
        qkv = self.proj_qkv(x)  # B T 3C
        q, k, v = qkv.split(self.config.hiddensize, dim=2)  # B T C
        q = q.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        k = k.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, self.config.hiddensize // self.config.num_heads).transpose(1, 2)

        # Reset state if batch size changes.
        if self.prev_kv_state is None or self.prev_kv_state[0].shape[0] != B:
            prev_k, prev_v = None, None
        else:
            prev_k, prev_v = self.prev_kv_state

        outputs = []
        for t in range(T):
            k_t = k[:, :, t, :]  # B nh hd
            v_t = v[:, :, t, :]  # B nh hd

            if prev_k is None:
                new_k, new_v = k_t, v_t
            else:
                delta = (k_t - prev_k).pow(2).sum(dim=-1)  # B nh
                mask = delta > self.threshold
                mask_exp = mask.unsqueeze(-1)  # B nh 1
                # Gate computed from q_t, shared across heads.
                gate = torch.sigmoid(self.gate_proj(q[:, :, t, :]))  # B nh 1
                blended_k = gate * k_t + (1 - gate) * prev_k
                blended_v = gate * v_t + (1 - gate) * prev_v
                new_k = torch.where(mask_exp, blended_k, prev_k)
                new_v = torch.where(mask_exp, blended_v, prev_v)

            attn_t = F.scaled_dot_product_attention(
                q[:, :, t, :].unsqueeze(2),  # B nh 1 hd
                new_k.unsqueeze(2),  # B nh 1 hd
                new_v.unsqueeze(2),  # B nh 1 hd
                is_causal=False,
            ).squeeze(2)  # B nh hd
            outputs.append(attn_t)

            prev_k, prev_v = new_k, new_v

        self.prev_kv_state = (prev_k.detach(), prev_v.detach())

        attention = torch.stack(outputs, dim=2)  # B nh T hd
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)  # B T C
        out = self.out_proj(attention)  # B T C
        out = self.dropout(out)
        return out
