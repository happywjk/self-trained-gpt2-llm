# %% 
import torch
import torch.nn as nn
import config
import torch.nn.functional as F
print(config.__file__)
class attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_qkv = nn.Linear(config.hiddensize, 3*config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize,config.hiddensize)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        self.dropout  = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x):
        B,T,C = x.shape
        assert self.config.hiddensize % self.config.num_heads == 0
        qkv = self.proj_qkv(x)  # B T 3C
        q,k,v = qkv.split(self.config.hiddensize,dim = 2) # B T C
        q = q.view(B,T,self.config.num_heads, self.config.hiddensize//self.config.num_heads).transpose(1,2) # B nh T hd
        k = k.view(B,T,self.config.num_heads, self.config.hiddensize//self.config.num_heads).transpose(1,2) # B nh T hd
        v = v.view(B,T,self.config.num_heads, self.config.hiddensize//self.config.num_heads).transpose(1,2) # B nh T hd
        attention = F.scaled_dot_product_attention(q,k,v,is_causal=True) # B nh T hd
        attention = attention.transpose(1,2).contiguous().view(B,T,C) # B T C
        out = self.out_proj(attention) # B T C
        out = self.dropout(out)
        return out


class mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hiddensize,4*config.hiddensize),
            nn.GELU(),
            nn.Linear(4*config.hiddensize,config.hiddensize),
            nn.Dropout(config.dropout)
        )
        self.net[2].NANOGPT_SCALE_INIT = 1
    
    def forward(self,x):
        x = self.net(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.atten = attention(config)
        self.ln1 = nn.LayerNorm(config.hiddensize)
        self.mlp = mlp(config)
        self.ln2 = nn.LayerNorm(config.hiddensize)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = x + self.atten(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class gpt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocabsize,config.hiddensize)
        self.position_embedding = nn.Embedding(config.blocksize,config.hiddensize)
        self.Block = nn.ModuleList([Block(config) for i in range(config.n_layers)])
        self.ln    = nn.LayerNorm(config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.vocabsize)
        self.token_embedding.weight = self.out_proj.weight
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,x, target = None):
        B,T = x.shape
        # print(x)
        t_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(0,T,dtype=torch.long,device = config.device))
        x = t_emb + pos_emb
        for block in self.Block:
            x = block(x)
        out = self.ln(x)
        logits = self.out_proj(out)
        if target == None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(B*T,-1),target.view(-1))
            return logits, loss

    def generate(self, prompt,max_token = 200):
        for i in range(max_token):
            prompt = prompt[:,-self.config.blocksize:]
            # print(prompt.shape)
            logits = self(prompt)   # B T C
            logits = logits[:,-1,:] # B 1 C
            prob   = F.softmax(logits,dim = 1)
            out    = torch.multinomial(prob,num_samples=1) # B 1
            prompt = torch.concat([prompt,out], dim = 1)
        return prompt


# %%


# %%
# tensor = torch.randint(low=0, high=10, size=(4, 8))
# out = model.generate(tensor)

# %%
