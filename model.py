# %% 
import torch
import torch.nn as nn
import config
import torch.nn.functional as F
from typing import Type

from attension.attension import AttentionBase, DenseAttention
from attension.block import Block
from attension.ffn import DenseFF, FeedForwardBase

print(config.__file__)

class gpt(nn.Module):
    def __init__(
        self,
        config,
        device,
        attention_impl: Type[AttentionBase] = DenseAttention,
        ff_impl: Type[FeedForwardBase] = DenseFF,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocabsize,config.hiddensize)
        self.position_embedding = nn.Embedding(config.blocksize,config.hiddensize)
        self.Block = nn.ModuleList([Block(config, attention_impl, ff_impl) for i in range(config.n_layers)])
        self.ln    = nn.LayerNorm(config.hiddensize)
        self.out_proj = nn.Linear(config.hiddensize, config.vocabsize)
        self.token_embedding.weight = self.out_proj.weight
        self.config = config
        self.device = device
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
        pos_emb = self.position_embedding(torch.arange(0,T,dtype=torch.long,device = self.device))
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
