# %%
import torch
import torch.nn as nn
import time
import math
import inspect
import config
import sentencepiece as spm
# %%
from dataloader import dataloaderlite
dataset = dataloaderlite(model_path = "tok_normal_novel_bpe.model")
sp    = spm.SentencePieceProcessor(model_file = "tok_normal_novel_bpe.model")
print("finish dataset")
torch.set_float32_matmul_precision("high")
# %%
def get_lr(step):
    if step < config.warmup_steps:
        lr = config.learning_rate*(step + 1)/config.warmup_steps
        return lr
    elif step >= config.maxstep:
        return 0.1*config.learning_rate
    else:
        decay_ratio = (step - config.warmup_steps)/(config.maxstep - config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return 0.1*config.learning_rate + coeff * (config.learning_rate - 0.1*config.learning_rate)

# %%
def large_batch(batch_size):
    assert batch_size % (config.batchsize*config.blocksize) ==0
    grad_accumulation = int(batch_size/(config.batchsize*config.blocksize))
    return grad_accumulation
    
@torch.no_grad()
def eval_loss(model):
    model.eval()
    out = {}
    for split in ["train","validation"]:
        losses = torch.zeros(100)
        for k in range(100):
            x,y = dataset.next_batch(mode = split)
            target,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def gen(model,prompt):
    model.eval()
    out = {}
    encode_prompt = torch.tensor(sp.encode(prompt)).to(config.device)
    encode_prompt = encode_prompt.unsqueeze(0).repeat(config.batchsize, 1)
    out = model.generate(encode_prompt,max_token = 50)
    text = sp.decode(out.tolist())
    model.train()
    return text

def configure_optimizers(self, weight_decay, learning_rate, device_type):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer
from model import gpt
model = gpt(config)
model.to(config.device)
model = torch.compile(model)
optimizer = configure_optimizers(model,weight_decay=0.1,learning_rate = config.learning_rate,device_type=config.device)
print("finish model")
grad_accum = large_batch(524288)
# grad_accum = large_batch(65536)
step = 0
for epoch in range(2):
    print(f"🚀 Starting epoch {epoch+1}")
    for i in range(config.maxstep//4):
        t0 = time.time()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad()
        for i in range(grad_accum):
            x,y = dataset.next_batch()
            # with torch.autocast(device_type = config.device, dtype = torch.bfloat16):
            logits, loss = model(x,y)
            loss = loss / grad_accum
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        t1 = time.time()
        dt = (t1 -t0)*1000
        tps = (config.blocksize* config.batchsize)/(t1-t0)
        print(f"iteration: {step}, learning rate: {lr}, norm: {norm:.2f}  dt:{dt:.2f}ms")
        if step % 100 ==0:
            out = eval_loss(model)
            print(f"train loss: {out["train"]}, validation loss: {out["validation"]}")
    
        if step % 300 == 0:
            out = gen(model, "我坐在工程学院自习")
            print(out)
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': lr,
            }
            save_path = f"checkpoint.pt"
            torch.save(checkpoint, save_path)
            print(f"✅ Model checkpoint saved to {save_path}")
        step = step + 1

# %%
