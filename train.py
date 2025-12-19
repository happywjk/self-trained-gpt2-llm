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
from plug_in.attension import LinearAttention, DeltaAttention
from torch.distributed.device_mesh import init_device_mesh
from support.hook import register_tp_hooks
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import _redistribute as dtensor_redistribute
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
def master_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
    
def large_batch(batch_size, world_size):
    assert batch_size % (config.batchsize*config.blocksize*world_size) ==0
    grad_accumulation = int(batch_size/(config.batchsize*config.blocksize*world_size))
    return grad_accumulation
    
@torch.no_grad()
def eval_loss(model,device):
    model.eval()
    out = {}
    for split in ["train","validation"]:
        losses = torch.zeros(100, device=device)
        for k in range(100):
            x,y = dataset.next_batch(device,mode = split)
            target,loss = model(x,y)
            losses[k] = loss.item()
        loss_accum = losses.mean()
        if ddp:
            dist.all_reduce(loss_accum, op =dist.ReduceOp.AVG)
        out[split] = loss_accum
    model.train()
    return out

@torch.no_grad()
def gen(model,prompt):
    model.eval()
    out = {}
    encode_prompt = torch.tensor(sp.encode(prompt)).to(device)
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
    master_print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    master_print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    master_print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

def setup_distributed():
    """Initialize DDP state from environment and return common metadata."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master_process = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        master_process = True
    return ddp, rank, local_rank, world_size, device, master_process

def cleanup_distributed(ddp):
    """Tear down DDP process group if it was initialized."""
    if ddp and dist.is_initialized():
        destroy_process_group()

def build_device_mesh(device_type, world_size):
    """Create a 1D device mesh spanning all ranks."""
    mesh = init_device_mesh(device_type, (world_size,))
    return mesh
# rank environment value is being set automatically when you initial script using torchrun --nproc_per_node
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_distributed()
device_mesh = build_device_mesh(torch.device(device).type, ddp_world_size)
if master_process:
    master_print(f"device mesh created: {device_mesh}")

dataset = dataloaderlite(model_path = "tok_normal_novel_bpe.model",rank=ddp_local_rank, num_process=ddp_world_size)
sp    = spm.SentencePieceProcessor(model_file = "tok_normal_novel_bpe.model")
master_print("finish dataset")
torch.set_float32_matmul_precision("high")

from model import gpt

model = gpt(config, device=device, device_mesh=device_mesh)
model.to(device)
# model = torch.compile(model)  关掉compile模式以进行debug
master_print(model)
raw_model = model
register_tp_hooks(raw_model)
optimizer = configure_optimizers(raw_model,weight_decay=0.1,learning_rate = config.learning_rate,device_type=device)
master_print("finish model")
grad_accum = large_batch(524288, ddp_world_size)
master_print(f"num of accumulation step is {grad_accum}")
step = 0
for epoch in range(2):
    master_print(f"🚀 Starting epoch {epoch+1}")
    for i in range(config.maxstep//4):
        t0 = time.time()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad()
        for i in range(grad_accum):
            x,y = dataset.next_batch(device)
            # with torch.autocast(device_type = config.device, dtype = torch.bfloat16):
            logits, loss = model(x,y)
            loss = loss / grad_accum
            loss.backward()
        # 将 DTensor grad 先聚合为 Replicate，避免 foreach_norm 混合 Tensor/DTensor 报错
        for p in model.parameters():
            if p.grad is not None and hasattr(p.grad, "placements"):
                p.grad = dtensor_redistribute(p.grad, device_mesh, [Replicate()])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        t1 = time.time()
        dt = (t1 -t0)*1000
        tps = (config.blocksize* config.batchsize*ddp_world_size)/(t1-t0)
        master_print(f"iteration: {step}, learning rate: {lr}, norm: {norm:.2f}  tps:{tps:.2f} tokens/second")
        if step % 5 ==0:
            out = eval_loss(model,device)
            master_print(f"train loss: {out['train']}, validation loss: {out['validation']}")
    
        if step % 300 == 0 and master_process:
            out = gen(raw_model, "我坐在工程学院自习")
            master_print(out)
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': lr,
            }
            save_path = f"checkpoint.pt"
            torch.save(checkpoint, save_path)
            master_print(f"✅ Model checkpoint saved to {save_path}")
        step = step + 1

cleanup_distributed(ddp)
# %%
