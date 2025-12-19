import torch.nn as nn
import torch.distributed as dist

def master_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
        
def tp_debug_hook(module, inputs, output):
    """用来查看 shard 后的输入/输出/权重的 hook."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    # 多卡时只在 rank 0 停/打印，不然 4 个终端一起输出会爆炸
    if rank != 0:
        return

    print("\n================= TP DEBUG HOOK =================")
    print(f"Module: {module.__class__.__name__}")
    
    # 看输入
    x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
    try:
        print("  input global shape:", getattr(x, "shape", None))
        if hasattr(x, "to_local"):
            print("  input local shape:", x.to_local().shape)
            print("  input placements:", getattr(x, "placements", None))
    except Exception as e:
        print("  input inspect error:", e)

    # 看输出
    try:
        print("  output global shape:", getattr(output, "shape", None))
        if hasattr(output, "to_local"):
            print("  output local shape:", output.to_local().shape)
            print("  output placements:", getattr(output, "placements", None))
    except Exception as e:
        print("  output inspect error:", e)

    # 看权重
    if hasattr(module, "weight"):
        w = module.weight
        print("  weight global shape:", w.shape)
        if hasattr(w, "to_local"):
            print("  weight local shape:", w.to_local().shape)
            print("  weight placements:", getattr(w, "placements", None))

def register_tp_hooks(model):
    """在原始模型上注册你关心的 TP 调试 hook."""
    # 先看看整体结构（可选）
    master_print("===== 模型模块列表（截断版）=====")
    for name, m in list(model.named_modules())[:50]:
        master_print(name, "->", m.__class__.__name__)
    master_print("===== 上面是前 50 个 module，更多可以自己调 =====")

    # 这里我们假设：
    # - 你用的是 MLP 里 parallelize 的 Linear（比如 net.0 / net.2）
    # - 或者你想 debug 所有 Linear
    for name, m in model.named_modules():
        # 例1：所有 Linear 都挂 hook
        if isinstance(m, nn.Linear):
            master_print(f"[HOOK] register on {name}")
            m.register_forward_hook(tp_debug_hook)

        # 例2：只给名字里带 "mlp" 或 "net.0/2" 的挂 hook，可以根据你实际的命名规则改
        # if isinstance(m, nn.Linear) and ("mlp" in name or name.endswith("net.0") or name.endswith("net.2")):
        #     master_print(f"[HOOK] register on {name}")
        #     m.register_forward_hook(tp_debug_hook)

