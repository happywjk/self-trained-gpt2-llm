"""
Minimal device mesh tutorial for this project.
- Run directly with: python device_mesh_tutorial.py
- Run multi-process: torchrun --nproc_per_node=2 device_mesh_tutorial.py
- All explanations are in comments so you can read inline while running.
"""

import torch
import os
import sys
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

try:
    # Optional: only used to show how a tensor can be sharded on the mesh.
    from torch.distributed._tensor import distribute_tensor, Shard, Replicate
except Exception:
    distribute_tensor = None
    Shard = None
    Replicate = None


def master_print(*args, **kwargs) -> None:
    """Print only on local rank 0."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args, **kwargs)


def create_mesh(device_type: str, override_world_size: int | None = None) -> "DeviceMesh":
    """
    Build a 1D mesh spanning all visible devices.
    - On CPU we default to size 1.
    - On CUDA we size it to the number of GPUs.
    - override_world_size lets us force a single-rank mesh for debugging.
    """
    world_size = override_world_size
    if world_size is None:
        world_size = torch.cuda.device_count() if device_type == "cuda" else 1
    mesh = init_device_mesh(device_type, (world_size,))
    return mesh


def maybe_init_process_group(device_type: str, world_size: int) -> bool:
    """
    Initialize process group if torchrun launched multiple ranks.
    Needed because distribute_tensor uses collectives and will hang without a process group.
    Returns True if a group was initialized.
    """
    if world_size == 1 or dist.is_initialized():
        return False
    backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return True


def maybe_cleanup_process_group(did_init: bool) -> None:
    """Destroy process group if we created it here."""
    if did_init and dist.is_initialized():
        dist.destroy_process_group()


def describe_mesh(mesh: "DeviceMesh") -> None:
    """
    Print the key attributes so you can see what was created.
    Uses attribute fallbacks because APIs differ slightly across PyTorch versions.
    """
    master_print("mesh repr:", mesh)
    master_print("mesh shape:", mesh.shape)
    # Global mesh: try public attr first, then private fallback.
    # Avoid chaining with `or` because a tensor does not have a boolean value.
    global_mesh = getattr(mesh, "mesh", None)
    if global_mesh is None:
        global_mesh = getattr(mesh, "_mesh", None)
    if global_mesh is not None:
        master_print("global mesh ranks:\n", global_mesh)
    else:
        master_print("global mesh ranks: (not exposed in this torch version)")
    # Local mesh: try method, then attributes.
    if hasattr(mesh, "get_local_mesh"):
        local_mesh = mesh.get_local_mesh()
    else:
        # Same pattern to avoid boolean evaluation on tensors.
        local_mesh = getattr(mesh, "local_mesh", None)
        if local_mesh is None:
            local_mesh = getattr(mesh, "_local_mesh", None)
    if local_mesh is not None:
        master_print("local mesh ranks:", local_mesh)
    else:
        master_print("local mesh ranks: (not exposed in this torch version)")


def demo_shard(mesh: "DeviceMesh", device_type: str) -> None:
    """
    Optional demonstration of sharding a tensor across the mesh.
    Requires torch.distributed._tensor to be available (PyTorch 2.1+).
    Shows which slice each rank would hold; safe to run even in single process.
    """
    if distribute_tensor is None or Shard is None or Replicate is None:
        master_print("sharding demo skipped (torch.distributed._tensor not available)")
        return

    # Simple 1D tensor so the shard split is easy to see.
    x = torch.arange(16, device=device_type).reshape(2,8)
    master_print("shape of x",x)
    
    # Handle both 1D and 2D meshes
    if mesh.ndim == 1:
        # For 1D mesh: just shard on dimension 0
        x_mesh = distribute_tensor(x, mesh, placements=[Shard(0)])
        # Try to report local rank if API exists
        local_rank_fn = getattr(mesh, "get_local_rank", None)
        rank_msg = f"rank {local_rank_fn()}" if callable(local_rank_fn) else "local shard"
    else:
        # For 2D mesh: shard tensor dim 0 on mesh dim 0, tensor dim 1 on mesh dim 1
        x_mesh = distribute_tensor(x, mesh, placements=[Shard(0), Shard(1)])
        # For 2D mesh, get_local_rank needs mesh_dim parameter
        local_rank_fn = getattr(mesh, "get_local_rank", None)
        if callable(local_rank_fn):
            # Get rank for both dimensions
            rank_0 = local_rank_fn(mesh_dim=0)
            rank_1 = local_rank_fn(mesh_dim=1)
            rank_msg = f"rank ({rank_0}, {rank_1})"
        else:
            rank_msg = "local shard"
    
    print(f"{rank_msg}: shard = {x_mesh}")


def project_usage_note(device_type: str, world_size: int) -> None:
    """
    Connect back to train.py so you know how the mesh is used in this repo.
    - train.py builds the same 1D mesh via build_device_mesh(device_type, world_size)
    - The mesh spans all ranks created by torchrun (WORLD_SIZE)
    - Current training mainly uses DDP; the mesh is ready for tensor-parallel APIs if needed
    """
    master_print("\nProject usage:")
    master_print(f"- train.py builds a mesh with device_type={device_type}, world_size={world_size}")
    master_print("- It is printed at startup to confirm creation.")
    master_print("- You can reuse the same mesh for distributed tensor APIs (e.g., distribute_tensor).")


if __name__ == "__main__":
    # Pick device automatically: CUDA if present, otherwise CPU.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    debug_single = os.environ.get("DEBUG_SINGLE_RANK", "0") == "1"

    # If debugging single rank, bail out early on non-zero ranks so collectives won't hang.
    local_rank_env = int(os.environ.get("LOCAL_RANK", 0))
    if debug_single and local_rank_env != 0:
        master_print(f"[rank {local_rank_env}] skipping debug path (DEBUG_SINGLE_RANK=1)")
        sys.exit(0)

    # Choose world size; force 1 when debugging single rank.
    world_size = 1 if debug_single else (torch.cuda.device_count() if device_type == "cuda" else 1)
    master_print(world_size)

    # Create and describe the mesh. In multi-process runs, each rank sees its local view.
    # mesh = create_mesh(device_type, override_world_size=world_size)
    mesh_1d = init_device_mesh("cuda", mesh_shape=(4,))
    mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    # master_print("mesh repr 1d:", mesh_1d)
    # master_print("mesh shape 1d:", mesh_1d.shape)
    # master_print("mesh repr 2d:", mesh_2d)
    # master_print("mesh shape 2d:", mesh_2d.shape)
    # master_print(dir(mesh_2d))
    # describe_mesh(mesh_2d)

    # Init process group when running with torchrun so distribute_tensor has collectives available.
    # did_init_pg = maybe_init_process_group(device_type, world_size)

    # describe_mesh(mesh)d

    # # Show how a tensor would be sharded across the mesh (optional).
    demo_shard(mesh_2d, device_type)

    # # Remind how this ties back to the training script in this repo.
    # project_usage_note(device_type, world_size)

    # maybe_cleanup_process_group(did_init_pg)
