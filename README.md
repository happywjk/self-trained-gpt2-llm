# self-trained-gpt2-llm – Dev Environment and Distributed Runs

This repo includes a ready-to-use Codespaces/Dev Container for PyTorch (CUDA-enabled) development, training, and inference.

## Quick Start (Codespaces)
- Open the repo in GitHub Codespaces. It will detect `.devcontainer/` and build the image.
- On first create or rebuild, it installs `requirements.txt` and prints PyTorch + CUDA info.
- If your Codespace has a GPU, `CUDA available: True` will be printed.

To rebuild into this environment later:
- In VS Code: Command Palette → “Codespaces: Rebuild Container”.
- Or Dev Containers locally: “Dev Containers: Rebuild Container”.

## Local Dev Container (Docker) – Optional
If developing locally with Docker and a GPU:

```bash
# Build the devcontainer image locally
docker build -f .devcontainer/Dockerfile -t self-trained-gpt2-llm:dev .devcontainer

# Run with your workspace mounted and GPU access
docker run --rm -it --gpus all \
  --shm-size=16g \
  -v "$PWD":/workspaces/self-trained-gpt2-llm \
  -w /workspaces/self-trained-gpt2-llm \
  self-trained-gpt2-llm:dev bash
```

## GPU Notes
- Codespaces GPU must be enabled for your account/repo to get CUDA.
- The container works on CPU-only hosts; CUDA-related calls will return false.
- Shared memory is bumped to 16 GB to avoid DataLoader/NCCL OOM in large runs.

## Distributed Training (single node, multi-GPU)
Use `torchrun` to launch multiple processes—one per GPU. Replace `N` with your GPU count.

```bash
# Example: 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# Optional diagnostics
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN

# Launch training
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --config config.py
```

Tips:
- For CPU-only debugging, add `--nproc_per_node=1` and ensure your code selects `cpu`.
- Common NCCL fallbacks if networking is odd:
  - `export NCCL_IB_DISABLE=1`
  - `export NCCL_P2P_DISABLE=0`

## Distributed Training (multi-node)
If you have multiple machines/containers with GPUs and network access:

```bash
# On all nodes
export MASTER_ADDR=10.0.0.1   # IP of node 0
export MASTER_PORT=29500
export WORLD_SIZE=2           # total nodes

# Node 0
export NODE_RANK=0
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=2 --nproc_per_node=2 train.py --config config.py

# Node 1
export NODE_RANK=1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=2 --nproc_per_node=2 train.py --config config.py
```

- Ensure nodes can reach each other on `MASTER_ADDR:MASTER_PORT`.
- Keep the same codebase and data paths accessible on each node.

## Distributed Inference
Inference can be single GPU or parallelized with the same `torchrun` pattern if you implement sharded/tensor-parallel logic.

```bash
# Single GPU
python infer.py --model-checkpoint path/to/ckpt

# Multi-GPU (if your script supports it)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 infer.py \
  --model-checkpoint path/to/ckpt
```

## Reusing This Environment Next Time
- Commit the `.devcontainer` folder. Codespaces will auto-use it on next open.
- Faster startup: enable Prebuilds in GitHub → Settings → Codespaces → Prebuilds.
- Pin a specific base image tag in `.devcontainer/Dockerfile` for reproducibility.

### Optional: Publish the Dev Image to GHCR
You can build and push the container for reuse elsewhere:

```bash
# Using Dev Containers CLI (install first if needed)
# devcontainer build --workspace-folder . --image-name ghcr.io/<owner>/self-trained-gpt2-llm-dev:latest --push

# Or plain Docker (from repo root)
docker build -f .devcontainer/Dockerfile -t ghcr.io/<owner>/self-trained-gpt2-llm-dev:latest .devcontainer
# Authenticate: echo $GHCR_TOKEN | docker login ghcr.io -u <owner> --password-stdin
docker push ghcr.io/<owner>/self-trained-gpt2-llm-dev:latest
```

Then reference it in `.devcontainer/devcontainer.json` by replacing the `build` section with:

```json
{
  "image": "ghcr.io/<owner>/self-trained-gpt2-llm-dev:latest"
}
```

## Verifying PyTorch
Inside the container:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If `cuda True` and device count > 0, you have GPU access.
