import os
import sys
import uuid
import time
import glob
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from muon import Muon
from model import BlockGPT, BlockGPTConfig


code = "\n".join([
    open(__file__).read(),
    open("model.py").read()
])

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# -----------------------------------------------------------------------------
# Parameters


@dataclass
class Hyperparameters:
    train_files = "data/finewebedu10B/finewebedu_train_*.bin"
    val_files = "data/finewebedu10B/finewebedu_val_*.bin"
    val_tokens = 10_485_760
    train_seq_len = 16 * 1024
    val_seq_len = 64 * 1024
    grad_accum_steps_per_device = (8 // int(os.environ["WORLD_SIZE"]))
    cooldown_frac = 0.4
    vocab_size = 50_257
    val_loss_every = 125
    save_checkpoint = False


# -----------------------------------------------------------------------------
# Simple Distributed Data Loader


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=True
        )
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    rank: int,
    world_size: int,
):
    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_bs = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0

    while True:
        if pos + batch_size >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_bs:][:local_bs]
        inputs = buf.to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs


# -----------------------------------------------------------------------------
# Main


def evaluate(model, loader, steps):
    """Run model in eval mode over `steps` batches from `loader`."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(steps):
            x = next(loader)
            total += model(x)
    return total / steps


def train_step(model, loader, step, optimizers, optimizer2, accum_steps):
    # forward/backward accumulation
    for _ in range(accum_steps):
        x = next(loader)
        loss = model(x)
        loss.backward()

    # gradient all‐reduce across ranks
    for _, p in model.named_parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    # adjust learning rates
    lr = get_lr(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr

    # Muon momentum warmup
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1.0)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95

    # step and clear
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)


args = Hyperparameters()

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = rank == 0

logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)


def print0(s: str, console: bool = False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)


# dump the training script
print0(code)
print0("=" * 100)
print0(f"Python {sys.version}")
print0(f"PyTorch {torch.version.__version__}")
print0(os.popen("nvidia-smi").read())
print0("=" * 100)


model = BlockGPT(BlockGPTConfig()).cuda()

for m in model.modules():
    if isinstance(m, torch.nn.Embedding):
        m.bfloat16()

for p in model.parameters():
    dist.broadcast(p.detach(), 0)


hidden_matrix_params = [
    p
    for n, p in model.blocks.named_parameters()
    if p.ndim >= 2 and "embed" not in n
]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

adam_params = [
    dict(params=head_params, lr=0.0022),
    dict(params=embed_params, lr=0.06),
    dict(params=scalar_params, lr=0.004),
]
optimizer1 = torch.optim.Adam(
    adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True
)
optimizer2 = Muon(
    hidden_matrix_params,
    lr=0.05,
    momentum=0.95,
)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


def get_lr(step: int) -> float:
    x = step / args.num_iterations
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    w = (1 - x) / args.cooldown_frac
    return w * 1.0 + (1 - w) * 0.1


model = torch.compile(model, dynamic=False)

train_loader = distributed_data_generator(
    args.train_files,
    world_size * args.train_seq_len,
    rank,
    world_size,
)
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()

for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations

    # Validation
    if last_step or (
        args.val_loss_every > 0 and step % args.val_loss_every == 0
    ):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        # validation via evaluate()
        val_batch = world_size * args.val_seq_len
        assert args.val_tokens % val_batch == 0
        val_steps = args.val_tokens // val_batch
        val_loader = distributed_data_generator(args.val_files, val_batch, rank, world_size)
        val_loss = evaluate(model, val_loader, val_steps)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        num_toks = step * args.grad_accum_steps_per_device \
            * args.train_seq_len * world_size
        print0(
            f"step:{step}/{args.num_iterations} "
            f"val_loss:{val_loss:.4f} "
            f"train_time:{training_time_ms:.0f}ms "
            f"step_avg:{training_time_ms / max(step, 1):.2f}ms "
            f"tokens:{num_toks / 1e6:.2f}M",
            console=True,
        )
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if last_step and master_process and args.save_checkpoint:
            ckpt = dict(
                step=step,
                code=code,
                model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
            )
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(ckpt, f"logs/{run_id}/state_step{step:06d}.pt")

        if last_step:
            break

    # Training
    train_step(model, train_loader, step, optimizers, optimizer2, args.grad_accum_steps_per_device)

    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{args.num_iterations} "
        f"train_time:{approx_time:.0f}ms "
        f"step_avg:{approx_time / (step + 1):.2f}ms",
        console=True,
    )

print0(
    f"peak memory allocated: "
    f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: "
    f"{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    console=True,
)
dist.destroy_process_group()
