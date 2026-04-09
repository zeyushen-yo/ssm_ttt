"""
Training script for all three models: Transformer-SWA, Vanilla-SSM, SSM+TTT.

Implements spec sections 6 (Phase 1-3 training) and 9 (diagnostics).

Usage:
  python train.py --model_type ssm_ttt --config configs/tiny_pilot.yaml
"""

import argparse
import json
import math
import os
import signal
import time
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

os.environ["WANDB_MODE"] = "offline"
try:
    import wandb
except ImportError:
    wandb = None

from data.dataloader import get_tokenizer, create_train_dataloader
from models.ssm_ttt_model import create_vanilla_ssm, create_ssm_ttt, SSMTTTModel
from models.swa_transformer import SWATransformerLM
from models.ttt_wrapper import TTTMamba2Block


def get_model(config: dict, device="cuda", dtype=torch.bfloat16):
    """Create model based on config."""
    model_type = config["model_type"]
    model_args = config["model_args"]
    factory = {"device": device, "dtype": dtype}

    if model_type == "transformer_swa":
        model = SWATransformerLM(**model_args, **factory)
    elif model_type == "vanilla_ssm":
        model = create_vanilla_ssm(**model_args, **factory)
    elif model_type == "ssm_ttt":
        model = create_ssm_ttt(**model_args, **factory)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def get_optimizer(model, config: dict):
    """Set up AdamW optimizer with 4 parameter groups per spec v3."""
    lr = config.get("lr", 5e-4)
    weight_decay = config.get("weight_decay", 0.1)
    betas = tuple(config.get("betas", [0.9, 0.95]))
    ttt_target_lr_mult = config.get("ttt_target_lr_mult", 3.0)

    group_a = []  # backbone decayed (matrix weights)
    group_b = []  # backbone no decay (biases, norms, embeddings, scalars)
    group_c = []  # TTT target-builder params (mix_coeffs, W_tgt)
    group_d = []  # TTT scalar control params (log_inner_lr, log_decay_logit, fast_gate)

    ttt_target_names = {"mix_coeffs", "W_tgt"}
    ttt_scalar_names = {"log_inner_lr", "log_decay_logit", "fast_gate"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf_name = name.split(".")[-1]
        if leaf_name in ttt_target_names:
            group_c.append(param)
        elif leaf_name in ttt_scalar_names:
            group_d.append(param)
        elif param.ndim <= 1 or "bias" in name or "norm" in name or "embedding" in name:
            group_b.append(param)
        else:
            group_a.append(param)

    param_groups = [
        {"params": group_a, "weight_decay": weight_decay, "lr": lr},
        {"params": group_b, "weight_decay": 0.0, "lr": lr},
        {"params": group_c, "weight_decay": 0.0, "lr": lr * ttt_target_lr_mult},
        {"params": group_d, "weight_decay": 0.0, "lr": lr},
    ]

    n_a = sum(p.numel() for p in group_a)
    n_b = sum(p.numel() for p in group_b)
    n_c = sum(p.numel() for p in group_c)
    n_d = sum(p.numel() for p in group_d)
    print(f"Optimizer groups: A(backbone-decay)={n_a:,} B(backbone-nodecay)={n_b:,} "
          f"C(ttt-target, lr={lr*ttt_target_lr_mult:.1e})={n_c:,} D(ttt-scalar)={n_d:,}")

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    return optimizer


def get_lr_scheduler(optimizer, config: dict, total_steps: int):
    """Cosine schedule with warmup."""
    warmup_steps = config.get("warmup_steps", 1024)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collect_ttt_diagnostics(model):
    """Collect TTT fast-weight health diagnostics (spec 9.1)."""
    if hasattr(model, 'get_ttt_diagnostics'):
        return model.get_ttt_diagnostics()
    return {}


def train(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Setup
    os.makedirs(config["output_dir"], exist_ok=True)
    log_file = os.path.join(config["output_dir"], "train_log.jsonl")

    # WandB (offline)
    if wandb is not None:
        wandb.init(
            project="ssm_ttt",
            name=os.path.basename(config["output_dir"]),
            config=config,
            dir=config["output_dir"],
            mode="offline",
        )

    # Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    # Inject vocab_size into model args
    config["model_args"]["vocab_size"] = vocab_size

    # Model
    model = get_model(config, device=device, dtype=dtype)
    param_info = model.count_parameters() if hasattr(model, 'count_parameters') else {
        "total": sum(p.numel() for p in model.parameters())
    }
    print(f"Model: {config['model_type']}")
    print(f"Parameters: {param_info}")
    for k, v in param_info.items():
        print(f"  {k}: {v:,}")

    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Data
    seq_len = config.get("seq_len", 2048)
    batch_size = config.get("batch_size", 8)
    dataloader = create_train_dataloader(
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=config.get("num_workers", 4),
        data_dir=config.get("data_dir", None),
        dataset_name=config.get("dataset_name", "monology/pile-uncopyrighted"),
        seed=config.get("seed", 42),
        pack_documents=config.get("pack_documents", False),
        boundary_aware=config.get("boundary_aware", False),
    )

    # Optimizer
    optimizer = get_optimizer(model, config)
    total_tokens = int(config.get("total_tokens", 1_000_000_000))
    tokens_per_step = batch_size * seq_len
    total_steps = total_tokens // tokens_per_step
    print(f"Total steps: {total_steps}, tokens/step: {tokens_per_step}")

    scheduler = get_lr_scheduler(optimizer, config, total_steps)

    # Gradient clipping
    grad_clip = config.get("grad_clip", 1.0)

    # Mixed precision (bfloat16 doesn't need GradScaler)
    use_amp = config.get("use_amp", True) and device == "cuda"

    # Resume from checkpoint if available
    start_step = 0
    tokens_seen = 0
    resume_ckpt = config.get("resume_checkpoint", None)
    if resume_ckpt is None:
        output_dir = config["output_dir"]
        ckpt_files = sorted(
            [f for f in os.listdir(output_dir)
             if f.startswith("checkpoint_") and f.endswith(".pt") and f != "checkpoint_final.pt"],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        ) if os.path.exists(output_dir) else []
        if ckpt_files:
            resume_ckpt = os.path.join(output_dir, ckpt_files[-1])

    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        tokens_seen = ckpt.get("tokens_seen", start_step * tokens_per_step)
        print(f"Resumed at step {start_step}, tokens_seen: {tokens_seen:,}")

    # Training loop
    model.train()
    data_iter = iter(dataloader)

    log_interval = config.get("log_interval", 50)
    save_interval = config.get("save_interval", 5000)
    eval_interval = config.get("eval_interval", 2000)

    preempt_requested = [False]
    def _sigterm_handler(signum, frame):
        print(f"\nSIGTERM received — will save checkpoint after current step")
        preempt_requested[0] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGUSR1, _sigterm_handler)

    running_loss = 0.0
    start_time = time.time()
    step_start_time = time.time()

    for step in range(start_step + 1, total_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        segment_ids = batch.get("segment_ids", None)
        if segment_ids is not None:
            segment_ids = segment_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            output = model(input_ids, labels=labels, segment_ids=segment_ids)
            loss = output.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss at step {step}, skipping")
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        tokens_seen += tokens_per_step

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - step_start_time
            tps = (log_interval * tokens_per_step) / elapsed
            lr = scheduler.get_last_lr()[0]

            log_entry = {
                "step": step,
                "loss": avg_loss,
                "ppl": ppl,
                "lr": lr,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "tokens_seen": tokens_seen,
                "tokens_per_sec": tps,
                "elapsed_sec": time.time() - start_time,
            }

            # TTT diagnostics (spec 9.1)
            ttt_diag = collect_ttt_diagnostics(model)
            log_entry.update(ttt_diag)

            print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                  f"LR: {lr:.2e} | Grad: {log_entry['grad_norm']:.4f} | "
                  f"Tok/s: {tps:.0f}")
            if ttt_diag:
                diag_str = " | ".join(f"{k}: {v:.6f}" for k, v in ttt_diag.items())
                print(f"  TTT: {diag_str}")

            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if wandb is not None:
                wandb.log(log_entry, step=step)

            running_loss = 0.0
            step_start_time = time.time()

        if step % save_interval == 0 or step == total_steps:
            save_path = os.path.join(config["output_dir"], f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "tokens_seen": tokens_seen,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

        if preempt_requested[0]:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            save_path = os.path.join(config["output_dir"], f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "tokens_seen": tokens_seen,
            }, save_path)
            print(f"Preemption checkpoint saved to {save_path} at step {step}")
            sys.exit(0)

    # Final save
    save_path = os.path.join(config["output_dir"], "checkpoint_final.pt")
    torch.save({
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "tokens_seen": tokens_seen,
    }, save_path)
    print(f"Training complete. Final checkpoint: {save_path}")

    if wandb is not None:
        wandb.finish()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSM/Transformer models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)
