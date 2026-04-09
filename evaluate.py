"""
Figure-2-style sliding-window perplexity evaluation — v2.

Spec Section 7: For each validation document:
1. Fix a scored suffix of the last 2048 tokens
2. For each context length L in {2k, 4k, 8k, 16k, 32k}:
   - Provide only the last L tokens to the model
   - Compute NLL/perplexity only on the fixed final 2048-token suffix
3. Average over all validation documents

v2 additions:
- Per-layer deltaW_rel diagnostics for TTT models at each context length
- Support for larger validation sets (>= 50 long documents)
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from data.dataloader import get_tokenizer
from models.ssm_ttt_model import create_vanilla_ssm, create_ssm_ttt, SSMTTTModel
from models.swa_transformer import SWATransformerLM
from models.ttt_wrapper import TTTMamba2Block


CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


def load_val_documents(data_dir, min_len=2048):
    """Load validation documents from pre-tokenized data."""
    val_path = os.path.join(data_dir, "val.bin")
    offsets_path = os.path.join(data_dir, "val_offsets.npy")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    data = np.memmap(val_path, dtype=np.uint16, mode='r')
    offsets = np.load(offsets_path)
    num_docs = len(offsets) - 1

    docs = []
    for i in range(num_docs):
        start = offsets[i]
        end = offsets[i + 1]
        doc = data[start:end].astype(np.int64)
        if len(doc) >= min_len:
            docs.append(torch.from_numpy(doc.copy()))

    print(f"Loaded {len(docs)} validation documents (>={min_len} tokens)")
    return docs


def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """Load a model from checkpoint, reconstructing the correct architecture."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt["config"]
    model_type = config["model_type"]
    model_args = config["model_args"]

    dtype = torch.bfloat16

    if model_type == "transformer_swa":
        model = SWATransformerLM(**model_args, device=device, dtype=dtype)
    elif model_type == "ssm_ttt":
        model = create_ssm_ttt(**model_args, device=device, dtype=dtype)
    elif model_type == "vanilla_ssm":
        model = create_vanilla_ssm(**model_args, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def collect_ttt_eval_diagnostics(model):
    """Collect per-layer TTT diagnostics during evaluation (spec v3 diagnostics)."""
    diagnostics = {}
    if not isinstance(model, SSMTTTModel):
        return diagnostics
    for i, layer in enumerate(model.layers):
        if isinstance(layer, TTTMamba2Block):
            wrapper = layer.ttt_wrapper
            diag = wrapper._last_diagnostics
            if diag:
                diagnostics[f"layer_{i}/deltaW_rel_mean"] = diag.get("deltaW_rel_mean", 0.0)
                diagnostics[f"layer_{i}/deltaW_fro_mean"] = diag.get("deltaW_fro_mean", 0.0)
                diagnostics[f"layer_{i}/G_rel_mean"] = diag.get("G_rel_mean", 0.0)
                diagnostics[f"layer_{i}/decay"] = diag.get("decay", 0.0)
                diagnostics[f"layer_{i}/effective_window_tokens"] = diag.get("effective_window_tokens", 0.0)
                diagnostics[f"layer_{i}/inner_lr"] = diag.get("inner_lr", 0.0)
                if "fast_gate" in diag:
                    diagnostics[f"layer_{i}/fast_gate"] = diag["fast_gate"]
                diagnostics[f"layer_{i}/n_resets"] = diag.get("n_resets", 0)
                if "frac_chunks_with_boundary" in diag:
                    diagnostics[f"layer_{i}/frac_chunks_with_boundary"] = diag["frac_chunks_with_boundary"]
                    diagnostics[f"layer_{i}/avg_subspans_per_chunk"] = diag["avg_subspans_per_chunk"]
            tb = wrapper.target_builder
            mix_norm = tb.mix_coeffs.data.norm().item()
            diagnostics[f"layer_{i}/mix_coeffs_norm"] = mix_norm
            wtgt = tb.W_tgt.data.float()
            diag_norm = wtgt.diagonal().norm().item()
            off_diag = wtgt.clone()
            off_diag.fill_diagonal_(0.0)
            off_diag_norm = off_diag.norm().item()
            diagnostics[f"layer_{i}/W_tgt_diag_norm"] = diag_norm
            diagnostics[f"layer_{i}/W_tgt_offdiag_norm"] = off_diag_norm
            if layer._last_deltaW is not None:
                dW = layer._last_deltaW
                W0_norm = wrapper.base_out_proj_weight.float().norm().item()
                dW_norms = torch.norm(dW.reshape(dW.shape[0], -1), dim=1)
                diagnostics[f"layer_{i}/deltaW_rel_max"] = (dW_norms.max().item() / (W0_norm + 1e-8))
    return diagnostics


@torch.no_grad()
def evaluate_sliding_window_ppl(
    model,
    documents,
    context_lengths=None,
    scored_suffix_len=2048,
    device="cuda",
    dtype=torch.bfloat16,
    is_ttt=False,
):
    """
    Evaluate sliding-window perplexity at multiple context lengths.

    Returns: {context_len: ppl}, {context_len: ttt_diagnostics}
    """
    if context_lengths is None:
        context_lengths = CONTEXT_LENGTHS

    model.eval()
    results = {}
    ttt_diag_by_ctx = {}

    for L in context_lengths:
        if L < scored_suffix_len:
            print(f"  Skipping L={L} < suffix={scored_suffix_len}")
            continue

        total_nll = 0.0
        total_tokens = 0
        n_docs = 0
        all_diags = []

        for doc in tqdm(documents, desc=f"L={L}", leave=False):
            if len(doc) < L:
                continue

            input_ids = doc[-L:].unsqueeze(0).to(device)  # [1, L]

            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(input_ids)
            logits = output.logits.float()  # [1, L, V]

            suffix_start = L - scored_suffix_len
            if suffix_start == 0:
                score_logits = logits[:, 0:L - 1, :]
                score_labels = input_ids[:, 1:L]
                n_scored = L - 1
            else:
                score_logits = logits[:, suffix_start - 1:L - 1, :]
                score_labels = input_ids[:, suffix_start:L]
                n_scored = scored_suffix_len

            nll = F.cross_entropy(
                score_logits.reshape(-1, score_logits.size(-1)),
                score_labels.reshape(-1),
                reduction='sum',
            )
            total_nll += nll.item()
            total_tokens += n_scored
            n_docs += 1

            if is_ttt:
                diag = collect_ttt_eval_diagnostics(model)
                if diag:
                    all_diags.append(diag)

        if total_tokens > 0:
            avg_nll = total_nll / total_tokens
            ppl = math.exp(min(avg_nll, 100))
        else:
            avg_nll = float('inf')
            ppl = float('inf')

        results[L] = ppl

        if all_diags:
            avg_diag = {}
            for key in all_diags[0]:
                vals = [d[key] for d in all_diags if key in d]
                avg_diag[key] = sum(vals) / len(vals)
            ttt_diag_by_ctx[L] = avg_diag

        diag_str = ""
        if L in ttt_diag_by_ctx:
            rel_means = {k: v for k, v in ttt_diag_by_ctx[L].items() if "deltaW_rel_mean" in k}
            eff_windows = {k: v for k, v in ttt_diag_by_ctx[L].items() if "effective_window_tokens" in k}
            if rel_means:
                diag_str = " | " + " ".join(f"{k.split('/')[0]}={v:.4f}" for k, v in sorted(rel_means.items()))
            if eff_windows:
                first_ew = list(eff_windows.values())[0]
                diag_str += f" | eff_window={first_ew:.0f}tok"

        print(f"  Context {L:>6} ({L//1024:>2}k): PPL = {ppl:8.2f}  "
              f"(n_docs={n_docs}, avg_nll={avg_nll:.4f}){diag_str}")

    return results, ttt_diag_by_ctx


def plot_figure2(results_dict, output_path="figure2.png", title="Sliding-Window Perplexity"):
    """Plot Figure-2 style comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {"Transformer-SWA": "#1f77b4", "Vanilla-SSM": "#ff7f0e", "SSM+TTT": "#2ca02c"}
    markers = {"Transformer-SWA": "s", "Vanilla-SSM": "^", "SSM+TTT": "o"}

    for name, results in results_dict.items():
        ctx = sorted(results.keys())
        ppls = [results[c] for c in ctx]
        color = None
        marker = "o"
        for prefix in colors:
            if prefix in name:
                color = colors[prefix]
                marker = markers[prefix]
                break
        ax.plot(ctx, ppls, marker=marker, label=name,
                color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks([c for c in sorted(list(results_dict.values())[0].keys())])
    ax.set_xticklabels([f"{c//1024}k" for c in sorted(list(results_dict.values())[0].keys())])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="figure2.png")
    parser.add_argument("--scored_suffix_len", type=int, default=2048)
    parser.add_argument("--max_docs", type=int, default=50)
    parser.add_argument("--context_lengths", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.names)

    context_lengths = args.context_lengths or CONTEXT_LENGTHS

    min_len = max(context_lengths)
    documents = load_val_documents(args.data_dir, min_len=min_len)
    if len(documents) > args.max_docs:
        documents = documents[:args.max_docs]
    print(f"Using {len(documents)} documents for evaluation")

    all_results = {}
    all_ttt_diag = {}
    for ckpt_path, name in zip(args.checkpoints, args.names):
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model, config = load_model_from_checkpoint(ckpt_path, device=args.device)
        is_ttt = config["model_type"] == "ssm_ttt"

        results, ttt_diag = evaluate_sliding_window_ppl(
            model, documents,
            context_lengths=context_lengths,
            scored_suffix_len=args.scored_suffix_len,
            device=args.device,
            is_ttt=is_ttt,
        )
        all_results[name] = results
        if ttt_diag:
            all_ttt_diag[name] = ttt_diag

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<20}" + "".join(f"{'%dk' % (L//1024):>8}" for L in sorted(context_lengths))
    print(header)
    for name, results in all_results.items():
        row = f"{name:<20}" + "".join(f"{results.get(L, float('inf')):8.2f}" for L in sorted(context_lengths))
        print(row)

    if all_ttt_diag:
        print(f"\nTTT Diagnostics (deltaW_rel_mean per layer):")
        for name, diag_by_ctx in all_ttt_diag.items():
            print(f"\n  {name}:")
            for L in sorted(diag_by_ctx.keys()):
                diag = diag_by_ctx[L]
                rel_means = {k: v for k, v in sorted(diag.items()) if "deltaW_rel_mean" in k}
                parts = " ".join(f"{k.split('/')[0]}={v:.4f}" for k, v in rel_means.items())
                eff_w = [v for k, v in diag.items() if "effective_window_tokens" in k]
                fg = [v for k, v in diag.items() if "fast_gate" in k]
                extra = ""
                if eff_w:
                    extra += f" | eff_window={eff_w[0]:.0f}tok"
                if fg:
                    extra += f" | fast_gate={fg[0]:.4f}"
                print(f"    {L//1024}k: {parts}{extra}")

    results_path = args.output.replace(".png", "_results.json")
    save_data = {
        "ppl": {k: {str(kk): vv for kk, vv in v.items()} for k, v in all_results.items()},
    }
    if all_ttt_diag:
        save_data["ttt_diagnostics"] = {
            name: {str(L): diag for L, diag in ctx_diag.items()}
            for name, ctx_diag in all_ttt_diag.items()
        }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to {results_path}")

    plot_figure2(all_results, output_path=args.output)
