#!/usr/bin/env python3
"""
Structural pattern transfer test: is transfer determined by the delimiter
character or by the structural pattern?

Tests 5 formats: same data, varying character and pattern independently.
- Format A: tab + GCF-like flat layout (tab-as-field-separator)
- Format B: tab + TOON-style header+rows (tab-as-TSV)
- Format C: tab + wrapping layout (tab-as-wrapper)
- Format D: pipe + wrapping layout (pipe-as-wrapper)
- Control: pipe + GCF flat layout (standard GCF)

If pattern matters: A transfers, B/C/D don't, control transfers.
If character matters: A/B/C don't transfer (tab), D/control transfer (pipe).

Usage:
  python eval_structural_pattern.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --output structural-pattern-results.json
"""

import argparse
import copy
import datetime
import gc
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')

MODEL_CONFIGS = {
    "410m": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
    },
}


# =========================================================================
# Five test formats: same data, different character + pattern combinations
# =========================================================================

NAMES = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson",
         "Fiona Grant", "George Wu", "Hannah Lee", "Ivan Petrov", "Julia Santos"]
STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]

def gen_data(n=30):
    """Generate raw data rows."""
    rows = []
    for i in range(n):
        rows.append({
            "orderId": f"ORD-{i+1:05d}",
            "customer": NAMES[i % len(NAMES)],
            "status": STATUSES[i % len(STATUSES)],
            "total": str(round(29.97 + i * 12.50, 2)),
        })
    return rows


def format_a_tab_gcf(rows):
    """Tab as field separator in GCF-like flat layout."""
    lines = [f"## orders [{len(rows)}]{{orderId,customer,status,total}}"]
    for r in rows:
        lines.append(f"{r['orderId']}\t{r['customer']}\t{r['status']}\t{r['total']}")
    return "\n".join(lines)


def format_b_tab_tsv(rows):
    """Tab in TOON/TSV-style: header row + data rows."""
    lines = ["orderId\tcustomer\tstatus\ttotal"]
    for r in rows:
        lines.append(f"{r['orderId']}\t{r['customer']}\t{r['status']}\t{r['total']}")
    return "\n".join(lines)


def format_c_tab_wrapper(rows):
    """Tab as wrapper: each field on its own line, tab-delimited key-value."""
    lines = []
    for r in rows:
        lines.append(f"\torderId\t{r['orderId']}\t")
        lines.append(f"\tcustomer\t{r['customer']}\t")
        lines.append(f"\tstatus\t{r['status']}\t")
        lines.append(f"\ttotal\t{r['total']}\t")
        lines.append("")
    return "\n".join(lines)


def format_d_pipe_wrapper(rows):
    """Pipe as wrapper: each field on its own line, pipe-delimited key-value."""
    lines = []
    for r in rows:
        lines.append(f"|orderId|{r['orderId']}|")
        lines.append(f"|customer|{r['customer']}|")
        lines.append(f"|status|{r['status']}|")
        lines.append(f"|total|{r['total']}|")
        lines.append("")
    return "\n".join(lines)


def format_control_gcf(rows):
    """Standard GCF: pipe as field separator in flat layout."""
    lines = [f"## orders [{len(rows)}]{{orderId,customer,status,total}}"]
    for r in rows:
        lines.append(f"{r['orderId']}|{r['customer']}|{r['status']}|{r['total']}")
    return "\n".join(lines)


FORMATS = {
    "A_tab_gcf_layout": {
        "generator": format_a_tab_gcf,
        "character": "tab",
        "pattern": "flat separator",
        "description": "Tab in GCF-like flat layout (## header, tab-separated rows)",
    },
    "B_tab_tsv_layout": {
        "generator": format_b_tab_tsv,
        "character": "tab",
        "pattern": "header+rows",
        "description": "Tab in TOON/TSV style (header row + data rows)",
    },
    "C_tab_wrapper": {
        "generator": format_c_tab_wrapper,
        "character": "tab",
        "pattern": "wrapping",
        "description": "Tab as key-value wrapper (one field per line)",
    },
    "D_pipe_wrapper": {
        "generator": format_d_pipe_wrapper,
        "character": "pipe",
        "pattern": "wrapping",
        "description": "Pipe as key-value wrapper (one field per line)",
    },
    "E_control_gcf": {
        "generator": format_control_gcf,
        "character": "pipe",
        "pattern": "flat separator",
        "description": "Standard GCF (pipe-separated flat rows, control)",
    },
}


# =========================================================================
# Helpers
# =========================================================================

def load_model(checkpoint_path, size, tokenizer_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"
    config = GPTNeoXConfig(**cfg)
    model = GPTNeoXForCausalLM(config)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, tok


def is_delimiter_token(tok, token_id):
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    return any(c in id_to_token.get(token_id, "") for c in BARRIER_CHARS)


def compute_ppl(model, tok, texts, device):
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        ids = tok.encode(text).ids[:2048]
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
        total_loss += loss.item()
        total_tokens += len(ids) - 1
    return math.exp(min(total_loss / max(total_tokens, 1), 20))


def identify_delimiter_heads(model, tok, device, threshold=0.5):
    """Identify delimiter heads using GCF-only text (trained format)."""
    gcf_text = format_control_gcf(gen_data(30))
    ids = tok.encode(gcf_text).ids[:512]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    seq_len = len(ids)

    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    delim_positions = set(i for i, tid in enumerate(ids) if any(c in id_to_token.get(tid, "") for c in BARRIER_CHARS))
    base_rate = len(delim_positions) / max(seq_len, 1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    heads = []
    for layer_idx, attn in enumerate(outputs.attentions):
        for head_idx in range(attn.shape[1]):
            w = attn[0, head_idx].float().cpu()
            delim_attn = sum(w[:, d].mean().item() for d in delim_positions)
            total_attn = sum(w[:, p].mean().item() for p in range(seq_len))
            raw = delim_attn / max(total_attn, 1e-10)
            if raw > threshold:
                heads.append((layer_idx, head_idx, raw))

    del outputs
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    heads.sort(key=lambda x: x[2], reverse=True)
    return heads


def ablate_heads(model, heads):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads:
        dense = model.gpt_neox.layers[layer_idx].attention.dense
        start = head_idx * head_dim
        end = start + head_dim
        dense.weight.data[:, start:end] = 0.0


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Structural pattern transfer test")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--size", default="410m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metadata = {
        "experiment": "structural_pattern_transfer_test",
        "description": "Is cross-format transfer determined by delimiter character or structural pattern?",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("STRUCTURAL PATTERN TRANSFER TEST")
    print("=" * 90)
    print(f"\nDevice: {device}")
    print(f"\nQuestion: does transfer depend on the delimiter character or the structural pattern?")
    print(f"\nPredictions:")
    print(f"  If CHARACTER matters: tab formats (A,B,C) don't transfer. Pipe formats (D,E) transfer.")
    print(f"  If PATTERN matters:   flat-separator formats (A,E) transfer. Others don't.")
    print()

    # Generate data
    rows = gen_data(30)

    print("Formats:")
    for name, info in FORMATS.items():
        text = info["generator"](rows)
        tokens = len(text) // 4  # rough estimate
        print(f"  {name}: {info['character']} + {info['pattern']} ({len(text)} chars)")
    print()

    # Load model
    print("Loading model...", flush=True)
    model, tok = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model.to(device)

    # Identify delimiter heads
    print("Identifying delimiter heads...", flush=True)
    delimiter_heads = identify_delimiter_heads(model, tok, device)
    print(f"Found {len(delimiter_heads)} delimiter heads")

    # Baseline PPL for each format
    print(f"\n{'Format':<22} {'Char':>6} {'Pattern':>16} {'Baseline':>10} {'Ablated':>10} {'Delta':>8} {'Transfers?':>11}")
    print("-" * 88)

    results = {}
    for name, info in FORMATS.items():
        text = info["generator"](rows)

        # Baseline
        baseline = compute_ppl(model, tok, [text], device)

        # Ablated
        model_copy = copy.deepcopy(model)
        model_copy.to(device)
        ablate_heads(model_copy, [(l, h) for l, h, _ in delimiter_heads])
        ablated = compute_ppl(model_copy, tok, [text], device)
        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

        delta = ((ablated - baseline) / baseline) * 100

        if delta > 5:
            transfers = "YES"
        elif delta < -5:
            transfers = "no"
        else:
            transfers = "weak"

        results[name] = {
            "character": info["character"],
            "pattern": info["pattern"],
            "description": info["description"],
            "baseline_ppl": round(baseline, 2),
            "ablated_ppl": round(ablated, 2),
            "delta_pct": round(delta, 1),
            "transfers": transfers,
        }

        print(f"{name:<22} {info['character']:>6} {info['pattern']:>16} {baseline:>10.1f} {ablated:>10.1f} {delta:>+7.1f}% {transfers:>11}")

    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    a_delta = results["A_tab_gcf_layout"]["delta_pct"]
    b_delta = results["B_tab_tsv_layout"]["delta_pct"]
    c_delta = results["C_tab_wrapper"]["delta_pct"]
    d_delta = results["D_pipe_wrapper"]["delta_pct"]
    e_delta = results["E_control_gcf"]["delta_pct"]

    print(f"\nDecisive test: A (tab+flat) vs B (tab+TSV)")
    print(f"  A (tab, GCF layout):  {a_delta:+.1f}%")
    print(f"  B (tab, TSV layout):  {b_delta:+.1f}%")
    if a_delta > 5 and b_delta < 5:
        print(f"  RESULT: PATTERN MATTERS. Same character, different outcome based on layout.")
    elif a_delta < -5 and b_delta < -5:
        print(f"  RESULT: CHARACTER MATTERS. Both tab formats fail regardless of layout.")
    elif a_delta > 5 and b_delta > 5:
        print(f"  RESULT: NEITHER. Both tab formats transfer. Something else is driving it.")
    else:
        print(f"  RESULT: INCONCLUSIVE. Deltas within noise range.")

    print(f"\nConfirmation test: D (pipe+wrapper) vs E (pipe+flat)")
    print(f"  D (pipe, wrapper):   {d_delta:+.1f}%")
    print(f"  E (pipe, flat/GCF):  {e_delta:+.1f}%")
    if e_delta > 5 and d_delta < 5:
        print(f"  CONFIRMED: Pipe in flat layout transfers. Pipe in wrapper layout doesn't.")
        print(f"  The heads learned 'flat field separator', not 'attend to pipe'.")
    elif e_delta > 5 and d_delta > 5:
        print(f"  CHARACTER WINS: Pipe transfers regardless of layout.")
    else:
        print(f"  INCONCLUSIVE.")

    print(f"\nFull pattern matrix:")
    print(f"  Flat separator: A={a_delta:+.1f}%, E={e_delta:+.1f}%")
    print(f"  Header+rows:    B={b_delta:+.1f}%")
    print(f"  Wrapping:       C={c_delta:+.1f}%, D={d_delta:+.1f}%")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "delimiter_heads": len(delimiter_heads),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # R2 upload
    print("\nUploading to R2...", flush=True)
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        s3 = boto3.client("s3",
            endpoint_url="https://b5e39abd50c5b82163c5fe72db9b880e.r2.cloudflarestorage.com",
            aws_access_key_id="d77b3d0a3829377b3b71ffc11f610435",
            aws_secret_access_key="9206e3609275a5b8655d5c5b0f3faf536415e324f4493cfe3ce2b4ffb53e0244",
            config=BotoConfig(signature_version="s3v4"),
        )
        for local, key in [
            (args.output, "logs/run-002-ablation/structural-pattern-results.json"),
        ]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/structural-pattern-log.txt")
            print(f"  Uploaded structural-pattern-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
