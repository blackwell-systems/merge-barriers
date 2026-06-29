#!/usr/bin/env python3
"""
Head transplant v2: comprehensive controls.

1. Forward transplant: A's delimiter heads -> B (from v1)
2. Random head transplant: A's random non-delimiter heads -> B (critical control)
3. Reverse transplant: B's heads -> A (directionality control)
4. Cross-position transplant: A's delimiter heads -> wrong positions in B
5. Multiple random subsets: 5 random groups of 5 delimiter heads each
6. Extended format coverage: includes TOON and CSV (unseen formats)

Usage:
  python eval_transplant_v2.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoint-b.pt --tokenizer-b standard-64k.json \
    --output transplant-v2-results.json
"""

import argparse
import copy
import datetime
import gc
import json
import math
import random
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
# Test data (includes unseen formats)
# =========================================================================

def gen_gcf_generic(n=50):
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i%5]}|{statuses[i%5]}|{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_json(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    records = [{"orderId":f"ORD-{i+1:05d}","customer":names[i%5],"status":statuses[i%5],"total":round(29.97+i*12.50,2)} for i in range(n)]
    return json.dumps({"orders":records}, indent=2)

def gen_toon(n=50):
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["orderId\tcustomer\tstatus\ttotal"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}\t{names[i%5]}\t{statuses[i%5]}\t{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_csv(n=50):
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["orderId,customer,status,total"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d},{names[i%5]},{statuses[i%5]},{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

NL_TEXT = ("The architecture of modern distributed systems has evolved significantly. "
    "Microservices replaced monolithic applications, bringing independent deployment "
    "and technology diversity, but also complexity in service discovery and tracing. "
    "Container orchestration platforms became the standard deployment target.")

FORMAT_TEXTS = {
    "gcf_generic": [gen_gcf_generic(50)],
    "json": [gen_json(50)],
    "toon": [gen_toon(50)],
    "csv": [gen_csv(50)],
    "nl": [NL_TEXT],
}

FORMAT_TRAINED = {
    "gcf_generic": True,
    "json": True,
    "toon": False,
    "csv": False,
    "nl": True,
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
    step = cp.get("step", 0)
    print(f"Loaded model from step {step} (tokenizer: {Path(tokenizer_path).stem})")
    model.eval()
    return model, tok, step


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
    test_texts = [FORMAT_TEXTS["gcf_generic"][0], FORMAT_TEXTS["json"][0]]
    n_heads = model.config.num_attention_heads
    head_scores = {}

    for text in test_texts:
        ids = tok.encode(text).ids[:1024]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            for head_idx in range(n_heads):
                attn_weights = attn[0, head_idx].float().cpu()
                seq_len = attn_weights.shape[0]
                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(seq_len))
                score = delim_attn / max(total_attn, 1e-10)
                key = (layer_idx, head_idx)
                if key not in head_scores:
                    head_scores[key] = []
                head_scores[key].append(score)

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    heads = []
    all_scores = {}
    for (l, h), scores in head_scores.items():
        avg = sum(scores) / len(scores)
        all_scores[(l, h)] = avg
        if avg > threshold:
            heads.append((l, h, avg))

    heads.sort(key=lambda x: x[2], reverse=True)
    return heads, all_scores


def measure_all(model, tok, device):
    return {fmt: compute_ppl(model, tok, texts, device) for fmt, texts in FORMAT_TEXTS.items()}


def transplant_heads(donor, recipient, heads_donor, heads_recip=None):
    """Copy attention head weights from donor to recipient.

    heads_donor: list of (layer, head) in donor model
    heads_recip: list of (layer, head) in recipient (same as donor if None)
    """
    if heads_recip is None:
        heads_recip = heads_donor

    n_heads = donor.config.num_attention_heads
    head_dim = donor.config.hidden_size // n_heads
    hidden = donor.config.hidden_size

    for (dl, dh), (rl, rh) in zip(heads_donor, heads_recip):
        d_start = dh * head_dim
        d_end = d_start + head_dim
        r_start = rh * head_dim
        r_end = r_start + head_dim

        d_layer = donor.gpt_neox.layers[dl].attention
        r_layer = recipient.gpt_neox.layers[rl].attention

        with torch.no_grad():
            # Q
            r_layer.query_key_value.weight.data[r_start:r_end, :] = \
                d_layer.query_key_value.weight.data[d_start:d_end, :]
            # K
            r_layer.query_key_value.weight.data[hidden+r_start:hidden+r_end, :] = \
                d_layer.query_key_value.weight.data[hidden+d_start:hidden+d_end, :]
            # V
            r_layer.query_key_value.weight.data[2*hidden+r_start:2*hidden+r_end, :] = \
                d_layer.query_key_value.weight.data[2*hidden+d_start:2*hidden+d_end, :]

            if d_layer.query_key_value.bias is not None:
                r_layer.query_key_value.bias.data[r_start:r_end] = \
                    d_layer.query_key_value.bias.data[d_start:d_end]
                r_layer.query_key_value.bias.data[hidden+r_start:hidden+r_end] = \
                    d_layer.query_key_value.bias.data[hidden+d_start:hidden+d_end]
                r_layer.query_key_value.bias.data[2*hidden+r_start:2*hidden+r_end] = \
                    d_layer.query_key_value.bias.data[2*hidden+d_start:2*hidden+d_end]

            # Output projection
            r_layer.dense.weight.data[:, r_start:r_end] = \
                d_layer.dense.weight.data[:, d_start:d_end]


def run_transplant(donor, recipient, tok, heads_donor, heads_recip, device, label):
    """Run a single transplant and measure."""
    model_copy = copy.deepcopy(recipient)
    model_copy.to(device)

    donor.to(device)
    transplant_heads(donor, model_copy, heads_donor, heads_recip)
    donor.cpu()

    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    ppls = measure_all(model_copy, tok, device)
    del model_copy
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return ppls


def print_deltas(ppls, baseline, label, fmt_names):
    line = f"  {label:<40}"
    result = {}
    for fmt in fmt_names:
        delta = ((ppls[fmt] - baseline[fmt]) / baseline[fmt]) * 100
        line += f" {delta:>+8.1f}%"
        result[f"{fmt}_ppl"] = round(ppls[fmt], 2)
        result[f"{fmt}_delta"] = round(delta, 1)
    print(line)
    return result


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Head transplant v2: comprehensive controls")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--tokenizer-b", required=True)
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
        "experiment": "head_transplant_v2",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("HEAD TRANSPLANT v2: COMPREHENSIVE CONTROLS")
    print("=" * 90)
    print(f"\nDevice: {device}")

    print("\nLoading models...")
    model_a, tok_a, _ = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_b, tok_b, _ = load_model(args.checkpoint_b, args.size, args.tokenizer_b)

    # Identify delimiter heads in both models
    model_a.to(device)
    delim_heads_a, all_scores_a = identify_delimiter_heads(model_a, tok_a, device)
    model_a.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    model_b.to(device)
    delim_heads_b, all_scores_b = identify_delimiter_heads(model_b, tok_b, device)
    model_b.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nModel A delimiter heads: {len(delim_heads_a)}")
    print(f"Model B delimiter heads: {len(delim_heads_b)}")

    # Non-delimiter heads in A
    delim_set_a = {(l, h) for l, h, _ in delim_heads_a}
    n_layers = MODEL_CONFIGS[args.size]["num_hidden_layers"]
    n_heads = MODEL_CONFIGS[args.size]["num_attention_heads"]
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    non_delim_heads_a = [(l, h) for l, h in all_heads if (l, h) not in delim_set_a]

    # Baselines
    print("\n" + "=" * 90)
    print("BASELINES")
    print("=" * 90)

    model_b.to(device)
    baseline_b = measure_all(model_b, tok_b, device)
    model_b.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    model_a.to(device)
    baseline_a = measure_all(model_a, tok_a, device)
    model_a.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    fmt_names = list(FORMAT_TEXTS.keys())
    print(f"\n{'':40}", end="")
    for fmt in fmt_names:
        print(f" {fmt:>9}", end="")
    print()
    print("-" * (40 + 10 * len(fmt_names)))

    line = f"  {'Model A baseline':<40}"
    for fmt in fmt_names:
        line += f" {baseline_a[fmt]:>9.0f}"
    print(line)

    line = f"  {'Model B baseline':<40}"
    for fmt in fmt_names:
        line += f" {baseline_b[fmt]:>9.0f}"
    print(line)

    results = {}

    # Header for delta tables
    def print_header(title):
        print(f"\n{'':40}", end="")
        for fmt in fmt_names:
            print(f" {fmt:>9}", end="")
        print()
        print("-" * (40 + 10 * len(fmt_names)))

    # =====================================================================
    # EXPERIMENT 1: Forward transplant (A delimiter -> B)
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXP 1: Forward transplant (A's delimiter heads -> B)")
    print("=" * 90)
    print_header("Deltas vs Model B baseline")

    exp1_results = []
    for n in [5, 10, 20]:
        heads = [(l, h) for l, h, _ in delim_heads_a[:n]]
        ppls = run_transplant(model_a, model_b, tok_b, heads, None, device, f"Top {n} delimiter heads")
        r = print_deltas(ppls, baseline_b, f"Top {n} delimiter heads A->B", fmt_names)
        r["n_heads"] = n
        r["type"] = "delimiter"
        exp1_results.append(r)

    results["exp1_forward_delimiter"] = exp1_results

    # =====================================================================
    # EXPERIMENT 2: Random head transplant (A random -> B) [CRITICAL CONTROL]
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXP 2: Random head transplant (A's non-delimiter heads -> B)")
    print("=" * 90)
    print("CRITICAL CONTROL: if random heads also improve B, the effect")
    print("is not delimiter-specific.")
    print_header("Deltas vs Model B baseline")

    exp2_results = []
    for n in [5, 10, 20]:
        rng = random.Random(42)
        shuffled = list(non_delim_heads_a)
        rng.shuffle(shuffled)
        heads = shuffled[:n]
        ppls = run_transplant(model_a, model_b, tok_b, heads, None, device, f"Random {n} non-delimiter A->B")
        r = print_deltas(ppls, baseline_b, f"Random {n} non-delimiter A->B", fmt_names)
        r["n_heads"] = n
        r["type"] = "random"
        exp2_results.append(r)

    results["exp2_random_control"] = exp2_results

    # =====================================================================
    # EXPERIMENT 3: Reverse transplant (B's heads -> A)
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXP 3: Reverse transplant (B's heads -> A)")
    print("=" * 90)
    print("Directionality control: does putting B's heads into A hurt?")
    print_header("Deltas vs Model A baseline")

    exp3_results = []
    # Use same positions as top delimiter heads in A
    for n in [5, 10, 20]:
        heads = [(l, h) for l, h, _ in delim_heads_a[:n]]
        ppls = run_transplant(model_b, model_a, tok_a, heads, None, device, f"B's heads -> A top {n} positions")
        r = print_deltas(ppls, baseline_a, f"B's heads -> A top {n} positions", fmt_names)
        r["n_heads"] = n
        exp3_results.append(r)

    results["exp3_reverse"] = exp3_results

    # =====================================================================
    # EXPERIMENT 4: Cross-position transplant
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXP 4: Cross-position transplant (A's delimiter heads -> wrong positions in B)")
    print("=" * 90)
    print("Tests whether knowledge is position-dependent or weight-portable.")
    print_header("Deltas vs Model B baseline")

    exp4_results = []
    # Shift positions: put layer L head H into layer (L+12)%24 head (H+8)%16
    n = 20
    donor_heads = [(l, h) for l, h, _ in delim_heads_a[:n]]
    shifted_recip = [((l + 12) % 24, (h + 8) % 16) for l, h in donor_heads]
    ppls = run_transplant(model_a, model_b, tok_b, donor_heads, shifted_recip, device, f"Top {n} shifted positions")
    r = print_deltas(ppls, baseline_b, f"Top {n} delimiter, shifted positions", fmt_names)
    r["n_heads"] = n
    r["type"] = "cross_position"
    exp4_results.append(r)

    # Also test correct positions for comparison
    ppls = run_transplant(model_a, model_b, tok_b, donor_heads, None, device, f"Top {n} correct positions")
    r = print_deltas(ppls, baseline_b, f"Top {n} delimiter, correct positions", fmt_names)
    r["n_heads"] = n
    r["type"] = "correct_position"
    exp4_results.append(r)

    results["exp4_cross_position"] = exp4_results

    # =====================================================================
    # EXPERIMENT 5: Multiple random subsets of delimiter heads
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXP 5: Random subsets of 5 delimiter heads (5 seeds)")
    print("=" * 90)
    print("Tests whether all delimiter heads contribute or only the top ones.")
    print_header("Deltas vs Model B baseline")

    exp5_results = []
    delim_list = [(l, h) for l, h, _ in delim_heads_a]
    for seed in range(5):
        rng = random.Random(seed)
        subset = rng.sample(delim_list, 5)
        ppls = run_transplant(model_a, model_b, tok_b, subset, None, device, f"Random 5 delimiter (seed {seed})")
        r = print_deltas(ppls, baseline_b, f"Random 5 delimiter (seed {seed})", fmt_names)
        r["seed"] = seed
        r["heads"] = [{"layer": l, "head": h} for l, h in subset]
        exp5_results.append(r)

    # Compute mean and std
    for fmt in fmt_names:
        deltas = [r[f"{fmt}_delta"] for r in exp5_results]
        mean_d = sum(deltas) / len(deltas)
        std_d = (sum((d - mean_d) ** 2 for d in deltas) / len(deltas)) ** 0.5
        print(f"  {fmt}: mean {mean_d:+.1f}%, std {std_d:.1f}%")

    results["exp5_random_subsets"] = exp5_results

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Compare delimiter vs random at n=20
    delim_20 = exp1_results[2]  # n=20 delimiter
    random_20 = exp2_results[2]  # n=20 random

    print(f"\nAt 20 heads transplanted (A -> B):")
    print(f"  {'':30} {'GCF':>9} {'JSON':>9} {'TOON':>9} {'CSV':>9} {'NL':>9}")
    print(f"  {'Delimiter heads':30}", end="")
    for fmt in fmt_names:
        print(f" {delim_20.get(f'{fmt}_delta', 0):>+8.1f}%", end="")
    print()
    print(f"  {'Random non-delimiter heads':30}", end="")
    for fmt in fmt_names:
        print(f" {random_20.get(f'{fmt}_delta', 0):>+8.1f}%", end="")
    print()

    gcf_delim = delim_20.get("gcf_generic_delta", 0)
    gcf_random = random_20.get("gcf_generic_delta", 0)

    if gcf_delim < gcf_random - 10:
        print(f"\n  DELIMITER-SPECIFIC: delimiter heads improve GCF {gcf_delim:+.1f}% vs random {gcf_random:+.1f}%")
    elif gcf_delim < -5 and gcf_random < -5:
        print(f"\n  NOT DELIMITER-SPECIFIC: both improve GCF (delimiter {gcf_delim:+.1f}%, random {gcf_random:+.1f}%)")
    else:
        print(f"\n  INCONCLUSIVE: delimiter {gcf_delim:+.1f}%, random {gcf_random:+.1f}%")

    # Check reverse
    reverse_20 = exp3_results[2]
    gcf_reverse = reverse_20.get("gcf_generic_delta", 0)
    if gcf_reverse > 5:
        print(f"  DIRECTIONAL: B's heads hurt A ({gcf_reverse:+.1f}% GCF), confirming A->B is not symmetric")
    elif gcf_reverse < -5:
        print(f"  NOT DIRECTIONAL: B's heads also help A ({gcf_reverse:+.1f}% GCF)")

    # Check cross-position
    cross_pos = exp4_results[0]
    correct_pos = exp4_results[1]
    gcf_cross = cross_pos.get("gcf_generic_delta", 0)
    gcf_correct = correct_pos.get("gcf_generic_delta", 0)
    if gcf_cross > gcf_correct + 20:
        print(f"  POSITION-DEPENDENT: correct position {gcf_correct:+.1f}%, shifted {gcf_cross:+.1f}%")
    else:
        print(f"  POSITION-INDEPENDENT: correct {gcf_correct:+.1f}%, shifted {gcf_cross:+.1f}%")

    # Check unseen format transfer
    toon_delim = delim_20.get("toon_delta", 0)
    csv_delim = delim_20.get("csv_delta", 0)
    if toon_delim < -10 or csv_delim < -10:
        print(f"  CROSS-FORMAT TRANSFER via transplant: TOON {toon_delim:+.1f}%, CSV {csv_delim:+.1f}%")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "delimiter_heads_a": len(delim_heads_a),
            "delimiter_heads_b": len(delim_heads_b),
            "baseline_a": {k: round(v, 4) for k, v in baseline_a.items()},
            "baseline_b": {k: round(v, 4) for k, v in baseline_b.items()},
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
        import os
        s3 = boto3.client("s3",
            endpoint_url="https://b5e39abd50c5b82163c5fe72db9b880e.r2.cloudflarestorage.com",
            aws_access_key_id="d77b3d0a3829377b3b71ffc11f610435",
            aws_secret_access_key="9206e3609275a5b8655d5c5b0f3faf536415e324f4493cfe3ce2b4ffb53e0244",
            config=BotoConfig(signature_version="s3v4"),
        )
        for local, key in [
            (args.output, "logs/run-002-ablation/transplant-v2-results.json"),
        ]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/transplant-v2-log.txt")
            print(f"  Uploaded transplant-v2-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
