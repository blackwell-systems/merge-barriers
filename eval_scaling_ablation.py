#!/usr/bin/env python3
"""
Scaling ablation: does the delimiter head causal effect strengthen with payload size?

Tests the core ablation (delimiter vs random head removal) at 5 payload sizes:
10, 30, 50, 100, 200 rows. If the delimiter-random gap widens with scale,
that's consistent with the PPL scaling finding (2.1x to 5.3x from 3 to 100 records).

Usage:
  python eval_scaling_ablation.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --output scaling-ablation-results.json
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
# Test data generators
# =========================================================================

def gen_gcf_generic(n):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson",
             "Fiona Grant", "George Wu", "Hannah Lee", "Ivan Petrov", "Julia Santos"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def gen_json(n):
    names = ["Alice", "Bob", "Carla", "David", "Eva"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    records = [{"orderId": f"ORD-{i+1:05d}", "customer": names[i % 5],
                "status": statuses[i % 5], "total": round(29.97 + i * 12.50, 2)} for i in range(n)]
    return json.dumps({"orders": records}, indent=2)


def gen_toon(n):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = ["orderId\tcustomer\tstatus\ttotal"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}\t{names[i % 5]}\t{statuses[i % 5]}\t{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


NL_TEXT = ("The architecture of modern distributed systems has evolved significantly. "
    "Microservices replaced monolithic applications, bringing independent deployment "
    "and technology diversity, but also complexity in service discovery and tracing. "
    "Container orchestration platforms became the standard deployment target.")


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
    print(f"Loaded model from step {step}")
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
    # Use 50-row GCF and JSON for identification (trained formats, moderate size)
    test_texts = [gen_gcf_generic(50), gen_json(50)]
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
    for (l, h), scores in head_scores.items():
        avg = sum(scores) / len(scores)
        if avg > threshold:
            heads.append((l, h, avg))

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
    parser = argparse.ArgumentParser(description="Scaling ablation experiment")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--size", default="410m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--control-seeds", type=int, default=3)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metadata = {
        "experiment": "scaling_ablation",
        "description": "Does the delimiter head causal effect strengthen with payload size?",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "control_seeds": args.control_seeds,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("SCALING ABLATION: PAYLOAD SIZE vs CAUSAL EFFECT")
    print("=" * 90)
    print(f"\nDevice: {device}")
    print(f"Control seeds: {args.control_seeds}")
    print(f"Hypothesis: delimiter-random gap widens with scale")

    print("\nLoading model...")
    model, tok = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model.to(device)

    # Identify delimiter heads
    print("\nIdentifying delimiter heads...")
    delimiter_heads = identify_delimiter_heads(model, tok, device)
    delim_set = {(l, h) for l, h, _ in delimiter_heads}
    n_layers = model.config.num_hidden_layers
    n_heads_per = model.config.num_attention_heads
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per)]
    non_delim = [(l, h) for l, h in all_heads if (l, h) not in delim_set]
    print(f"Delimiter heads: {len(delimiter_heads)} / {n_layers * n_heads_per}")

    # Test sizes
    sizes = [10, 30, 50, 100, 200]
    formats = ["gcf_generic", "json", "toon", "nl"]

    print(f"\nPayload sizes: {sizes}")
    print(f"Formats: {formats}")

    # Token counts for reference
    print(f"\nToken counts per size:")
    for n in sizes:
        gcf_toks = len(tok.encode(gen_gcf_generic(n)).ids)
        json_toks = len(tok.encode(gen_json(n)).ids)
        toon_toks = len(tok.encode(gen_toon(n)).ids)
        truncated_json = " (TRUNCATED)" if json_toks > 2048 else ""
        truncated_gcf = " (TRUNCATED)" if gcf_toks > 2048 else ""
        print(f"  {n:>4} rows: GCF={gcf_toks:>5}{truncated_gcf}  JSON={json_toks:>5}{truncated_json}  TOON={toon_toks:>5}")

    # Run at each size
    all_results = []

    print(f"\n{'Size':>5} {'Format':<12} {'Baseline':>10} {'Delim abl':>10} {'Delim Δ':>9} {'Random Δ (mean)':>16} {'Gap':>8}")
    print("-" * 78)

    for n in sizes:
        # Generate test data at this size
        test_data = {
            "gcf_generic": [gen_gcf_generic(n)],
            "json": [gen_json(n)],
            "toon": [gen_toon(n)],
            "nl": [NL_TEXT],
        }

        size_result = {"size": n, "formats": {}}

        for fmt in formats:
            # Baseline
            baseline = compute_ppl(model, tok, test_data[fmt], device)

            # Delimiter ablation
            model_delim = copy.deepcopy(model)
            model_delim.to(device)
            ablate_heads(model_delim, [(l, h) for l, h, _ in delimiter_heads])
            delim_ppl = compute_ppl(model_delim, tok, test_data[fmt], device)
            delim_delta = ((delim_ppl - baseline) / baseline) * 100
            del model_delim
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

            # Random ablation (multiple seeds)
            random_deltas = []
            for seed in range(args.control_seeds):
                rng = random.Random(seed)
                shuffled = list(non_delim)
                rng.shuffle(shuffled)
                model_rand = copy.deepcopy(model)
                model_rand.to(device)
                ablate_heads(model_rand, shuffled[:len(delimiter_heads)])
                rand_ppl = compute_ppl(model_rand, tok, test_data[fmt], device)
                rand_delta = ((rand_ppl - baseline) / baseline) * 100
                random_deltas.append(rand_delta)
                del model_rand
                gc.collect()
                if device != "cpu":
                    torch.cuda.empty_cache()

            rand_mean = sum(random_deltas) / len(random_deltas)
            gap = delim_delta - rand_mean

            size_result["formats"][fmt] = {
                "baseline_ppl": round(baseline, 2),
                "delim_ablated_ppl": round(delim_ppl, 2),
                "delim_delta_pct": round(delim_delta, 1),
                "random_delta_mean_pct": round(rand_mean, 1),
                "random_delta_samples": [round(d, 1) for d in random_deltas],
                "gap_pp": round(gap, 1),
            }

            print(f"{n:>5} {fmt:<12} {baseline:>10.1f} {delim_ppl:>10.1f} {delim_delta:>+8.1f}% {rand_mean:>+14.1f}%  {gap:>+7.1f}pp")

        all_results.append(size_result)
        print()

    # Summary: gap trend
    print("=" * 90)
    print("SCALING SUMMARY: Delimiter-Random gap by payload size")
    print("=" * 90)

    print(f"\n{'Size':>5}", end="")
    for fmt in formats:
        print(f" {fmt:>14}", end="")
    print()
    print("-" * (5 + 15 * len(formats)))

    for r in all_results:
        line = f"{r['size']:>5}"
        for fmt in formats:
            gap = r["formats"][fmt]["gap_pp"]
            line += f" {gap:>+13.1f}pp"
        print(line)

    # Check if gap widens with scale
    print("\nScaling trend (GCF generic gap):")
    gcf_gaps = [(r["size"], r["formats"]["gcf_generic"]["gap_pp"]) for r in all_results]
    for size, gap in gcf_gaps:
        bar = "+" * max(0, int(gap / 2)) if gap > 0 else "-" * max(0, int(-gap / 2))
        print(f"  {size:>4} rows: {gap:>+7.1f}pp  {bar}")

    first_gap = gcf_gaps[0][1]
    last_gap = gcf_gaps[-1][1]
    if last_gap > first_gap + 5:
        print(f"\n  CONFIRMED: gap widens from {first_gap:+.1f}pp at {gcf_gaps[0][0]} rows to {last_gap:+.1f}pp at {gcf_gaps[-1][0]} rows")
    elif last_gap < first_gap - 5:
        print(f"\n  REVERSED: gap narrows from {first_gap:+.1f}pp to {last_gap:+.1f}pp")
    else:
        print(f"\n  STABLE: gap is {first_gap:+.1f}pp to {last_gap:+.1f}pp (within noise)")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "delimiter_heads": len(delimiter_heads),
            "sizes_tested": sizes,
            "results": all_results,
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
            (args.output, "logs/run-002-ablation/scaling-ablation-results.json"),
        ]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/scaling-ablation-log.txt")
            print(f"  Uploaded scaling-ablation-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
