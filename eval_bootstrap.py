#!/usr/bin/env python3
"""
Bootstrap confidence intervals for ablation PPL measurements.

Reruns the key ablation measurement (baseline, delimiter ablation, control ablation)
5 times with different test data samples to estimate variance.

Usage:
  python eval_bootstrap.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --output bootstrap-results.json
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
    "410m-llama": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2816,
        "max_position_embeddings": 2048,
        "rope_theta": 500000.0,
    },
}


# =========================================================================
# Generate diverse test data (different seeds for bootstrap)
# =========================================================================

def gen_gcf_generic(seed, n=50):
    rng = random.Random(seed)
    first_names = ["Alice","Bob","Carla","David","Eva","Fiona","George","Hannah","Ivan","Julia",
                   "Kevin","Laura","Marco","Nina","Oscar","Paula","Quinn","Rosa","Sam","Tina"]
    last_names = ["Chen","Smith","Rodriguez","Park","Johansson","Grant","Wu","Lee","Petrov","Santos",
                  "Miller","Brown","Garcia","Wilson","Taylor","Anderson","Thomas","Jackson","White","Harris"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        first = rng.choice(first_names)
        last = rng.choice(last_names)
        status = rng.choice(statuses)
        total = round(rng.uniform(10, 500), 2)
        lines.append(f"ORD-{rng.randint(10000,99999)}|{first} {last}|{status}|{total}")
    return "\n".join(lines)

def gen_json(seed, n=50):
    rng = random.Random(seed)
    first_names = ["Alice","Bob","Carla","David","Eva","Fiona","George","Hannah","Ivan","Julia"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    records = []
    for i in range(n):
        records.append({
            "orderId": f"ORD-{rng.randint(10000,99999)}",
            "customer": rng.choice(first_names),
            "status": rng.choice(statuses),
            "total": round(rng.uniform(10, 500), 2),
        })
    return json.dumps({"orders": records}, indent=2)

NL_TEXTS = [
    "The architecture of modern distributed systems has evolved significantly over the past decade. "
    "Microservices replaced monolithic applications, bringing benefits like independent deployment.",
    "Natural language processing has been transformed by transformer architectures. Pre-training on "
    "large corpora followed by task-specific fine-tuning has become the dominant paradigm.",
    "Network engineers monitor traffic patterns across data center fabrics to identify bottlenecks. "
    "Modern spine-leaf architectures distribute load evenly across parallel paths.",
]


# =========================================================================
# Helpers
# =========================================================================

def load_model(checkpoint_path, size, tokenizer_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"

    if "llama" in size:
        cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"] // 4)
        cfg.setdefault("rope_theta", 500000.0)
        config = LlamaConfig(**cfg)
        model = LlamaForCausalLM(config)
    else:
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


def identify_delimiter_heads(model, tok, texts, device, excess_threshold=0.15):
    n_heads = model.config.num_attention_heads
    head_excess_scores = {}

    for text in texts:
        ids = tok.encode(text).ids[:1024]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))
        base_rate = len(delim_positions) / max(len(ids), 1)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            for head_idx in range(n_heads):
                attn_weights = attn[0, head_idx].float().cpu()
                sl = attn_weights.shape[0]
                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(sl))
                raw = delim_attn / max(total_attn, 1e-10)
                excess = raw - base_rate
                key = (layer_idx, head_idx)
                if key not in head_excess_scores:
                    head_excess_scores[key] = []
                head_excess_scores[key].append(excess)

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    heads = []
    for (l, h), scores in head_excess_scores.items():
        avg = sum(scores) / len(scores)
        if avg > excess_threshold:
            heads.append((l, h, avg))

    heads.sort(key=lambda x: x[2], reverse=True)
    print(f"  Excess threshold: {excess_threshold}, heads: {len(heads)} / {model.config.num_hidden_layers * n_heads}")
    return heads


def _get_output_proj(model, layer_idx):
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_idx].attention.dense
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn.o_proj
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")


def ablate_heads(model, heads):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads:
        proj = _get_output_proj(model, layer_idx)
        start = head_idx * head_dim
        end = start + head_dim
        proj.weight.data[:, start:end] = 0.0


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--size", default="410m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=5)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metadata = {
        "experiment": "bootstrap_confidence_intervals",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "n_bootstrap": args.n_bootstrap,
    }

    print("=" * 90)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({args.n_bootstrap} samples)")
    print("=" * 90)

    print("\nLoading model...")
    model, tok = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model.to(device)

    # Identify delimiter heads once (using seed 0 data)
    id_texts = [gen_gcf_generic(0, 50), gen_json(0, 50)]
    delimiter_heads = identify_delimiter_heads(model, tok, id_texts, device)
    delim_set = {(l, h) for l, h, _ in delimiter_heads}
    n_layers = model.config.num_hidden_layers
    n_heads_per = model.config.num_attention_heads
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per)]
    non_delim = [(l, h) for l, h in all_heads if (l, h) not in delim_set]

    print(f"Delimiter heads: {len(delimiter_heads)}")

    # Bootstrap: for each seed, generate new test data, measure baseline + ablation
    fmt_names = ["gcf_generic", "json", "nl"]
    all_baselines = {fmt: [] for fmt in fmt_names}
    all_delim_ablated = {fmt: [] for fmt in fmt_names}
    all_random_ablated = {fmt: [] for fmt in fmt_names}
    all_deltas_delim = {fmt: [] for fmt in fmt_names}
    all_deltas_random = {fmt: [] for fmt in fmt_names}

    print(f"\n{'Seed':>5} {'':>5}", end="")
    for fmt in fmt_names:
        print(f" {'base_'+fmt:>14} {'delim_'+fmt:>14} {'rand_'+fmt:>14}", end="")
    print()
    print("-" * (10 + 43 * len(fmt_names)))

    for seed in range(args.n_bootstrap):
        # Generate test data with this seed
        test_data = {
            "gcf_generic": [gen_gcf_generic(seed * 100 + 1, 50), gen_gcf_generic(seed * 100 + 2, 30)],
            "json": [gen_json(seed * 100 + 1, 50), gen_json(seed * 100 + 2, 30)],
            "nl": NL_TEXTS,
        }

        # Baseline
        base = {}
        for fmt in fmt_names:
            base[fmt] = compute_ppl(model, tok, test_data[fmt], device)
            all_baselines[fmt].append(base[fmt])

        # Delimiter ablation
        model_delim = copy.deepcopy(model)
        model_delim.to(device)
        ablate_heads(model_delim, [(l, h) for l, h, _ in delimiter_heads])
        delim_abl = {}
        for fmt in fmt_names:
            delim_abl[fmt] = compute_ppl(model_delim, tok, test_data[fmt], device)
            all_delim_ablated[fmt].append(delim_abl[fmt])
            delta = ((delim_abl[fmt] - base[fmt]) / base[fmt]) * 100
            all_deltas_delim[fmt].append(delta)
        del model_delim
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

        # Random ablation (same count)
        rng = random.Random(seed)
        shuffled = list(non_delim)
        rng.shuffle(shuffled)
        model_rand = copy.deepcopy(model)
        model_rand.to(device)
        ablate_heads(model_rand, shuffled[:len(delimiter_heads)])
        rand_abl = {}
        for fmt in fmt_names:
            rand_abl[fmt] = compute_ppl(model_rand, tok, test_data[fmt], device)
            all_random_ablated[fmt].append(rand_abl[fmt])
            delta = ((rand_abl[fmt] - base[fmt]) / base[fmt]) * 100
            all_deltas_random[fmt].append(delta)
        del model_rand
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

        line = f"{seed:>5} {'':>5}"
        for fmt in fmt_names:
            line += f" {base[fmt]:>14.1f} {delim_abl[fmt]:>14.1f} {rand_abl[fmt]:>14.1f}"
        print(line)

    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY: Mean +/- Std across bootstrap samples")
    print("=" * 90)

    print(f"\n{'Format':<14} {'Baseline':>14} {'Delim abl':>14} {'Delim delta':>14} {'Random abl':>14} {'Random delta':>14}")
    print("-" * 86)

    summary = {}
    for fmt in fmt_names:
        b_mean = sum(all_baselines[fmt]) / len(all_baselines[fmt])
        b_std = (sum((x - b_mean)**2 for x in all_baselines[fmt]) / len(all_baselines[fmt])) ** 0.5

        dd_mean = sum(all_deltas_delim[fmt]) / len(all_deltas_delim[fmt])
        dd_std = (sum((x - dd_mean)**2 for x in all_deltas_delim[fmt]) / len(all_deltas_delim[fmt])) ** 0.5

        rd_mean = sum(all_deltas_random[fmt]) / len(all_deltas_random[fmt])
        rd_std = (sum((x - rd_mean)**2 for x in all_deltas_random[fmt]) / len(all_deltas_random[fmt])) ** 0.5

        da_mean = sum(all_delim_ablated[fmt]) / len(all_delim_ablated[fmt])
        ra_mean = sum(all_random_ablated[fmt]) / len(all_random_ablated[fmt])

        print(f"{fmt:<14} {b_mean:>10.1f}+/-{b_std:<4.1f} {da_mean:>10.1f}      {dd_mean:>+8.1f}+/-{dd_std:<4.1f} {ra_mean:>10.1f}      {rd_mean:>+8.1f}+/-{rd_std:<4.1f}")

        summary[fmt] = {
            "baseline_mean": round(b_mean, 2), "baseline_std": round(b_std, 2),
            "delim_delta_mean": round(dd_mean, 2), "delim_delta_std": round(dd_std, 2),
            "random_delta_mean": round(rd_mean, 2), "random_delta_std": round(rd_std, 2),
            "baseline_samples": all_baselines[fmt],
            "delim_delta_samples": all_deltas_delim[fmt],
            "random_delta_samples": all_deltas_random[fmt],
        }

    # Key question: is delimiter ablation significantly different from random?
    print("\nSignificance check (delimiter vs random delta):")
    for fmt in fmt_names:
        dd = all_deltas_delim[fmt]
        rd = all_deltas_random[fmt]
        diffs = [d - r for d, r in zip(dd, rd)]
        diff_mean = sum(diffs) / len(diffs)
        diff_std = (sum((x - diff_mean)**2 for x in diffs) / len(diffs)) ** 0.5
        # Simple check: is the difference consistently positive or negative?
        same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        print(f"  {fmt}: delim-random = {diff_mean:+.1f}% +/- {diff_std:.1f}%  {'(consistent direction)' if same_sign else '(mixed direction)'}")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "delimiter_heads": len(delimiter_heads),
            "summary": summary,
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
        for local, key in [(args.output, "logs/run-002-ablation/bootstrap-results.json")]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/bootstrap-log.txt")
            print(f"  Uploaded bootstrap-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
