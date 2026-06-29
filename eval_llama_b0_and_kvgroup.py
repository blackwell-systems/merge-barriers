#!/usr/bin/env python3
"""
Two experiments for run-003:
1. Model B0 head identification and ablation (is standard Llama's specialization functional?)
2. KV-group ablation on Model A0 (does ablating full KV groups recover NeoX-like effects?)

Usage:
  python eval_llama_b0_and_kvgroup.py \
    --checkpoint-a /root/checkpoints/step-40000/checkpoint.pt \
    --tokenizer-a /root/structok-64k.json \
    --checkpoint-b /root/b0-checkpoint/checkpoint.pt \
    --tokenizer-b /root/standard-64k.json \
    --output /root/run-003-b0-kvgroup-results.json
"""

import argparse
import copy
import datetime
import gc
import json
import math
import platform
import random
from pathlib import Path

import torch
import torch.nn.functional as F

BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')

MODEL_CONFIGS = {
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
# Data generators
# =========================================================================

def gen_gcf_generic(n=50):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson",
             "Fiona Grant", "George Wu", "Hannah Lee", "Ivan Petrov", "Julia Santos"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)

def gen_json(n=50):
    names = ["Alice", "Bob", "Carla", "David", "Eva"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    records = [{"orderId": f"ORD-{i+1:05d}", "customer": names[i%5],
                "status": statuses[i%5], "total": round(29.97+i*12.50, 2)} for i in range(n)]
    return json.dumps({"orders": records}, indent=2)

def gen_yaml(n=30):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    lines = ["employees:"]
    for i in range(n):
        lines.append(f"  - name: {names[i % len(names)]}")
        lines.append(f"    id: EMP-{i+1:04d}")
        lines.append(f"    role: developer")
        lines.append(f"    salary: {50000 + i * 2500}")
    return "\n".join(lines)

NL_TEXT = ("The architecture of modern distributed systems has evolved significantly. "
    "Microservices replaced monolithic applications, bringing independent deployment "
    "and technology diversity, but also complexity in service discovery and tracing.")


# =========================================================================
# Model helpers
# =========================================================================

def load_model(checkpoint_path, size, tokenizer_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"
    config = LlamaConfig(**cfg)
    model = LlamaForCausalLM(config)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    step = cp.get("step", 0)
    print(f"Loaded Llama model from step {step} (tokenizer: {Path(tokenizer_path).stem})")
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


def identify_delimiter_heads(model, tok, device, excess_threshold=0.15):
    test_texts = [gen_gcf_generic(50), gen_json(50), gen_yaml(30)]
    n_heads = model.config.num_attention_heads
    head_excess_scores = {}

    for text in test_texts:
        ids = tok.encode(text).ids[:1024]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))
        seq_len = len(ids)
        base_rate = len(delim_positions) / max(seq_len, 1)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            for head_idx in range(n_heads):
                attn_weights = attn[0, head_idx].float().cpu()
                sl = attn_weights.shape[0]
                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(sl))
                raw_score = delim_attn / max(total_attn, 1e-10)
                excess = raw_score - base_rate
                key = (layer_idx, head_idx)
                if key not in head_excess_scores:
                    head_excess_scores[key] = []
                head_excess_scores[key].append(excess)

        del outputs; gc.collect(); torch.cuda.empty_cache()

    heads = []
    for (l, h), scores in head_excess_scores.items():
        avg = sum(scores) / len(scores)
        if avg > excess_threshold:
            heads.append((l, h, avg))
    heads.sort(key=lambda x: x[2], reverse=True)
    print(f"  Excess threshold: {excess_threshold}")
    print(f"  Heads above threshold: {len(heads)} / {model.config.num_hidden_layers * n_heads}")
    return heads


def ablate_heads(model, heads):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads:
        proj = model.model.layers[layer_idx].self_attn.o_proj
        start = head_idx * head_dim
        end = start + head_dim
        proj.weight.data[:, start:end] = 0.0


def ablate_kv_groups(model, kv_groups):
    """Ablate entire KV groups (all query heads sharing one KV head).

    With GQA 4:1, query heads 0-3 share KV head 0, heads 4-7 share KV head 1, etc.
    Ablating a KV group zeros all 4 query head slices in o_proj.
    """
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    heads_per_group = n_heads // n_kv  # 4
    head_dim = model.config.hidden_size // n_heads

    for layer_idx, kv_idx in kv_groups:
        proj = model.model.layers[layer_idx].self_attn.o_proj
        # Zero all query heads in this KV group
        for q in range(heads_per_group):
            head_idx = kv_idx * heads_per_group + q
            start = head_idx * head_dim
            end = start + head_dim
            proj.weight.data[:, start:end] = 0.0


def identify_delimiter_kv_groups(model, tok, device, excess_threshold=0.15):
    """Identify KV groups where the majority of query heads are delimiter-specialized."""
    heads = identify_delimiter_heads(model, tok, device, excess_threshold)
    head_set = {(l, h) for l, h, _ in heads}

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    heads_per_group = n_heads // n_kv

    kv_groups = []
    for l in range(n_layers):
        for kv in range(n_kv):
            q_heads = [(l, kv * heads_per_group + q) for q in range(heads_per_group)]
            delim_count = sum(1 for qh in q_heads if qh in head_set)
            if delim_count >= 2:  # majority (2+ of 4)
                avg_excess = sum(s for ll, h, s in heads if ll == l and kv * heads_per_group <= h < (kv + 1) * heads_per_group) / max(delim_count, 1)
                kv_groups.append((l, kv, delim_count, avg_excess))

    kv_groups.sort(key=lambda x: x[3], reverse=True)
    print(f"  KV groups with majority delimiter heads: {len(kv_groups)} / {n_layers * n_kv}")
    for l, kv, count, excess in kv_groups[:10]:
        print(f"    L{l} KV{kv}: {count}/4 query heads, avg excess {excess:.3f}")
    return kv_groups


# =========================================================================
# Experiment 1: Model B0 head identification and ablation
# =========================================================================

def run_b0_ablation(model_b, tok_b, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: MODEL B0 (STANDARD LLAMA) HEAD IDENTIFICATION AND ABLATION")
    print("=" * 90)

    test_data = {
        "gcf_generic": [gen_gcf_generic(50)],
        "json": [gen_json(50)],
        "yaml": [gen_yaml(30)],
        "nl": [NL_TEXT],
    }

    # Baselines
    print("\nModel B0 baselines:")
    baselines = {}
    for fmt, texts in test_data.items():
        ppl = compute_ppl(model_b, tok_b, texts, device)
        baselines[fmt] = ppl
        print(f"  {fmt}: {ppl:.1f}")

    # Head identification
    print("\nIdentifying delimiter heads on Model B0...")
    b_heads = identify_delimiter_heads(model_b, tok_b, device, 0.15)
    print(f"Model B0 delimiter heads: {len(b_heads)}")

    if len(b_heads) == 0:
        print("No delimiter heads found. Standard Llama does not develop specialization.")
        return {
            "experiment": "b0_ablation",
            "baselines": {k: round(v, 2) for k, v in baselines.items()},
            "delimiter_heads": 0,
            "conclusion": "Standard Llama develops zero delimiter-specialized heads. Mechanism is tokenizer-dependent.",
        }

    # Ablation
    print(f"\nAblating {len(b_heads)} delimiter heads on Model B0...")
    model_b_abl = copy.deepcopy(model_b)
    model_b_abl.to(device)
    ablate_heads(model_b_abl, [(l, h) for l, h, _ in b_heads])

    ablated = {}
    print(f"\n{'Format':<15} {'Baseline':>10} {'Ablated':>10} {'Delta':>10}")
    print("-" * 50)
    for fmt, texts in test_data.items():
        ppl = compute_ppl(model_b_abl, tok_b, texts, device)
        delta = ((ppl - baselines[fmt]) / baselines[fmt]) * 100
        ablated[fmt] = {"ppl": round(ppl, 2), "delta": round(delta, 1)}
        print(f"  {fmt:<15} {baselines[fmt]:>10.1f} {ppl:>10.1f} {delta:>+9.1f}%")

    del model_b_abl; gc.collect(); torch.cuda.empty_cache()

    functional = any(abs(v["delta"]) > 10 for v in ablated.values())
    conclusion = "B0 heads are FUNCTIONAL (causal)" if functional else "B0 heads are NON-FUNCTIONAL (not causal, same as NeoX Model B)"
    print(f"\nConclusion: {conclusion}")

    return {
        "experiment": "b0_ablation",
        "baselines": {k: round(v, 2) for k, v in baselines.items()},
        "delimiter_heads": len(b_heads),
        "head_details": [{"layer": l, "head": h, "excess": round(s, 4)} for l, h, s in b_heads],
        "ablated": ablated,
        "functional": functional,
        "conclusion": conclusion,
    }


# =========================================================================
# Experiment 2: KV-group ablation on Model A0
# =========================================================================

def run_kvgroup_ablation(model_a, tok_a, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: KV-GROUP ABLATION ON MODEL A0")
    print("=" * 90)
    print("\nRationale: single query-head ablation is weak under GQA because 3 siblings")
    print("still share the same KV projection. Ablating entire KV groups (all 4 query")
    print("heads sharing one KV head) is the equivalent of NeoX's per-head ablation.")

    test_data = {
        "gcf_generic": [gen_gcf_generic(50)],
        "json": [gen_json(50)],
        "yaml": [gen_yaml(30)],
        "nl": [NL_TEXT],
    }

    # Baselines
    baselines = {}
    for fmt, texts in test_data.items():
        baselines[fmt] = compute_ppl(model_a, tok_a, texts, device)

    # Identify delimiter KV groups
    print("\nIdentifying delimiter KV groups...")
    kv_groups = identify_delimiter_kv_groups(model_a, tok_a, device, 0.15)
    kv_set = [(l, kv) for l, kv, _, _ in kv_groups]

    n_layers = model_a.config.num_hidden_layers
    n_kv = model_a.config.num_key_value_heads
    total_kv = n_layers * n_kv
    non_delim_kv = [(l, kv) for l in range(n_layers) for kv in range(n_kv) if (l, kv) not in set(kv_set)]

    print(f"\nDelimiter KV groups: {len(kv_groups)} / {total_kv}")

    # Delimiter KV-group ablation
    print("\nAblating delimiter KV groups...")
    model_abl = copy.deepcopy(model_a)
    model_abl.to(device)
    ablate_kv_groups(model_abl, kv_set)

    delim_results = {}
    print(f"\n{'Format':<15} {'Baseline':>10} {'Delim KV abl':>12} {'Delta':>10}")
    print("-" * 55)
    for fmt, texts in test_data.items():
        ppl = compute_ppl(model_abl, tok_a, texts, device)
        delta = ((ppl - baselines[fmt]) / baselines[fmt]) * 100
        delim_results[fmt] = {"ppl": round(ppl, 2), "delta": round(delta, 1)}
        print(f"  {fmt:<15} {baselines[fmt]:>10.1f} {ppl:>12.1f} {delta:>+9.1f}%")
    del model_abl; gc.collect(); torch.cuda.empty_cache()

    # Random KV-group control (3 seeds)
    print("\nRandom KV-group control (3 seeds)...")
    random_results = {fmt: [] for fmt in test_data}
    for seed in range(3):
        rng = random.Random(seed)
        all_kv = [(l, kv) for l in range(n_layers) for kv in range(n_kv)]
        rng.shuffle(all_kv)
        rand_kv = all_kv[:len(kv_groups)]

        model_rand = copy.deepcopy(model_a)
        model_rand.to(device)
        ablate_kv_groups(model_rand, rand_kv)

        for fmt, texts in test_data.items():
            ppl = compute_ppl(model_rand, tok_a, texts, device)
            delta = ((ppl - baselines[fmt]) / baselines[fmt]) * 100
            random_results[fmt].append(delta)
        del model_rand; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'Format':<15} {'Delim KV Δ':>12} {'Random KV Δ':>12} {'Gap':>10}")
    print("-" * 55)
    summary = {}
    for fmt in test_data:
        d_delta = delim_results[fmt]["delta"]
        r_mean = sum(random_results[fmt]) / len(random_results[fmt])
        gap = d_delta - r_mean
        summary[fmt] = {"delim_delta": d_delta, "random_delta_mean": round(r_mean, 1), "gap": round(gap, 1)}
        print(f"  {fmt:<15} {d_delta:>+11.1f}% {r_mean:>+11.1f}% {gap:>+9.1f}pp")

    gcf_delta = delim_results["gcf_generic"]["delta"]
    json_delta = delim_results["json"]["delta"]
    if gcf_delta > 0:
        conclusion = "KV-GROUP ABLATION RECOVERS NEOX DIRECTION: GCF hurts when delimiter KV groups removed"
    else:
        conclusion = "KV-group ablation does NOT recover NeoX direction on GCF"

    if json_delta < 0:
        conclusion += ". JSON improves (format-adversarial, matches NeoX)"
    else:
        conclusion += ". JSON worsens (different from NeoX)"

    print(f"\nConclusion: {conclusion}")

    return {
        "experiment": "kvgroup_ablation",
        "delimiter_kv_groups": len(kv_groups),
        "total_kv_groups": total_kv,
        "baselines": {k: round(v, 2) for k, v in baselines.items()},
        "delim_ablated": delim_results,
        "random_control": {k: [round(d, 1) for d in v] for k, v in random_results.items()},
        "summary": summary,
        "conclusion": conclusion,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--tokenizer-b", required=True)
    parser.add_argument("--size", default="410m-llama")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 90)
    print("RUN-003: B0 ABLATION + KV-GROUP ABLATION")
    print("=" * 90)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load models
    print("\nLoading Model A (structok)...")
    model_a, tok_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    print("Loading Model B (standard)...")
    model_b, tok_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)
    model_b.to(device)

    results = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    }

    # Experiment 1: B0 heads
    results["b0_ablation"] = run_b0_ablation(model_b, tok_b, device)

    # Free B
    del model_b; gc.collect(); torch.cuda.empty_cache()

    # Experiment 2: KV-group ablation on A
    results["kvgroup_ablation"] = run_kvgroup_ablation(model_a, tok_a, device)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    b0 = results["b0_ablation"]
    print(f"B0 delimiter heads: {b0['delimiter_heads']} ({b0['conclusion']})")
    kv = results["kvgroup_ablation"]
    print(f"KV-group ablation: {kv['conclusion']}")


if __name__ == "__main__":
    main()
