#!/usr/bin/env python3
"""
Run-003 Llama ablation: layer-wise, sufficiency, single-head ranking,
attention patterns, and emergence timing.

Usage:
  python eval_llama_ablation.py \
    --checkpoint /root/checkpoints/step-40000/checkpoint.pt \
    --tokenizer /root/structok-64k.json \
    --output /root/run-003-llama-ablation-results.json \
    --checkpoint-dir /root/checkpoints
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


def gen_gcf_graph(n_symbols=30, n_edges=20):
    packages = ["pkg/auth", "pkg/server", "pkg/db", "pkg/cache", "pkg/config"]
    names = ["Validate", "Process", "Handle", "Create", "Update", "Delete", "Get", "Set"]
    kinds = ["fn", "type", "method", "iface"]
    provs = ["lsp_resolved", "ast_inferred", "structural"]
    lines = [f"GCF profile=graph symbols={n_symbols} edges={n_edges}"]
    groups = {"targets": [], "related": [], "extended": []}
    for i in range(n_symbols):
        g = "targets" if i < n_symbols // 3 else ("related" if i < 2 * n_symbols // 3 else "extended")
        groups[g].append(f"@{i} {kinds[i%4]} {packages[i%5]}.{names[i%8]}{i} {round(max(0.10, 0.95-i*0.02),2)} {provs[i%3]}")
    for g, syms in groups.items():
        if syms:
            lines.append(f"## {g}")
            lines.extend(syms)
    lines.append(f"## edges [{n_edges}]")
    for i in range(n_edges):
        lines.append(f"@{(i*3)%n_symbols}<@{(i*3+1)%n_symbols} calls")
    return "\n".join(lines)


def gen_json(n=50):
    names = ["Alice", "Bob", "Carla", "David", "Eva"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    records = [{"orderId": f"ORD-{i+1:05d}", "customer": names[i%5],
                "status": statuses[i%5], "total": round(29.97+i*12.50, 2)} for i in range(n)]
    return json.dumps({"orders": records}, indent=2)


def gen_yaml(n=50):
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
    cfg["_attn_implementation"] = "eager"  # needed for output_attentions
    config = LlamaConfig(**cfg)
    model = LlamaForCausalLM(config)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    step = cp.get("step", 0)
    print(f"Loaded Llama model from step {step}")
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


def reverse_ablate(model, heads_to_keep):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    keep_set = set(heads_to_keep)
    for l in range(n_layers):
        for h in range(n_heads):
            if (l, h) not in keep_set:
                proj = _get_output_proj(model, l)
                start = h * head_dim
                end = start + head_dim
                proj.weight.data[:, start:end] = 0.0


def identify_delimiter_heads(model, tok, device, excess_threshold=0.15):
    test_texts = [gen_gcf_generic(50), gen_gcf_graph(30, 20), gen_json(50), gen_yaml(30)]
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

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

    heads = []
    for (l, h), scores in head_excess_scores.items():
        avg = sum(scores) / len(scores)
        if avg > excess_threshold:
            heads.append((l, h, avg))
    heads.sort(key=lambda x: x[2], reverse=True)

    print(f"  Excess threshold: {excess_threshold}")
    print(f"  Heads above threshold: {len(heads)} / {model.config.num_hidden_layers * n_heads}")
    return heads


# =========================================================================
# Experiment 1: Layer-wise ablation
# =========================================================================

def run_layer_wise(model, tok, delimiter_heads, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: LAYER-WISE ABLATION")
    print("=" * 90)

    n_layers = model.config.num_hidden_layers
    groups = {
        "early (0-7)": [(l, h) for l, h, _ in delimiter_heads if l <= 7],
        "middle (8-15)": [(l, h) for l, h, _ in delimiter_heads if 8 <= l <= 15],
        "late (16-23)": [(l, h) for l, h, _ in delimiter_heads if l >= 16],
    }

    test_data = {"gcf_generic": [gen_gcf_generic(50)], "json": [gen_json(50)], "yaml": [gen_yaml(30)], "nl": [NL_TEXT]}
    baseline = {fmt: compute_ppl(model, tok, texts, device) for fmt, texts in test_data.items()}

    print(f"\n{'Group':<20} {'Heads':>6} {'GCF delta':>10} {'JSON delta':>10} {'YAML delta':>10} {'NL delta':>10}")
    print("-" * 70)

    results = {}
    for group_name, group_heads in groups.items():
        model_abl = copy.deepcopy(model)
        model_abl.to(device)
        ablate_heads(model_abl, group_heads)
        deltas = {}
        for fmt, texts in test_data.items():
            ppl = compute_ppl(model_abl, tok, texts, device)
            delta = ((ppl - baseline[fmt]) / baseline[fmt]) * 100
            deltas[fmt] = round(delta, 1)
        del model_abl
        gc.collect()
        torch.cuda.empty_cache()

        results[group_name] = {"heads": len(group_heads), "deltas": deltas}
        print(f"{group_name:<20} {len(group_heads):>6} {deltas['gcf_generic']:>+9.1f}% {deltas['json']:>+9.1f}% {deltas['yaml']:>+9.1f}% {deltas['nl']:>+9.1f}%")

    return {"experiment": "layer_wise", "baseline": {k: round(v, 2) for k, v in baseline.items()}, "groups": results}


# =========================================================================
# Experiment 2: Sufficiency scaling
# =========================================================================

def run_sufficiency(model, tok, delimiter_heads, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: SUFFICIENCY SCALING")
    print("=" * 90)

    delim_set = {(l, h) for l, h, _ in delimiter_heads}
    all_heads = [(l, h) for l in range(model.config.num_hidden_layers) for h in range(model.config.num_attention_heads)]
    sizes = [30, 50, 100, 200]

    print(f"\n{'Size':>5} {'Baseline':>10} {'Delim only':>12} {'Delim delta':>12} {'Random delta':>12} {'Sufficient?':>12}")
    print("-" * 75)

    results = []
    for n in sizes:
        texts = [gen_gcf_generic(n)]
        baseline = compute_ppl(model, tok, texts, device)

        model_d = copy.deepcopy(model)
        model_d.to(device)
        reverse_ablate(model_d, delim_set)
        delim_ppl = compute_ppl(model_d, tok, texts, device)
        delim_delta = ((delim_ppl - baseline) / baseline) * 100
        del model_d; gc.collect(); torch.cuda.empty_cache()

        rand_deltas = []
        for seed in range(3):
            rng = random.Random(seed + 100)
            shuffled = list(all_heads)
            rng.shuffle(shuffled)
            keep = set(shuffled[:len(delimiter_heads)])
            model_r = copy.deepcopy(model)
            model_r.to(device)
            reverse_ablate(model_r, keep)
            rand_ppl = compute_ppl(model_r, tok, texts, device)
            rand_deltas.append(((rand_ppl - baseline) / baseline) * 100)
            del model_r; gc.collect(); torch.cuda.empty_cache()

        rand_mean = sum(rand_deltas) / len(rand_deltas)
        sufficient = "YES" if delim_delta < rand_mean else "NO"

        results.append({"size": n, "baseline": round(baseline, 2), "delim_delta": round(delim_delta, 1),
                        "random_delta_mean": round(rand_mean, 1), "sufficient": sufficient})
        print(f"{n:>5} {baseline:>10.1f} {delim_ppl:>12.1f} {delim_delta:>+11.1f}% {rand_mean:>+11.1f}% {sufficient:>12}")

    return {"experiment": "sufficiency_scaling", "delimiter_heads": len(delimiter_heads), "results": results}


# =========================================================================
# Experiment 3: Single-head importance ranking
# =========================================================================

def run_head_ranking(model, tok, delimiter_heads, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT 3: SINGLE-HEAD IMPORTANCE RANKING")
    print("=" * 90)

    texts = [gen_gcf_generic(50)]
    baseline = compute_ppl(model, tok, texts, device)
    print(f"Baseline GCF PPL: {baseline:.1f}")

    rankings = []
    for l, h, excess in delimiter_heads:
        model_abl = copy.deepcopy(model)
        model_abl.to(device)
        ablate_heads(model_abl, [(l, h)])
        ppl = compute_ppl(model_abl, tok, texts, device)
        delta = ((ppl - baseline) / baseline) * 100
        rankings.append({"layer": l, "head": h, "excess": round(excess, 4), "delta_pct": round(delta, 2)})
        del model_abl; gc.collect(); torch.cuda.empty_cache()

    rankings.sort(key=lambda x: x["delta_pct"], reverse=True)

    hurt = sum(1 for r in rankings if r["delta_pct"] > 0)
    help_count = sum(1 for r in rankings if r["delta_pct"] < 0)
    total_positive = sum(r["delta_pct"] for r in rankings if r["delta_pct"] > 0)
    top5_positive = sum(r["delta_pct"] for r in rankings[:5] if r["delta_pct"] > 0)
    top5_frac = (top5_positive / total_positive * 100) if total_positive > 0 else 0

    print(f"\n{hurt} of {len(rankings)} heads hurt GCF when removed")
    print(f"{help_count} of {len(rankings)} heads help GCF when removed")
    print(f"Top 5 account for {top5_frac:.0f}% of total degradation")

    print(f"\nTop 10 most important heads:")
    for r in rankings[:10]:
        print(f"  L{r['layer']}H{r['head']} (excess {r['excess']:.3f}): {r['delta_pct']:+.2f}%")

    return {"experiment": "head_ranking", "baseline_ppl": round(baseline, 2),
            "hurt_count": hurt, "help_count": help_count,
            "top5_fraction_pct": round(top5_frac, 1), "rankings": rankings}


# =========================================================================
# Experiment 4: Attention pattern analysis
# =========================================================================

def run_attention_patterns(model, tok, delimiter_heads, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT 4: ATTENTION PATTERN ANALYSIS")
    print("=" * 90)

    top5 = delimiter_heads[:5]
    results = {}

    for fmt_name, text in [("gcf_generic", gen_gcf_generic(50)), ("json", gen_json(50))]:
        ids = tok.encode(text).ids[:512]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))
        content_positions = set(range(len(ids))) - delim_positions

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        fmt_results = {"delim_positions": len(delim_positions), "content_positions": len(content_positions), "heads": []}
        print(f"\n{fmt_name} ({len(ids)} tokens: {len(delim_positions)} delimiter, {len(content_positions)} content)")
        print(f"{'Head':<12} {'d->d':>8} {'d->c':>8} {'c->d':>8} {'c->c':>8}")
        print("-" * 48)

        for l, h, excess in top5:
            w = outputs.attentions[l][0, h].float().cpu()
            dd = sum(w[d, d2].item() for d in delim_positions for d2 in delim_positions) / max(len(delim_positions), 1)
            dc = sum(w[d, c].item() for d in delim_positions for c in content_positions) / max(len(delim_positions), 1)
            cd = sum(w[c, d].item() for c in content_positions for d in delim_positions) / max(len(content_positions), 1)
            cc = sum(w[c, c2].item() for c in content_positions for c2 in content_positions) / max(len(content_positions), 1)
            total_d = dd + dc
            total_c = cd + cc
            dd_n = dd / max(total_d, 1e-10)
            dc_n = dc / max(total_d, 1e-10)
            cd_n = cd / max(total_c, 1e-10)
            cc_n = cc / max(total_c, 1e-10)

            head_data = {"layer": l, "head": h, "dd": round(dd_n, 3), "dc": round(dc_n, 3),
                        "cd": round(cd_n, 3), "cc": round(cc_n, 3)}
            fmt_results["heads"].append(head_data)
            print(f"L{l}H{h} ({excess:.2f})  {dd_n:>8.3f} {dc_n:>8.3f} {cd_n:>8.3f} {cc_n:>8.3f}")

        results[fmt_name] = fmt_results
        del outputs; gc.collect(); torch.cuda.empty_cache()

    return {"experiment": "attention_patterns", "results": results}


# =========================================================================
# Experiment 5: Emergence timing
# =========================================================================

def run_emergence(tok, device, checkpoint_dir, size):
    print("\n" + "=" * 90)
    print("EXPERIMENT 5: EMERGENCE TIMING")
    print("=" * 90)

    checkpoint_dir = Path(checkpoint_dir)
    steps = sorted([int(d.name.split("-")[1]) for d in checkpoint_dir.glob("step-*") if d.is_dir()])
    print(f"Checkpoints found: {steps}")

    results = []
    for step in steps:
        cp_path = str(checkpoint_dir / f"step-{step}" / "checkpoint.pt")
        print(f"\nStep {step}:", flush=True)
        try:
            model, _ = load_model(cp_path, size, str(tok))
            model.to(device)
            # Need tokenizer object, not path
            from tokenizers import Tokenizer
            tok_obj = Tokenizer.from_file(str(tok))
            heads = identify_delimiter_heads(model, tok_obj, device, 0.15)

            # Concentration: fraction of total excess in top 10%
            if heads:
                top_n = max(1, len(heads) // 10)
                total_excess = sum(s for _, _, s in heads)
                top_excess = sum(s for _, _, s in heads[:top_n])
                concentration = (top_excess / max(total_excess, 1e-10)) * 100
            else:
                concentration = 0

            entry = {"step": step, "heads": len(heads), "concentration": round(concentration, 1),
                     "top_excess": round(heads[0][2], 4) if heads else 0}
            results.append(entry)
            print(f"  Heads: {len(heads)}, Concentration: {concentration:.1f}%")

            del model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"step": step, "error": str(e)})

    return {"experiment": "emergence", "results": results}


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Llama ablation experiments")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--size", default="410m-llama")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None, help="Dir with step-N folders for emergence")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    metadata = {
        "experiment": "llama_ablation_combined",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("LLAMA ABLATION EXPERIMENTS (Run-003 Stage 1)")
    print("=" * 90)
    print(f"Device: {device}")
    if "gpu_name" in metadata:
        print(f"GPU: {metadata['gpu_name']}")

    # Load model
    print("\nLoading model...")
    model, tok = load_model(args.checkpoint, args.size, args.tokenizer)
    model.to(device)

    # Identify heads
    print("\nIdentifying delimiter heads (threshold 0.15)...")
    delimiter_heads = identify_delimiter_heads(model, tok, device, 0.15)
    print(f"Found {len(delimiter_heads)} delimiter heads")

    results = {"metadata": metadata, "delimiter_heads_count": len(delimiter_heads)}

    # 1. Layer-wise
    results["layer_wise"] = run_layer_wise(model, tok, delimiter_heads, device)

    # 2. Sufficiency
    results["sufficiency"] = run_sufficiency(model, tok, delimiter_heads, device)

    # 3. Head ranking
    results["head_ranking"] = run_head_ranking(model, tok, delimiter_heads, device)

    # 4. Attention patterns
    results["attention_patterns"] = run_attention_patterns(model, tok, delimiter_heads, device)

    # 5. Emergence (if checkpoint dir provided)
    if args.checkpoint_dir:
        del model; gc.collect(); torch.cuda.empty_cache()
        results["emergence"] = run_emergence(args.tokenizer, device, args.checkpoint_dir, args.size)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nResults saved to {args.output}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Delimiter heads: {results['delimiter_heads_count']}")
    if "layer_wise" in results:
        late = results["layer_wise"]["groups"].get("late (16-23)", {})
        print(f"Layer-wise: late layers GCF delta = {late.get('deltas', {}).get('gcf_generic', '?')}%")
    if "sufficiency" in results:
        for r in results["sufficiency"]["results"]:
            print(f"Sufficiency {r['size']} rows: delim {r['delim_delta']:+.1f}% vs random {r['random_delta_mean']:+.1f}% -> {r['sufficient']}")
    if "head_ranking" in results:
        print(f"Head ranking: top 5 = {results['head_ranking']['top5_fraction_pct']}% of effect")


if __name__ == "__main__":
    main()
