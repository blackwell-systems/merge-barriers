#!/usr/bin/env python3
"""
Head ablation v3: three follow-up experiments.

1. Cross-format transfer ablation: does ablating delimiter heads kill
   the TOON advantage (2.3x, never in training)?
2. Single-head importance ranking: which individual heads matter most?
3. Threshold sensitivity: does the causal effect hold at 40% and 60%?

Usage:
  python eval_ablation_v3.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --output ablation-v3-results.json
"""

import argparse
import gc
import json
import math
import copy
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
# Test data (includes TOON for cross-format transfer)
# =========================================================================

def generate_gcf_generic(n_rows=50):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson",
             "Fiona Grant", "George Wu", "Hannah Lee", "Ivan Petrov", "Julia Santos"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n_rows}]{{orderId,customer,status,total}}"]
    for i in range(n_rows):
        lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def generate_gcf_graph(n_symbols=30, n_edges=20):
    packages = ["pkg/auth", "pkg/server", "pkg/db", "pkg/cache", "pkg/config"]
    names = ["Validate", "Process", "Handle", "Create", "Update", "Delete", "Get", "Set", "Check", "Build"]
    kinds = ["fn", "type", "method", "iface"]
    provs = ["lsp_resolved", "ast_inferred", "structural"]

    lines = [f"GCF profile=graph tool=context_for_task symbols={n_symbols} edges={n_edges}"]
    groups = {"targets": [], "related": [], "extended": []}
    for i in range(n_symbols):
        kind = kinds[i % len(kinds)]
        prov = provs[i % len(provs)]
        score = round(max(0.10, 0.95 - i * 0.02), 2)
        g = "targets" if i < n_symbols // 3 else "related" if i < 2 * n_symbols // 3 else "extended"
        groups[g].append(f"@{i} {kind} {packages[i % len(packages)]}.{names[i % len(names)]}{i} {score} {prov}")

    for name, syms in groups.items():
        if syms:
            lines.append(f"## {name}")
            lines.extend(syms)

    edge_types = ["calls", "imports", "implements", "references"]
    lines.append(f"## edges [{n_edges}]")
    for i in range(n_edges):
        lines.append(f"@{(i*3) % n_symbols}<@{(i*3+1) % n_symbols} {edge_types[i % len(edge_types)]}")
    return "\n".join(lines)


def generate_json(n_rows=50):
    names = ["Alice", "Bob", "Carla", "David", "Eva"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    records = [{"orderId": f"ORD-{i+1:05d}", "customer": names[i % len(names)],
                "status": statuses[i % len(statuses)], "total": round(29.97 + i * 12.50, 2)}
               for i in range(n_rows)]
    return json.dumps({"orders": records}, indent=2)


def generate_toon(n_rows=50):
    """Generate TOON-style tab-separated data (never in training corpus)."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = ["orderId\tcustomer\tstatus\ttotal"]
    for i in range(n_rows):
        lines.append(f"ORD-{i+1:05d}\t{names[i % len(names)]}\t{statuses[i % len(statuses)]}\t{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def generate_csv(n_rows=50):
    """Generate CSV data."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = ["orderId,customer,status,total"]
    for i in range(n_rows):
        lines.append(f"ORD-{i+1:05d},{names[i % len(names)]},{statuses[i % len(statuses)]},{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


NL_TEXTS = [
    "The architecture of modern distributed systems has evolved significantly over the past decade. "
    "Microservices replaced monolithic applications, bringing benefits like independent deployment and "
    "technology diversity, but also introducing complexity in service discovery and distributed tracing.",
]

FORMAT_TEXTS = {}


def build_test_data():
    global FORMAT_TEXTS
    FORMAT_TEXTS = {
        "gcf_generic": [generate_gcf_generic(50), generate_gcf_generic(30)],
        "gcf_graph": [generate_gcf_graph(30, 20)],
        "json": [generate_json(50), generate_json(30)],
        "toon": [generate_toon(50), generate_toon(30)],
        "csv": [generate_csv(50), generate_csv(30)],
        "nl": NL_TEXTS,
    }


# =========================================================================
# Model loading and helpers
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
    test_texts = [FORMAT_TEXTS["gcf_generic"][0], FORMAT_TEXTS["gcf_graph"][0],
                  FORMAT_TEXTS["json"][0], FORMAT_TEXTS["toon"][0]]

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


def ablate_heads(model, heads_to_ablate):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads_to_ablate:
        dense = model.gpt_neox.layers[layer_idx].attention.dense
        start = head_idx * head_dim
        end = start + head_dim
        dense.weight.data[:, start:end] = 0.0


def measure_all_formats(model, tok, device):
    result = {}
    for fmt, texts in FORMAT_TEXTS.items():
        result[fmt] = compute_ppl(model, tok, texts, device)
    return result


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Head ablation v3: follow-up experiments")
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
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    import datetime, platform
    metadata = {
        "experiment": "head_ablation_v3",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("HEAD ABLATION v3: FOLLOW-UP EXPERIMENTS")
    print("=" * 90)
    print(f"\nTimestamp: {metadata['timestamp_utc']}")
    print(f"Device: {device}")

    build_test_data()

    print("\nTest data:")
    for fmt, texts in FORMAT_TEXTS.items():
        total_chars = sum(len(t) for t in texts)
        print(f"  {fmt}: {len(texts)} texts, {total_chars:,} chars")

    print("\nLoading Model A (merge barriers)...")
    model_a, tok_a, step = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    # Baseline
    print("\n" + "=" * 90)
    print("BASELINES")
    print("=" * 90)
    baseline = measure_all_formats(model_a, tok_a, device)
    for fmt, ppl in baseline.items():
        print(f"  {fmt}: {ppl:.2f}")

    # Identify heads at default threshold (50%)
    delimiter_heads_50, all_scores = identify_delimiter_heads(model_a, tok_a, device, 0.5)
    n_total = model_a.config.num_hidden_layers * model_a.config.num_attention_heads
    print(f"\nDelimiter heads at 50% threshold: {len(delimiter_heads_50)} / {n_total}")

    # =====================================================================
    # EXPERIMENT 1: Cross-format transfer ablation
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: Cross-format transfer ablation")
    print("=" * 90)
    print("\nQuestion: does ablating delimiter heads kill the TOON/CSV advantage?")
    print("TOON and CSV were never in training. If delimiter heads drive cross-format")
    print("transfer, ablating them should disproportionately hurt TOON/CSV.")

    model_copy = copy.deepcopy(model_a)
    model_copy.to(device)
    ablate_heads(model_copy, [(l, h) for l, h, _ in delimiter_heads_50])
    ablated = measure_all_formats(model_copy, tok_a, device)

    print(f"\n{'Format':<14} {'Baseline':>10} {'Ablated':>10} {'Delta':>8} {'In training?':>14}")
    print("-" * 62)
    training_status = {
        "gcf_generic": "yes (8%)",
        "gcf_graph": "yes (8%)",
        "json": "yes (corpus)",
        "toon": "NO",
        "csv": "NO",
        "nl": "yes (corpus)",
    }
    exp1_results = {}
    for fmt in FORMAT_TEXTS:
        base = baseline[fmt]
        abl = ablated[fmt]
        delta = ((abl - base) / base) * 100
        exp1_results[fmt] = {"baseline": round(base, 2), "ablated": round(abl, 2), "delta_pct": round(delta, 1)}
        print(f"{fmt:<14} {base:>10.1f} {abl:>10.1f} {delta:>+7.1f}% {training_status.get(fmt, '?'):>14}")

    del model_copy
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # =====================================================================
    # EXPERIMENT 2: Single-head importance ranking
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: Single-head importance ranking")
    print("=" * 90)
    print(f"\nAblating each of {len(delimiter_heads_50)} delimiter heads individually.")
    print("Measures GCF generic PPL change per head to find the most important ones.")

    head_importance = []
    for idx, (layer, head, score) in enumerate(delimiter_heads_50):
        model_copy = copy.deepcopy(model_a)
        model_copy.to(device)
        ablate_heads(model_copy, [(layer, head)])

        gcf_ppl = compute_ppl(model_copy, tok_a, FORMAT_TEXTS["gcf_generic"], device)
        gcf_delta = ((gcf_ppl - baseline["gcf_generic"]) / baseline["gcf_generic"]) * 100

        json_ppl = compute_ppl(model_copy, tok_a, FORMAT_TEXTS["json"], device)
        json_delta = ((json_ppl - baseline["json"]) / baseline["json"]) * 100

        toon_ppl = compute_ppl(model_copy, tok_a, FORMAT_TEXTS["toon"], device)
        toon_delta = ((toon_ppl - baseline["toon"]) / baseline["toon"]) * 100

        head_importance.append({
            "layer": layer, "head": head, "delimiter_score": round(score, 4),
            "gcf_delta_pct": round(gcf_delta, 1),
            "json_delta_pct": round(json_delta, 1),
            "toon_delta_pct": round(toon_delta, 1),
        })

        if idx < 10 or gcf_delta > 5 or gcf_delta < -5:
            print(f"  L{layer:>2}H{head:>2} (score {score:.1%}): GCF {gcf_delta:+.1f}%  JSON {json_delta:+.1f}%  TOON {toon_delta:+.1f}%")

        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Sort by GCF impact (most damaging first)
    head_importance.sort(key=lambda x: x["gcf_delta_pct"], reverse=True)

    print(f"\nTop 10 most important for GCF (removing hurts most):")
    for h in head_importance[:10]:
        print(f"  L{h['layer']:>2}H{h['head']:>2}: GCF {h['gcf_delta_pct']:+.1f}%  JSON {h['json_delta_pct']:+.1f}%  TOON {h['toon_delta_pct']:+.1f}%")

    print(f"\nTop 5 least important (removing helps most):")
    for h in head_importance[-5:]:
        print(f"  L{h['layer']:>2}H{h['head']:>2}: GCF {h['gcf_delta_pct']:+.1f}%  JSON {h['json_delta_pct']:+.1f}%  TOON {h['toon_delta_pct']:+.1f}%")

    # How concentrated is the effect?
    positive_impact = [h for h in head_importance if h["gcf_delta_pct"] > 0]
    negative_impact = [h for h in head_importance if h["gcf_delta_pct"] < 0]
    print(f"\n  Heads that hurt GCF when removed: {len(positive_impact)} / {len(head_importance)}")
    print(f"  Heads that help GCF when removed: {len(negative_impact)} / {len(head_importance)}")
    if positive_impact:
        avg_hurt = sum(h["gcf_delta_pct"] for h in positive_impact) / len(positive_impact)
        print(f"  Avg degradation from important heads: +{avg_hurt:.1f}%")

    # =====================================================================
    # EXPERIMENT 3: Threshold sensitivity
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 3: Threshold sensitivity")
    print("=" * 90)
    print("\nDoes the causal effect hold at different delimiter-majority thresholds?")

    thresholds = [0.40, 0.50, 0.60, 0.70]
    exp3_results = []

    for threshold in thresholds:
        heads_at_threshold = [(l, h, s) for (l, h), s in all_scores.items() if s > threshold]
        heads_at_threshold.sort(key=lambda x: x[2], reverse=True)
        n_heads = len(heads_at_threshold)

        if n_heads == 0:
            print(f"\n  Threshold {threshold:.0%}: 0 heads (skipping)")
            exp3_results.append({"threshold": threshold, "n_heads": 0})
            continue

        model_copy = copy.deepcopy(model_a)
        model_copy.to(device)
        ablate_heads(model_copy, [(l, h) for l, h, _ in heads_at_threshold])
        ppls = measure_all_formats(model_copy, tok_a, device)

        result = {"threshold": threshold, "n_heads": n_heads}
        line = f"  Threshold {threshold:.0%}: {n_heads:>3} heads |"
        for fmt in ["gcf_generic", "json", "toon", "nl"]:
            delta = ((ppls[fmt] - baseline[fmt]) / baseline[fmt]) * 100
            result[f"{fmt}_delta_pct"] = round(delta, 1)
            line += f"  {fmt} {delta:+.0f}%"
        print(line)
        exp3_results.append(result)

        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print("\n1. Cross-format transfer:")
    toon_delta = exp1_results.get("toon", {}).get("delta_pct", 0)
    csv_delta = exp1_results.get("csv", {}).get("delta_pct", 0)
    gcf_delta = exp1_results.get("gcf_generic", {}).get("delta_pct", 0)
    json_delta = exp1_results.get("json", {}).get("delta_pct", 0)
    print(f"   TOON (never trained): {toon_delta:+.1f}%")
    print(f"   CSV (never trained):  {csv_delta:+.1f}%")
    print(f"   GCF (trained):        {gcf_delta:+.1f}%")
    print(f"   JSON (trained):       {json_delta:+.1f}%")
    if toon_delta > 0 and json_delta < 0:
        print("   CONFIRMED: delimiter heads drive cross-format transfer.")
    else:
        print("   Cross-format transfer not clearly delimiter-head dependent.")

    print(f"\n2. Head concentration:")
    print(f"   {len(positive_impact)} of {len(head_importance)} delimiter heads hurt GCF when removed")
    if positive_impact:
        top5_total = sum(h["gcf_delta_pct"] for h in head_importance[:5])
        all_total = sum(h["gcf_delta_pct"] for h in positive_impact)
        print(f"   Top 5 heads account for {top5_total:.0f}pp of {all_total:.0f}pp total degradation ({top5_total/max(all_total,1)*100:.0f}%)")

    print(f"\n3. Threshold sensitivity:")
    for r in exp3_results:
        if r["n_heads"] > 0:
            print(f"   {r['threshold']:.0%} ({r['n_heads']} heads): GCF {r.get('gcf_generic_delta_pct', 0):+.1f}%  TOON {r.get('toon_delta_pct', 0):+.1f}%")

    # Save locally
    if args.output:
        out = {
            "metadata": metadata,
            "baseline": {k: round(v, 4) for k, v in baseline.items()},
            "delimiter_heads_50": len(delimiter_heads_50),
            "experiment_1_cross_format": exp1_results,
            "experiment_2_head_importance": head_importance,
            "experiment_3_threshold_sensitivity": exp3_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Upload to R2
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

        r2_files = []
        if args.output and os.path.exists(args.output):
            r2_files.append((args.output, "logs/run-002-ablation/ablation-v3-results.json"))
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            r2_files.append((log_path, "logs/run-002-ablation/ablation-v3-log.txt"))

        for local, key in r2_files:
            s3.upload_file(local, "structok-training", key)
            size_kb = os.path.getsize(local) / 1024
            print(f"  Uploaded {key} ({size_kb:.0f} KB)", flush=True)
        print("R2 upload complete.", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
