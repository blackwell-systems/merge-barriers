#!/usr/bin/env python3
"""
Production model probing: count delimiter-specialized heads in real models.

Downloads production models from HF, feeds structured data, counts delimiter
heads using the same method as the ablation experiments. Correlates head
counts with comprehension scores from the GCF eval suite.

Usage:
  python eval_production_probing.py \
    --models "microsoft/phi-2,mistralai/Mistral-7B-v0.3" \
    --output probing-results.json

  # Include our controlled models as reference points
  python eval_production_probing.py \
    --models "microsoft/phi-2,mistralai/Mistral-7B-v0.3" \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoint-b.pt --tokenizer-b standard-64k.json \
    --output probing-results.json
"""

import argparse
import datetime
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')


# =========================================================================
# Test data (same as ablation, using trained formats for identification)
# =========================================================================

GCF_TEXT = """## orders [30]{orderId,customer,status,total}
ORD-00001|Alice Chen|pending|29.97
ORD-00002|Bob Smith|processing|42.47
ORD-00003|Carla Rodriguez|shipped|54.97
ORD-00004|David Park|delivered|67.47
ORD-00005|Eva Johansson|cancelled|79.97
ORD-00006|Fiona Grant|pending|92.47
ORD-00007|George Wu|processing|104.97
ORD-00008|Hannah Lee|shipped|117.47
ORD-00009|Ivan Petrov|delivered|129.97
ORD-00010|Julia Santos|cancelled|142.47
ORD-00011|Alice Chen|pending|154.97
ORD-00012|Bob Smith|processing|167.47
ORD-00013|Carla Rodriguez|shipped|179.97
ORD-00014|David Park|delivered|192.47
ORD-00015|Eva Johansson|cancelled|204.97
ORD-00016|Fiona Grant|pending|217.47
ORD-00017|George Wu|processing|229.97
ORD-00018|Hannah Lee|shipped|242.47
ORD-00019|Ivan Petrov|delivered|254.97
ORD-00020|Julia Santos|cancelled|267.47
ORD-00021|Alice Chen|pending|279.97
ORD-00022|Bob Smith|processing|292.47
ORD-00023|Carla Rodriguez|shipped|304.97
ORD-00024|David Park|delivered|317.47
ORD-00025|Eva Johansson|cancelled|329.97
ORD-00026|Fiona Grant|pending|342.47
ORD-00027|George Wu|processing|354.97
ORD-00028|Hannah Lee|shipped|367.47
ORD-00029|Ivan Petrov|delivered|379.97
ORD-00030|Julia Santos|cancelled|392.47"""

JSON_TEXT = json.dumps({"orders": [
    {"orderId": f"ORD-{i+1:05d}", "customer": ["Alice","Bob","Carla","David","Eva"][i%5],
     "status": ["pending","processing","shipped","delivered","cancelled"][i%5],
     "total": round(29.97 + i * 12.50, 2)}
    for i in range(30)
]}, indent=2)


# =========================================================================
# Comprehension scores from GCF eval suite (flatten experiment)
# =========================================================================

COMPREHENSION_SCORES = {
    "NousResearch/Meta-Llama-3.1-8B": {"gcf": 65.4, "json": 58.3, "source": "flatten experiment, Llama 3.1 8B"},
    "mistralai/Mistral-7B-v0.3": {"gcf": 64.6, "json": 63.6, "source": "flatten experiment, Mistral Small proxy"},
    "microsoft/phi-2": {"gcf": None, "json": None, "source": "not yet evaluated"},
    "Qwen/Qwen2.5-7B": {"gcf": None, "json": None, "source": "not yet evaluated"},
    # Reference models (not probed, PPL only)
    "structok-410m (Model A)": {"gcf": None, "json": None, "source": "run-002, PPL only"},
    "standard-410m (Model B)": {"gcf": None, "json": None, "source": "run-002, PPL only"},
}

MODEL_CONFIGS_410M = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
}


# =========================================================================
# Helpers
# =========================================================================

def is_delimiter_char(c):
    return c in BARRIER_CHARS


def count_delimiter_heads_hf(model_name, device, max_seq_len=256):
    """Load a HF model, count delimiter heads."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    n_layers = model.config.num_hidden_layers
    n_heads = getattr(model.config, "num_attention_heads", 32)
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    total_heads = n_layers * n_heads

    print(f"  Architecture: {n_layers} layers, {n_heads} query heads, {n_kv_heads} KV heads, {total_heads} total", flush=True)

    # Feed test texts
    test_texts = [GCF_TEXT, JSON_TEXT]
    head_scores = {}

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = inputs["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # Classify delimiter positions using the model's own tokenizer
        delim_positions = set()
        for pos in range(seq_len):
            token_id = input_ids[0, pos].item()
            token_str = tokenizer.decode([token_id])
            if any(is_delimiter_char(c) for c in token_str):
                delim_positions.add(pos)

        print(f"  Text: {seq_len} tokens, {len(delim_positions)} delimiter positions ({len(delim_positions)/seq_len*100:.0f}%)", flush=True)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        # Base rate: fraction of positions that are delimiters
        base_rate = len(delim_positions) / max(seq_len, 1)

        for layer_idx, attn in enumerate(outputs.attentions):
            # attn shape: (batch, n_heads, seq_len, seq_len)
            actual_heads = attn.shape[1]
            for head_idx in range(actual_heads):
                attn_weights = attn[0, head_idx].float().cpu()

                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(seq_len))
                raw_score = delim_attn / max(total_attn, 1e-10)

                # Excess delimiter attention: how much more than chance
                excess = raw_score - base_rate

                key = (layer_idx, head_idx)
                if key not in head_scores:
                    head_scores[key] = {"raw": [], "excess": [], "base_rates": []}
                head_scores[key]["raw"].append(raw_score)
                head_scores[key]["excess"].append(excess)
                head_scores[key]["base_rates"].append(base_rate)

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Count using EXCESS delimiter attention (above chance)
    results_by_threshold = {}
    top_heads = []
    avg_base_rate = 0.0
    n_base = 0

    for (l, h), data in head_scores.items():
        avg_excess = sum(data["excess"]) / len(data["excess"])
        avg_raw = sum(data["raw"]) / len(data["raw"])
        top_heads.append((l, h, avg_excess, avg_raw))
        for br in data["base_rates"]:
            avg_base_rate += br
            n_base += 1

    avg_base_rate = avg_base_rate / max(n_base, 1)
    top_heads.sort(key=lambda x: x[2], reverse=True)  # sort by excess

    print(f"  Avg delimiter base rate: {avg_base_rate:.1%} (chance level)", flush=True)

    # Thresholds on EXCESS (not raw). >10% excess means the head attends
    # to delimiters 10pp more than chance.
    for threshold_excess in [0.05, 0.10, 0.15, 0.20]:
        count = sum(1 for _, _, exc, _ in top_heads if exc > threshold_excess)
        total = len(top_heads)
        label = f"excess_{int(threshold_excess*100)}pp"
        results_by_threshold[label] = {
            "count": count,
            "total": total,
            "pct": round(count / total * 100, 1),
        }

    # Also report raw >50% for comparison with ablation experiments
    raw_50_count = sum(1 for _, _, _, raw in top_heads if raw > 0.50)
    results_by_threshold["raw_50pct"] = {
        "count": raw_50_count,
        "total": len(top_heads),
        "pct": round(raw_50_count / len(top_heads) * 100, 1),
    }

    avg_excess = sum(exc for _, _, exc, _ in top_heads) / len(top_heads)

    # Cleanup
    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_query_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "total_head_positions": len(top_heads),
        "avg_base_rate": round(avg_base_rate, 4),
        "avg_excess_score": round(avg_excess, 4),
        "thresholds": results_by_threshold,
        "top_20": [{"layer": l, "head": h, "excess": round(exc, 4), "raw": round(raw, 4)} for l, h, exc, raw in top_heads[:20]],
        "comprehension": COMPREHENSION_SCORES.get(model_name, {}),
    }


def count_delimiter_heads_custom(checkpoint_path, tokenizer_path, model_name, device):
    """Count delimiter heads in our custom 410M models."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from tokenizers import Tokenizer

    print(f"\n  Loading {model_name} from {Path(checkpoint_path).name}...", flush=True)
    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS_410M.copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"
    config = GPTNeoXConfig(**cfg)
    model = GPTNeoXForCausalLM(config)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    model.to(device)

    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    total_heads = n_layers * n_heads

    print(f"  Architecture: {n_layers} layers, {n_heads} heads, {total_heads} total", flush=True)

    test_texts = [GCF_TEXT, JSON_TEXT]
    head_scores = {}

    for text in test_texts:
        ids = tok.encode(text).ids[:512]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        seq_len = len(ids)

        vocab = tok.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        delim_positions = set()
        for pos, tid in enumerate(ids):
            token_str = id_to_token.get(tid, "")
            if any(c in token_str for c in BARRIER_CHARS):
                delim_positions.add(pos)

        base_rate = len(delim_positions) / max(seq_len, 1)
        print(f"  Text: {seq_len} tokens, {len(delim_positions)} delimiter positions ({base_rate*100:.0f}%)", flush=True)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            for head_idx in range(n_heads):
                attn_weights = attn[0, head_idx].float().cpu()
                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(seq_len))
                raw_score = delim_attn / max(total_attn, 1e-10)
                excess = raw_score - base_rate
                key = (layer_idx, head_idx)
                if key not in head_scores:
                    head_scores[key] = {"raw": [], "excess": [], "base_rates": []}
                head_scores[key]["raw"].append(raw_score)
                head_scores[key]["excess"].append(excess)
                head_scores[key]["base_rates"].append(base_rate)

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    top_heads = []
    avg_base_rate = 0.0
    n_base = 0
    for (l, h), data in head_scores.items():
        avg_excess = sum(data["excess"]) / len(data["excess"])
        avg_raw = sum(data["raw"]) / len(data["raw"])
        top_heads.append((l, h, avg_excess, avg_raw))
        for br in data["base_rates"]:
            avg_base_rate += br
            n_base += 1
    avg_base_rate = avg_base_rate / max(n_base, 1)
    top_heads.sort(key=lambda x: x[2], reverse=True)

    print(f"  Avg delimiter base rate: {avg_base_rate:.1%} (chance level)", flush=True)

    results_by_threshold = {}
    for threshold_excess in [0.05, 0.10, 0.15, 0.20]:
        count = sum(1 for _, _, exc, _ in top_heads if exc > threshold_excess)
        label = f"excess_{int(threshold_excess*100)}pp"
        results_by_threshold[label] = {
            "count": count,
            "total": len(top_heads),
            "pct": round(count / len(top_heads) * 100, 1),
        }
    raw_50_count = sum(1 for _, _, _, raw in top_heads if raw > 0.50)
    results_by_threshold["raw_50pct"] = {
        "count": raw_50_count,
        "total": len(top_heads),
        "pct": round(raw_50_count / len(top_heads) * 100, 1),
    }

    avg_excess = sum(exc for _, _, exc, _ in top_heads) / len(top_heads)

    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_query_heads": n_heads,
        "n_kv_heads": n_heads,
        "total_head_positions": len(top_heads),
        "avg_base_rate": round(avg_base_rate, 4),
        "avg_excess_score": round(avg_excess, 4),
        "thresholds": results_by_threshold,
        "top_20": [{"layer": l, "head": h, "excess": round(exc, 4), "raw": round(raw, 4)} for l, h, exc, raw in top_heads[:20]],
        "comprehension": COMPREHENSION_SCORES.get(model_name, {}),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Production model probing")
    parser.add_argument("--models", required=True, help="Comma-separated HF model names")
    parser.add_argument("--checkpoint-a", default=None, help="Model A checkpoint (optional reference)")
    parser.add_argument("--tokenizer-a", default=None)
    parser.add_argument("--checkpoint-b", default=None, help="Model B checkpoint (optional reference)")
    parser.add_argument("--tokenizer-b", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length for probing (reduce if OOM)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metadata = {
        "experiment": "production_model_probing",
        "description": "Count delimiter-specialized heads in production models and correlate with comprehension scores",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "max_seq_len": args.max_seq_len,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("PRODUCTION MODEL PROBING")
    print("=" * 90)
    print(f"\nDevice: {device}")
    print(f"Max sequence length: {args.max_seq_len}")

    all_results = []

    # Probe our custom models first (reference points)
    if args.checkpoint_a and args.tokenizer_a:
        result = count_delimiter_heads_custom(args.checkpoint_a, args.tokenizer_a, "structok-410m (Model A)", device)
        all_results.append(result)
        print(f"  -> {result['thresholds']['excess_10pp']['count']} delimiter heads at >10pp excess")

    if args.checkpoint_b and args.tokenizer_b:
        result = count_delimiter_heads_custom(args.checkpoint_b, args.tokenizer_b, "standard-410m (Model B)", device)
        all_results.append(result)
        print(f"  -> {result['thresholds']['excess_10pp']['count']} delimiter heads at >10pp excess")

    # Probe production models
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in model_names:
        try:
            result = count_delimiter_heads_hf(model_name, device, args.max_seq_len)
            all_results.append(result)
            print(f"  -> {result['thresholds']['excess_10pp']['count']} delimiter heads at >10pp excess")
        except Exception as e:
            print(f"  ERROR on {model_name}: {e}", flush=True)
            all_results.append({"model": model_name, "error": str(e)})

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\n{'Model':<35} {'Heads':>6} {'Base%':>6} {'Excess>10pp':>12} {'Exc%':>6} {'Raw>50%':>8} {'GCF':>6}")
    print("-" * 86)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<35} ERROR: {r['error'][:40]}")
            continue

        total = r["total_head_positions"]
        base_rate = r.get("avg_base_rate", 0) * 100
        excess_10 = r["thresholds"].get("excess_10pp", {}).get("count", 0)
        excess_pct = r["thresholds"].get("excess_10pp", {}).get("pct", 0)
        raw_50 = r["thresholds"].get("raw_50pct", {}).get("count", 0)
        gcf_score = r.get("comprehension", {}).get("gcf")
        gcf_str = f"{gcf_score}%" if gcf_score else "N/A"

        print(f"{r['model']:<35} {total:>6} {base_rate:>5.0f}% {excess_10:>12} {excess_pct:>5.1f}% {raw_50:>8} {gcf_str:>6}")

    # Correlation analysis (if we have enough data points)
    scored_results = [(r["thresholds"]["50pct"]["pct"], r["comprehension"]["gcf"])
                      for r in all_results
                      if "error" not in r and r.get("comprehension", {}).get("gcf") is not None]

    if len(scored_results) >= 2:
        print(f"\nCorrelation data points ({len(scored_results)}):")
        for pct, score in scored_results:
            print(f"  delimiter% = {pct:.1f}%, GCF comprehension = {score}%")
    else:
        print(f"\nInsufficient data points for correlation ({len(scored_results)}). Need comprehension scores for more models.")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "models": all_results,
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
            (args.output, "logs/run-002-ablation/production-probing-results.json"),
        ]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/production-probing-log.txt")
            print(f"  Uploaded production-probing-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
