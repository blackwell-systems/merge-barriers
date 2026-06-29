#!/usr/bin/env python3
"""
Production model probing v2: revised methodology.

Instead of threshold counting (which doesn't transfer across model sizes),
measures delimiter specialization through:

1. Distribution shape: histogram of per-head delimiter excess scores
2. Top-K analysis: average delimiter attention of top 10 heads (scale-invariant)
3. Excess concentration ratio: what fraction of total excess is in the top 10% of heads
4. GCF-only probing: avoids JSON's high delimiter density
5. Normalized metrics: per-head values, not raw counts

Usage:
  python eval_production_probing_v2.py \
    --models "microsoft/phi-2,mistralai/Mistral-7B-v0.3" \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoint-b.pt --tokenizer-b standard-64k.json \
    --output probing-v2-results.json
"""

import argparse
import datetime
import gc
import json
from pathlib import Path

import torch


BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')

MODEL_CONFIGS_410M = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
}

# GCF-only test data (20-35% delimiters, avoids JSON's 50%+ density)
GCF_TEXTS = [
    """## orders [30]{orderId,customer,status,total}
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
ORD-00030|Julia Santos|cancelled|392.47""",

    """GCF profile=graph tool=context symbols=20 edges=15
## targets
@0 fn pkg/auth.Validate0 0.95 lsp_resolved
@1 type pkg/server.Process1 0.92 ast_inferred
@2 method pkg/db.Handle2 0.89 structural
@3 iface pkg/cache.Create3 0.86 lsp_resolved
@4 fn pkg/config.Update4 0.83 ast_inferred
@5 type pkg/auth.Delete5 0.80 structural
@6 method pkg/server.Get6 0.77 lsp_resolved
## related
@7 iface pkg/db.Set7 0.74 ast_inferred
@8 fn pkg/cache.Check8 0.71 structural
@9 type pkg/config.Build9 0.68 lsp_resolved
@10 method pkg/auth.Parse10 0.65 ast_inferred
@11 iface pkg/server.Format11 0.62 structural
@12 fn pkg/db.Load12 0.59 lsp_resolved
@13 type pkg/cache.Save13 0.56 ast_inferred
## extended
@14 method pkg/config.Init14 0.53 structural
@15 iface pkg/auth.Close15 0.50 lsp_resolved
@16 fn pkg/server.Open16 0.47 ast_inferred
@17 type pkg/db.Read17 0.44 structural
@18 method pkg/cache.Write18 0.41 lsp_resolved
@19 iface pkg/config.Flush19 0.38 ast_inferred
## edges [15]
@0<@1 calls
@2<@3 imports
@4<@5 implements
@6<@7 references
@8<@9 calls
@10<@11 imports
@12<@13 implements
@14<@15 references
@1<@0 calls
@3<@2 imports
@5<@4 implements
@7<@6 references
@9<@8 calls
@11<@10 imports
@13<@12 implements""",
]

COMPREHENSION_SCORES = {
    "NousResearch/Meta-Llama-3.1-8B": {"gcf": 65.4, "json": 58.3},
    "mistralai/Mistral-7B-v0.3": {"gcf": 64.6, "json": 63.6},
    "microsoft/phi-2": {"gcf": None, "json": None},
}


def is_delimiter_char(c):
    return c in BARRIER_CHARS


def probe_model_hf(model_name, device, max_seq_len=256):
    """Probe a HF model for delimiter specialization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval().to(device)

    n_layers = model.config.num_hidden_layers
    n_heads = getattr(model.config, "num_attention_heads", 32)
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    total_heads = n_layers * n_heads

    print(f"  {n_layers} layers, {n_heads} query heads/layer, {total_heads} total", flush=True)

    scores = collect_scores(model, tokenizer, None, device, max_seq_len, use_hf_tokenizer=True)

    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return analyze_scores(scores, model_name, n_layers, n_heads, n_kv_heads, total_heads)


def probe_model_custom(checkpoint_path, tokenizer_path, model_name, device):
    """Probe our custom 410M model."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from tokenizers import Tokenizer

    print(f"\n  Loading {model_name}...", flush=True)
    tok = Tokenizer.from_file(tokenizer_path)
    cfg = MODEL_CONFIGS_410M.copy()
    cfg["vocab_size"] = tok.get_vocab_size()
    cfg["_attn_implementation"] = "eager"
    config = GPTNeoXConfig(**cfg)
    model = GPTNeoXForCausalLM(config)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval().to(device)

    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    total_heads = n_layers * n_heads

    print(f"  {n_layers} layers, {n_heads} heads/layer, {total_heads} total", flush=True)

    scores = collect_scores(model, None, tok, device, 512, use_hf_tokenizer=False)

    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return analyze_scores(scores, model_name, n_layers, n_heads, n_heads, total_heads)


def collect_scores(model, hf_tokenizer, custom_tokenizer, device, max_seq_len, use_hf_tokenizer):
    """Collect per-head delimiter excess scores from GCF-only test data."""
    head_excess_scores = {}

    for text in GCF_TEXTS:
        if use_hf_tokenizer:
            inputs = hf_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            seq_len = input_ids.shape[1]

            delim_positions = set()
            for pos in range(seq_len):
                token_str = hf_tokenizer.decode([input_ids[0, pos].item()])
                if any(is_delimiter_char(c) for c in token_str):
                    delim_positions.add(pos)
        else:
            ids = custom_tokenizer.encode(text).ids[:max_seq_len]
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            seq_len = len(ids)

            vocab = custom_tokenizer.get_vocab()
            id_to_token = {v: k for k, v in vocab.items()}
            delim_positions = set()
            for pos, tid in enumerate(ids):
                if any(c in id_to_token.get(tid, "") for c in BARRIER_CHARS):
                    delim_positions.add(pos)

        base_rate = len(delim_positions) / max(seq_len, 1)
        print(f"  GCF text: {seq_len} tokens, {len(delim_positions)} delimiters ({base_rate:.0%} base rate)", flush=True)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            n_heads_in_layer = attn.shape[1]
            for head_idx in range(n_heads_in_layer):
                attn_weights = attn[0, head_idx].float().cpu()

                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(seq_len))
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

    # Average excess across texts
    avg_scores = {}
    for key, excesses in head_excess_scores.items():
        avg_scores[key] = sum(excesses) / len(excesses)

    return avg_scores


def analyze_scores(scores, model_name, n_layers, n_heads, n_kv_heads, total_heads):
    """Compute all five metrics from excess scores."""

    all_excess = sorted(scores.values(), reverse=True)
    n_total = len(all_excess)

    # 1. Distribution shape: histogram bins
    bins = {"<-0.10": 0, "-0.10 to -0.05": 0, "-0.05 to 0": 0,
            "0 to 0.05": 0, "0.05 to 0.10": 0, "0.10 to 0.15": 0,
            "0.15 to 0.20": 0, "0.20 to 0.25": 0, ">0.25": 0}
    for exc in all_excess:
        if exc < -0.10: bins["<-0.10"] += 1
        elif exc < -0.05: bins["-0.10 to -0.05"] += 1
        elif exc < 0: bins["-0.05 to 0"] += 1
        elif exc < 0.05: bins["0 to 0.05"] += 1
        elif exc < 0.10: bins["0.05 to 0.10"] += 1
        elif exc < 0.15: bins["0.10 to 0.15"] += 1
        elif exc < 0.20: bins["0.15 to 0.20"] += 1
        elif exc < 0.25: bins["0.20 to 0.25"] += 1
        else: bins[">0.25"] += 1

    # 2. Top-K analysis (scale-invariant)
    top_10_avg = sum(all_excess[:10]) / 10
    top_20_avg = sum(all_excess[:20]) / 20

    # 3. Excess concentration ratio
    positive_excess = [e for e in all_excess if e > 0]
    total_positive_excess = sum(positive_excess) if positive_excess else 0
    top_10pct_n = max(1, n_total // 10)
    top_10pct_excess = sum(all_excess[:top_10pct_n])
    concentration_ratio = top_10pct_excess / max(total_positive_excess, 1e-10)

    # 4. Mean excess per head (normalized)
    mean_excess = sum(all_excess) / n_total
    median_excess = all_excess[n_total // 2]

    # 5. Std of excess (spread)
    std_excess = (sum((e - mean_excess) ** 2 for e in all_excess) / n_total) ** 0.5

    # Max excess (how specialized is the most specialized head?)
    max_excess = all_excess[0]
    min_excess = all_excess[-1]

    # Bimodality: gap between top 10% mean and bottom 90% mean
    top_10pct_mean = sum(all_excess[:top_10pct_n]) / top_10pct_n
    bottom_90pct_mean = sum(all_excess[top_10pct_n:]) / max(n_total - top_10pct_n, 1)
    bimodality_gap = top_10pct_mean - bottom_90pct_mean

    print(f"  Top-10 avg excess: {top_10_avg:.4f}")
    print(f"  Mean excess: {mean_excess:.4f}, std: {std_excess:.4f}")
    print(f"  Max excess: {max_excess:.4f}, min: {min_excess:.4f}")
    print(f"  Concentration (top 10% share of positive excess): {concentration_ratio:.1%}")
    print(f"  Bimodality gap (top 10% mean - bottom 90% mean): {bimodality_gap:.4f}")

    return {
        "model": model_name,
        "n_layers": n_layers,
        "n_query_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "total_head_positions": n_total,
        "top_10_avg_excess": round(top_10_avg, 4),
        "top_20_avg_excess": round(top_20_avg, 4),
        "mean_excess": round(mean_excess, 4),
        "median_excess": round(median_excess, 4),
        "std_excess": round(std_excess, 4),
        "max_excess": round(max_excess, 4),
        "min_excess": round(min_excess, 4),
        "concentration_ratio": round(concentration_ratio, 4),
        "bimodality_gap": round(bimodality_gap, 4),
        "distribution": bins,
        "comprehension": COMPREHENSION_SCORES.get(model_name, {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Production model probing v2")
    parser.add_argument("--models", required=True)
    parser.add_argument("--checkpoint-a", default=None)
    parser.add_argument("--tokenizer-a", default=None)
    parser.add_argument("--checkpoint-b", default=None)
    parser.add_argument("--tokenizer-b", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metadata = {
        "experiment": "production_model_probing_v2",
        "description": "Delimiter specialization via distribution shape, top-K, and concentration ratio (GCF-only probing)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "method": "GCF-only text, excess = raw_delimiter_attention - base_rate, scale-invariant metrics",
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("PRODUCTION MODEL PROBING v2")
    print("=" * 90)
    print(f"\nDevice: {device}")
    print(f"Method: GCF-only probing, excess scores, scale-invariant metrics")

    all_results = []

    if args.checkpoint_a and args.tokenizer_a:
        r = probe_model_custom(args.checkpoint_a, args.tokenizer_a, "structok-410m (Model A)", device)
        all_results.append(r)

    if args.checkpoint_b and args.tokenizer_b:
        r = probe_model_custom(args.checkpoint_b, args.tokenizer_b, "standard-410m (Model B)", device)
        all_results.append(r)

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            r = probe_model_hf(model_name, device, args.max_seq_len)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_results.append({"model": model_name, "error": str(e)})

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY: Scale-invariant delimiter specialization metrics")
    print("=" * 90)

    print(f"\n{'Model':<30} {'Heads':>6} {'Top10 exc':>10} {'Mean exc':>9} {'Concen':>7} {'Bimodal':>8} {'GCF':>6}")
    print("-" * 82)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<30} ERROR")
            continue
        gcf = r.get("comprehension", {}).get("gcf")
        gcf_str = f"{gcf}%" if gcf else "N/A"
        print(f"{r['model']:<30} {r['total_head_positions']:>6} {r['top_10_avg_excess']:>10.4f} {r['mean_excess']:>9.4f} {r['concentration_ratio']:>6.1%} {r['bimodality_gap']:>8.4f} {gcf_str:>6}")

    # Distribution comparison
    print(f"\nExcess score distribution (fraction of heads in each bin):")
    print(f"{'Bin':<18}", end="")
    for r in all_results:
        if "error" not in r:
            short = r["model"][:15]
            print(f" {short:>15}", end="")
    print()
    print("-" * (18 + 16 * len([r for r in all_results if "error" not in r])))

    for bin_name in ["<-0.10", "-0.10 to -0.05", "-0.05 to 0", "0 to 0.05", "0.05 to 0.10", "0.10 to 0.15", "0.15 to 0.20", "0.20 to 0.25", ">0.25"]:
        print(f"{bin_name:<18}", end="")
        for r in all_results:
            if "error" not in r:
                count = r["distribution"].get(bin_name, 0)
                pct = count / r["total_head_positions"] * 100
                print(f" {pct:>14.1f}%", end="")
        print()

    # Save
    if args.output:
        out = {"metadata": metadata, "models": all_results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # R2
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
        for local, key in [(args.output, "logs/run-002-ablation/production-probing-v2-results.json")]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/production-probing-v2-log.txt")
            print(f"  Uploaded probing-v2-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
