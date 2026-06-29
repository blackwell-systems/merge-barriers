#!/usr/bin/env python3
"""
Delimiter head emergence timing: probe checkpoints from a training run.

Loads each checkpoint saved during training, counts delimiter heads,
and plots how specialization develops over training steps.

Usage:
  python eval_emergence.py \
    --checkpoint-dir /root/checkpoints/ \
    --tokenizer structok-64k.json \
    --output emergence-results.json

  # Or probe specific checkpoints
  python eval_emergence.py \
    --checkpoints "step-500/checkpoint.pt,step-1000/checkpoint.pt,..." \
    --tokenizer structok-64k.json \
    --output emergence-results.json
"""

import argparse
import datetime
import gc
import json
import os
from pathlib import Path

import torch


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

# GCF-only test data for probing (same as ablation v2+)
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

    """GCF profile=graph tool=context symbols=15 edges=10
## targets
@0 fn pkg/auth.Validate0 0.95 lsp_resolved
@1 type pkg/server.Process1 0.92 ast_inferred
@2 method pkg/db.Handle2 0.89 structural
@3 iface pkg/cache.Create3 0.86 lsp_resolved
@4 fn pkg/config.Update4 0.83 ast_inferred
## related
@5 type pkg/auth.Delete5 0.80 structural
@6 method pkg/server.Get6 0.77 lsp_resolved
@7 iface pkg/db.Set7 0.74 ast_inferred
@8 fn pkg/cache.Check8 0.71 structural
@9 type pkg/config.Build9 0.68 lsp_resolved
## extended
@10 method pkg/auth.Parse10 0.65 ast_inferred
@11 iface pkg/server.Format11 0.62 structural
@12 fn pkg/db.Load12 0.59 lsp_resolved
@13 type pkg/cache.Save13 0.56 ast_inferred
@14 method pkg/config.Init14 0.53 structural
## edges [10]
@0<@1 calls
@2<@3 imports
@4<@5 implements
@6<@7 references
@8<@9 calls
@1<@0 calls
@3<@2 imports
@5<@4 implements
@7<@6 references
@9<@8 calls""",
]


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
    loss = cp.get("loss", None)
    model.eval()
    return model, tok, step, loss


def probe_checkpoint(model, tok, device):
    """Count delimiter heads and compute metrics for one checkpoint."""
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_scores = {}

    for text in GCF_TEXTS:
        ids = tok.encode(text).ids[:512]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        seq_len = len(ids)

        vocab = tok.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        delim_positions = set()
        for pos, tid in enumerate(ids):
            if any(c in id_to_token.get(tid, "") for c in BARRIER_CHARS):
                delim_positions.add(pos)

        base_rate = len(delim_positions) / max(seq_len, 1)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        for layer_idx, attn in enumerate(outputs.attentions):
            for head_idx in range(n_heads):
                attn_weights = attn[0, head_idx].float().cpu()
                delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions)
                total_attn = sum(attn_weights[:, p].mean().item() for p in range(seq_len))
                raw = delim_attn / max(total_attn, 1e-10)
                excess = raw - base_rate

                key = (layer_idx, head_idx)
                if key not in head_scores:
                    head_scores[key] = {"raw": [], "excess": []}
                head_scores[key]["raw"].append(raw)
                head_scores[key]["excess"].append(excess)

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Compute metrics
    all_excess = []
    per_layer_counts = {l: 0 for l in range(n_layers)}

    for (l, h), data in head_scores.items():
        avg_raw = sum(data["raw"]) / len(data["raw"])
        avg_excess = sum(data["excess"]) / len(data["excess"])
        all_excess.append(avg_excess)
        if avg_raw > 0.5:
            per_layer_counts[l] += 1

    all_excess.sort(reverse=True)
    n_total = len(all_excess)

    # Delimiter-majority heads (raw >50%)
    raw_50_count = sum(1 for (l, h), data in head_scores.items()
                       if sum(data["raw"]) / len(data["raw"]) > 0.5)

    # Top-10 excess
    top_10_excess = sum(all_excess[:10]) / 10 if len(all_excess) >= 10 else 0

    # Concentration
    positive_excess = [e for e in all_excess if e > 0]
    total_positive = sum(positive_excess) if positive_excess else 0
    top_10pct_n = max(1, n_total // 10)
    top_10pct_sum = sum(all_excess[:top_10pct_n])
    concentration = top_10pct_sum / max(total_positive, 1e-10)

    # Mean and std
    mean_excess = sum(all_excess) / n_total
    std_excess = (sum((e - mean_excess) ** 2 for e in all_excess) / n_total) ** 0.5

    return {
        "delimiter_heads_raw50": raw_50_count,
        "top_10_excess": round(top_10_excess, 4),
        "concentration": round(concentration, 4),
        "mean_excess": round(mean_excess, 4),
        "std_excess": round(std_excess, 4),
        "max_excess": round(all_excess[0], 4) if all_excess else 0,
        "per_layer": per_layer_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Delimiter head emergence timing")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory containing step-N/ subdirectories")
    parser.add_argument("--checkpoints", default=None,
                        help="Comma-separated checkpoint paths")
    parser.add_argument("--tokenizer", required=True)
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
        "experiment": "delimiter_head_emergence",
        "description": "When during training do delimiter-specialized heads emerge?",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("DELIMITER HEAD EMERGENCE TIMING")
    print("=" * 90)
    print(f"\nDevice: {device}")

    # Find checkpoints
    checkpoint_paths = []
    if args.checkpoints:
        checkpoint_paths = [p.strip() for p in args.checkpoints.split(",")]
    elif args.checkpoint_dir:
        cp_dir = Path(args.checkpoint_dir)
        for d in sorted(cp_dir.iterdir()):
            if d.is_dir() and d.name.startswith("step-"):
                cp_file = d / "checkpoint.pt"
                if cp_file.exists():
                    checkpoint_paths.append(str(cp_file))
    else:
        print("ERROR: provide --checkpoint-dir or --checkpoints")
        return

    print(f"Found {len(checkpoint_paths)} checkpoints")

    # Probe each checkpoint
    results = []

    print(f"\n{'Step':>6} {'Loss':>8} {'Delim heads':>12} {'Top-10 exc':>11} {'Concen':>7} {'Std':>7} {'Late (16-23)':>13}")
    print("-" * 72)

    for cp_path in checkpoint_paths:
        model, tok, step, loss = load_model(cp_path, args.size, args.tokenizer)
        model.to(device)

        metrics = probe_checkpoint(model, tok, device)

        # Late-layer head count
        late_heads = sum(metrics["per_layer"].get(l, 0) for l in range(16, 24))

        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        print(f"{step:>6} {loss_str:>8} {metrics['delimiter_heads_raw50']:>12} {metrics['top_10_excess']:>11.4f} {metrics['concentration']:>6.1%} {metrics['std_excess']:>7.4f} {late_heads:>13}")

        results.append({
            "step": step,
            "loss": round(loss, 6) if loss is not None else None,
            "checkpoint": cp_path,
            **metrics,
        })

        del model
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Detect phase transition
    print("\n" + "=" * 90)
    print("EMERGENCE ANALYSIS")
    print("=" * 90)

    if len(results) >= 2:
        # Find largest jump in head count
        max_jump = 0
        jump_step = None
        for i in range(1, len(results)):
            jump = results[i]["delimiter_heads_raw50"] - results[i-1]["delimiter_heads_raw50"]
            if jump > max_jump:
                max_jump = jump
                jump_step = results[i]["step"]

        print(f"\nLargest head count jump: +{max_jump} heads at step {jump_step}")

        first_count = results[0]["delimiter_heads_raw50"]
        last_count = results[-1]["delimiter_heads_raw50"]
        total_steps = results[-1]["step"] - results[0]["step"]

        if max_jump > last_count * 0.4:
            print(f"PHASE TRANSITION: {max_jump} heads appeared in one checkpoint interval")
            print(f"  (>{last_count * 0.4:.0f} heads = 40% of final count in single interval)")
        else:
            print(f"GRADUAL: heads increased from {first_count} to {last_count} over {total_steps} steps")
            print(f"  (~{(last_count - first_count) / max(total_steps / 500, 1):.1f} heads per 500 steps)")

        # When did heads first appear?
        first_nonzero = None
        for r in results:
            if r["delimiter_heads_raw50"] > 5:
                first_nonzero = r["step"]
                break
        if first_nonzero:
            print(f"\nFirst significant specialization (>5 heads): step {first_nonzero}")

        # Visual timeline
        print(f"\nTimeline:")
        for r in results:
            bar_len = r["delimiter_heads_raw50"] // 2
            bar = "█" * bar_len
            print(f"  Step {r['step']:>5}: {r['delimiter_heads_raw50']:>3} heads  {bar}")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "checkpoints_probed": len(results),
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
            (args.output, "logs/run-002-ablation/emergence-results.json"),
        ]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/emergence-log.txt")
            print(f"  Uploaded emergence-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
