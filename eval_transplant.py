#!/usr/bin/env python3
"""
Head transplant experiment: graft Model A's delimiter heads into Model B.

If Model B's structured data PPL improves after receiving Model A's delimiter
head weights (without any retraining), the heads carry portable structural
knowledge that transfers across models.

Usage:
  python eval_transplant.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoint-b.pt --tokenizer-b standard-64k.json \
    --output transplant-results.json
"""

import argparse
import copy
import datetime
import gc
import json
import math
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
# Test data
# =========================================================================

def gen_gcf_generic(n=50):
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i%5]}|{statuses[i%5]}|{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_gcf_graph(n=20):
    pkgs = ["pkg/auth","pkg/server","pkg/db","pkg/cache","pkg/config"]
    names = ["Validate","Process","Handle","Create","Update","Delete","Get","Set","Check","Build"]
    kinds = ["fn","type","method","iface"]
    lines = [f"GCF profile=graph tool=context symbols={n} edges={n-5}"]
    for i in range(n):
        g = "targets" if i < n//3 else "related" if i < 2*n//3 else "extended"
        if i == 0 or (i == n//3) or (i == 2*n//3):
            lines.append(f"## {g}")
        lines.append(f"@{i} {kinds[i%4]} {pkgs[i%5]}.{names[i%10]}{i} {round(0.95-i*0.03,2)} lsp_resolved")
    lines.append(f"## edges [{n-5}]")
    et = ["calls","imports","implements","references"]
    for i in range(n-5):
        lines.append(f"@{(i*3)%n}<@{(i*3+1)%n} {et[i%4]}")
    return "\n".join(lines)

def gen_json(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    records = [{"orderId":f"ORD-{i+1:05d}","customer":names[i%5],"status":statuses[i%5],"total":round(29.97+i*12.50,2)} for i in range(n)]
    return json.dumps({"orders":records}, indent=2)

NL_TEXT = ("The architecture of modern distributed systems has evolved significantly. "
    "Microservices replaced monolithic applications, bringing independent deployment "
    "and technology diversity, but also complexity in service discovery and tracing.")

FORMAT_TEXTS = {
    "gcf_generic": [gen_gcf_generic(50), gen_gcf_generic(30)],
    "gcf_graph": [gen_gcf_graph(20)],
    "json": [gen_json(50), gen_json(30)],
    "nl": [NL_TEXT],
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
    test_texts = [FORMAT_TEXTS["gcf_generic"][0], FORMAT_TEXTS["gcf_graph"][0],
                  FORMAT_TEXTS["json"][0]]

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


def measure_all(model, tok, device):
    return {fmt: compute_ppl(model, tok, texts, device) for fmt, texts in FORMAT_TEXTS.items()}


def transplant_heads(donor, recipient, heads):
    """Copy attention head weights from donor to recipient.

    Copies Q, K, V projection weights and output projection weights
    for each specified head.
    """
    n_heads = donor.config.num_attention_heads
    head_dim = donor.config.hidden_size // n_heads

    for layer_idx, head_idx in heads:
        start = head_idx * head_dim
        end = start + head_dim

        donor_layer = donor.gpt_neox.layers[layer_idx].attention
        recip_layer = recipient.gpt_neox.layers[layer_idx].attention

        # Copy Q, K, V projections for this head
        # GPTNeoX uses fused QKV: query_key_value.weight is [3*hidden, hidden]
        # Each head's Q is at [head*head_dim : (head+1)*head_dim]
        # K is at [hidden + head*head_dim : hidden + (head+1)*head_dim]
        # V is at [2*hidden + head*head_dim : 2*hidden + (head+1)*head_dim]
        hidden = donor.config.hidden_size

        # QKV weights
        with torch.no_grad():
            # Q
            recip_layer.query_key_value.weight.data[start:end, :] = \
                donor_layer.query_key_value.weight.data[start:end, :]
            # K
            recip_layer.query_key_value.weight.data[hidden+start:hidden+end, :] = \
                donor_layer.query_key_value.weight.data[hidden+start:hidden+end, :]
            # V
            recip_layer.query_key_value.weight.data[2*hidden+start:2*hidden+end, :] = \
                donor_layer.query_key_value.weight.data[2*hidden+start:2*hidden+end, :]

            # QKV biases
            if donor_layer.query_key_value.bias is not None:
                recip_layer.query_key_value.bias.data[start:end] = \
                    donor_layer.query_key_value.bias.data[start:end]
                recip_layer.query_key_value.bias.data[hidden+start:hidden+end] = \
                    donor_layer.query_key_value.bias.data[hidden+start:hidden+end]
                recip_layer.query_key_value.bias.data[2*hidden+start:2*hidden+end] = \
                    donor_layer.query_key_value.bias.data[2*hidden+start:2*hidden+end]

            # Output projection
            recip_layer.dense.weight.data[:, start:end] = \
                donor_layer.dense.weight.data[:, start:end]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Head transplant experiment")
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
        "experiment": "head_transplant",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("HEAD TRANSPLANT EXPERIMENT")
    print("=" * 90)
    print(f"\nQuestion: do delimiter heads carry portable structural knowledge?")
    print(f"Method: copy Model A's top delimiter head weights into Model B,")
    print(f"measure if Model B's structured data PPL improves without retraining.")
    print(f"\nDevice: {device}")

    print("\nLoading models...")
    model_a, tok_a, _ = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)
    model_b, tok_b, _ = load_model(args.checkpoint_b, args.size, args.tokenizer_b)
    model_b.to(device)

    # Identify delimiter heads in Model A
    print("\nIdentifying delimiter heads in Model A...")
    delimiter_heads_a = identify_delimiter_heads(model_a, tok_a, device)
    print(f"Found {len(delimiter_heads_a)} delimiter heads")

    # Baselines (using each model's own tokenizer)
    print("\n" + "=" * 90)
    print("BASELINES (each model with its own tokenizer)")
    print("=" * 90)

    baseline_a = measure_all(model_a, tok_a, device)
    baseline_b = measure_all(model_b, tok_b, device)

    print(f"\n{'Format':<14} {'Model A':>10} {'Model B':>10} {'A/B ratio':>10}")
    print("-" * 48)
    for fmt in FORMAT_TEXTS:
        ratio = baseline_b[fmt] / max(baseline_a[fmt], 1)
        print(f"{fmt:<14} {baseline_a[fmt]:>10.1f} {baseline_b[fmt]:>10.1f} {ratio:>9.1f}x")

    # Move model A off GPU for the transplant experiments
    model_a_cpu = model_a.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    # Transplant experiments at different head counts
    print("\n" + "=" * 90)
    print("TRANSPLANT: Model A heads -> Model B")
    print("=" * 90)

    transplant_counts = [5, 10, 20, 40, len(delimiter_heads_a)]
    transplant_results = []

    print(f"\n{'Heads':>6} ", end="")
    for fmt in FORMAT_TEXTS:
        print(f" {fmt:>12}", end="")
    print()
    print("-" * (6 + 13 * len(FORMAT_TEXTS)))

    for n_heads in transplant_counts:
        if n_heads > len(delimiter_heads_a):
            n_heads = len(delimiter_heads_a)

        # Fresh copy of Model B
        model_hybrid = copy.deepcopy(model_b)
        model_hybrid.to(device)

        # Bring model A back to device for weight copying
        model_a_cpu.to(device)

        # Transplant top N delimiter heads from A to B
        heads_to_transplant = [(l, h) for l, h, _ in delimiter_heads_a[:n_heads]]
        transplant_heads(model_a_cpu, model_hybrid, heads_to_transplant)

        # Move A back to CPU
        model_a_cpu.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # Measure hybrid model with Model B's tokenizer
        ppls = measure_all(model_hybrid, tok_b, device)

        result = {"n_heads": n_heads}
        line = f"{n_heads:>6} "
        for fmt in FORMAT_TEXTS:
            ppl = ppls[fmt]
            delta = ((ppl - baseline_b[fmt]) / baseline_b[fmt]) * 100
            result[f"{fmt}_ppl"] = round(ppl, 2)
            result[f"{fmt}_delta"] = round(delta, 1)
            line += f" {ppl:>10.1f}  "
        print(line)
        transplant_results.append(result)

        del model_hybrid
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY: PPL change from transplant (vs Model B baseline)")
    print("=" * 90)

    print(f"\n{'Heads':>6} ", end="")
    for fmt in FORMAT_TEXTS:
        print(f" {fmt:>12}", end="")
    print()
    print("-" * (6 + 13 * len(FORMAT_TEXTS)))

    for r in transplant_results:
        line = f"{r['n_heads']:>6} "
        for fmt in FORMAT_TEXTS:
            delta = r[f"{fmt}_delta"]
            line += f" {delta:>+10.1f}% "
        print(line)

    # Check if any improvement
    final = transplant_results[-1]
    gcf_improved = final.get("gcf_generic_delta", 0) < -5
    nl_stable = abs(final.get("nl_delta", 0)) < 20

    print()
    if gcf_improved:
        print("FINDING: Transplant IMPROVED Model B's structured data PPL.")
        print("Delimiter heads carry portable structural knowledge.")
    else:
        print("FINDING: Transplant did NOT improve Model B's structured data PPL.")
        print("The heads are context-dependent; they rely on clean token representations")
        print("that only exist in merge-barrier models.")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "delimiter_heads_a": len(delimiter_heads_a),
            "baseline_a": {k: round(v, 4) for k, v in baseline_a.items()},
            "baseline_b": {k: round(v, 4) for k, v in baseline_b.items()},
            "transplant_results": transplant_results,
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
        for local, key in [(args.output, "logs/run-002-ablation/transplant-results.json")]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/transplant-log.txt")
            print(f"  Uploaded logs/run-002-ablation/transplant-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
