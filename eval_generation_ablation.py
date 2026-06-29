#!/usr/bin/env python3
"""
Generation quality under ablation: do delimiter heads matter for output?

Gives Model A a GCF prompt (header + first 2 rows) and lets it generate
the next rows. Compares generation quality before and after ablating
delimiter heads. Measures:
- Delimiter accuracy (are pipes in the right places?)
- Field count consistency (same number of fields per row?)
- Structural validity (does output parse as valid GCF rows?)

Usage:
  python eval_generation_ablation.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --output generation-ablation-results.json
"""

import argparse
import copy
import datetime
import gc
import json
import os
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

# Prompt: header + 2 rows. Model should continue with more rows.
PROMPTS = [
    {
        "name": "generic_orders",
        "prompt": """## orders [10]{orderId,customer,status,total}
ORD-00001|Alice Chen|pending|29.97
ORD-00002|Bob Smith|processing|42.47
""",
        "expected_fields": 4,
        "delimiter": "|",
    },
    {
        "name": "graph_symbols",
        "prompt": """GCF profile=graph tool=context symbols=10 edges=5
## targets
@0 fn pkg/auth.Validate 0.95 lsp_resolved
@1 type pkg/server.Process 0.88 ast_inferred
""",
        "expected_fields": None,  # variable (symbol lines have 5 space-separated fields)
        "delimiter": " ",
    },
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
    model.eval()
    return model, tok


def is_delimiter_token(tok, token_id):
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    return any(c in id_to_token.get(token_id, "") for c in BARRIER_CHARS)


def identify_delimiter_heads(model, tok, device, threshold=0.5):
    text = PROMPTS[0]["prompt"]
    ids = tok.encode(text).ids[:256]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    heads = []
    for layer_idx, attn in enumerate(outputs.attentions):
        for head_idx in range(attn.shape[1]):
            w = attn[0, head_idx].float().cpu()
            seq_len = w.shape[0]
            delim_attn = sum(w[:, d].mean().item() for d in delim_positions)
            total_attn = sum(w[:, p].mean().item() for p in range(seq_len))
            raw = delim_attn / max(total_attn, 1e-10)
            if raw > threshold:
                heads.append((layer_idx, head_idx, raw))

    del outputs
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

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


def generate(model, tok, prompt, max_tokens=200, temperature=0.5, device="cuda"):
    """Generate text continuation."""
    ids = tok.encode(prompt).ids
    generated = list(ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            if len(generated) >= 2048:
                break
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(input_ids=input_ids).logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

            # Stop at double newline (end of section)
            decoded_last = tok.decode([next_token])
            if len(generated) > len(ids) + 10 and decoded_last == "\n":
                # Check if previous token was also newline
                prev_decoded = tok.decode([generated[-2]])
                if prev_decoded == "\n":
                    break

    # Return only the generated part
    full_text = tok.decode(generated)
    prompt_text = tok.decode(ids)
    continuation = full_text[len(prompt_text):]
    return continuation


def score_generation(text, expected_fields, delimiter):
    """Score generated text for structural quality."""
    lines = text.strip().split("\n")
    lines = [l for l in lines if l.strip()]  # remove empty

    if not lines:
        return {"n_lines": 0, "valid_lines": 0, "validity_rate": 0, "avg_fields": 0,
                "field_consistency": 0, "delimiter_count": 0, "sample": ""}

    valid = 0
    field_counts = []
    total_delimiters = 0

    for line in lines:
        # Skip section headers
        if line.startswith("##") or line.startswith("GCF "):
            continue

        delim_count = line.count(delimiter) if delimiter == "|" else line.count(delimiter)
        total_delimiters += delim_count

        if delimiter == "|":
            fields = len(line.split("|"))
        else:
            fields = len(line.split())

        field_counts.append(fields)

        if expected_fields is None or fields == expected_fields:
            valid += 1

    n_data_lines = len(field_counts)
    avg_fields = sum(field_counts) / max(len(field_counts), 1)

    # Field consistency: are all rows the same width?
    if field_counts:
        mode_fields = max(set(field_counts), key=field_counts.count)
        consistent = sum(1 for f in field_counts if f == mode_fields)
        field_consistency = consistent / len(field_counts)
    else:
        field_consistency = 0

    return {
        "n_lines": len(lines),
        "n_data_lines": n_data_lines,
        "valid_lines": valid,
        "validity_rate": round(valid / max(n_data_lines, 1), 4),
        "avg_fields": round(avg_fields, 1),
        "field_consistency": round(field_consistency, 4),
        "delimiter_count": total_delimiters,
        "sample": text[:300],
    }


def main():
    parser = argparse.ArgumentParser(description="Generation quality under ablation")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--size", default="410m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    metadata = {
        "experiment": "generation_quality_under_ablation",
        "description": "Do delimiter heads matter for GCF generation, not just comprehension?",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "n_samples": args.n_samples,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("GENERATION QUALITY UNDER ABLATION")
    print("=" * 90)
    print(f"\nDevice: {device}")
    print(f"Samples per condition: {args.n_samples}")

    print("\nLoading model...", flush=True)
    model, tok = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model.to(device)

    print("Identifying delimiter heads...", flush=True)
    delimiter_heads = identify_delimiter_heads(model, tok, device)
    print(f"Found {len(delimiter_heads)} delimiter heads")

    # Create ablated copy
    model_ablated = copy.deepcopy(model)
    model_ablated.to(device)
    ablate_heads(model_ablated, [(l, h) for l, h, _ in delimiter_heads])

    results = {}

    for prompt_info in PROMPTS:
        name = prompt_info["name"]
        prompt = prompt_info["prompt"]
        expected_fields = prompt_info["expected_fields"]
        delimiter = prompt_info["delimiter"]

        print(f"\n{'=' * 70}")
        print(f"Prompt: {name}")
        print(f"{'=' * 70}")

        for condition, mdl in [("baseline", model), ("ablated", model_ablated)]:
            print(f"\n  {condition.upper()}:")
            scores = []

            for i in range(args.n_samples):
                text = generate(mdl, tok, prompt, max_tokens=200, temperature=0.5, device=device)
                score = score_generation(text, expected_fields, delimiter)
                scores.append(score)
                print(f"    Sample {i+1}: {score['n_data_lines']} lines, "
                      f"{score['validity_rate']:.0%} valid, "
                      f"consistency={score['field_consistency']:.0%}, "
                      f"delimiters={score['delimiter_count']}")
                if i == 0:
                    # Show first sample
                    for line in score["sample"].split("\n")[:5]:
                        print(f"      > {line}")

            # Averages
            avg_validity = sum(s["validity_rate"] for s in scores) / len(scores)
            avg_consistency = sum(s["field_consistency"] for s in scores) / len(scores)
            avg_delimiters = sum(s["delimiter_count"] for s in scores) / len(scores)
            avg_lines = sum(s["n_data_lines"] for s in scores) / len(scores)

            print(f"\n    Avg: {avg_lines:.1f} lines, {avg_validity:.0%} valid, "
                  f"consistency={avg_consistency:.0%}, delimiters={avg_delimiters:.0f}")

            results[f"{name}_{condition}"] = {
                "condition": condition,
                "prompt": name,
                "avg_validity_rate": round(avg_validity, 4),
                "avg_field_consistency": round(avg_consistency, 4),
                "avg_delimiter_count": round(avg_delimiters, 1),
                "avg_data_lines": round(avg_lines, 1),
                "samples": scores,
            }

    del model_ablated
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"\n{'Prompt':<20} {'Condition':<12} {'Validity':>10} {'Consistency':>13} {'Delimiters':>12}")
    print("-" * 70)

    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['prompt']:<20} {r['condition']:<12} {r['avg_validity_rate']:>9.0%} {r['avg_field_consistency']:>12.0%} {r['avg_delimiter_count']:>12.0f}")

    # Check if ablation degrades generation
    for prompt_info in PROMPTS:
        name = prompt_info["name"]
        base = results[f"{name}_baseline"]
        abl = results[f"{name}_ablated"]

        validity_drop = base["avg_validity_rate"] - abl["avg_validity_rate"]
        consistency_drop = base["avg_field_consistency"] - abl["avg_field_consistency"]

        if validity_drop > 0.1 or consistency_drop > 0.1:
            print(f"\n  {name}: ABLATION DEGRADES GENERATION")
            print(f"    Validity:    {base['avg_validity_rate']:.0%} -> {abl['avg_validity_rate']:.0%} ({validity_drop:+.0%})")
            print(f"    Consistency: {base['avg_field_consistency']:.0%} -> {abl['avg_field_consistency']:.0%} ({consistency_drop:+.0%})")
        else:
            print(f"\n  {name}: ablation does not significantly degrade generation")

    # Save
    if args.output:
        out = {"metadata": metadata, "delimiter_heads": len(delimiter_heads), "results": results}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, default=str)
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
        for local, key in [(args.output, "logs/run-002-ablation/generation-ablation-results.json")]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/generation-ablation-log.txt")
            print(f"  Uploaded generation-ablation-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
