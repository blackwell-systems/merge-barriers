#!/usr/bin/env python3
"""
Attention decay analysis: structok vs standard BPE.

Reproduces the attention entropy crossover and grammar attention collapse
analysis from the GCF tokenizer study, but on our controlled models instead
of Pythia/Gemma. Same data, same architecture, different tokenizer.

Measures at each scale (5, 10, 20, 50, 100 orders):
1. Attention entropy (diffuse = noise, focused = signal)
2. Grammar token attention share (does it collapse at scale?)
3. Token repetition ratio (identical token IDs competing for attention)
4. Per-layer entropy profile (where does comprehension break down?)

Usage:
  python eval_attention.py \
    --checkpoint-a checkpoints/structok/checkpoint.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoints/standard/checkpoint.pt --tokenizer-b standard-64k.json \
    --output attention-analysis-results.json
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import numpy as np


MODEL_CONFIGS = {
    "410m": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
    },
}

BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')


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


def entropy(probs):
    """Shannon entropy of a probability distribution (bits)."""
    return -sum(float(p) * math.log2(float(p)) for p in probs if p > 1e-10)


def build_orders_json(n):
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    orders = []
    for i in range(n):
        orders.append({
            "orderId": f"ORD-{i+1:05d}",
            "customer": names[i % 5],
            "status": statuses[i % 5],
            "total": round(29.97 + i * 12.5, 2),
        })
    return json.dumps(orders, indent=2)


def build_orders_gcf(n):
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i % 5]}|{statuses[i % 5]}|{round(29.97 + i * 12.5, 2)}")
    return "\n".join(lines)


def classify_tokens(tok, ids, format_name):
    """Classify each token as grammar, field_name, or payload."""
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    labels = []
    for tid in ids:
        token_str = id_to_token.get(tid, "")

        if format_name == "GCF":
            # GCF grammar: delimiters, section markers
            if any(c in token_str for c in '|@<>{}[]\n'):
                labels.append("grammar")
            elif token_str.strip() in ("##", ","):
                labels.append("grammar")
            elif token_str.strip() == "":
                labels.append("whitespace")
            else:
                labels.append("payload")
        else:
            # JSON grammar: quotes, colons, commas, braces, brackets
            stripped = token_str.strip()
            if stripped in ('"', ':', ',', '{', '}', '[', ']'):
                labels.append("grammar")
            elif any(c in token_str for c in '"{}[]:,') and len(stripped) <= 3:
                labels.append("grammar")
            elif stripped in ("orderId", "customer", "status", "total"):
                labels.append("field_name")
            elif stripped.startswith('"') and any(c.isalpha() for c in stripped[1:]):
                # Merged quote+content (like "name)
                labels.append("field_name")
            elif stripped == "":
                labels.append("whitespace")
            else:
                labels.append("payload")

    return labels


def analyze_attention(model, tok, text, format_name, device):
    """Extract attention weights and compute entropy, grammar share, repetition."""
    ids = tok.encode(text).ids
    if len(ids) > 2048:
        ids = ids[:2048]

    seq_len = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    attentions = outputs.attentions  # tuple of (1, num_heads, seq_len, seq_len)
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    labels = classify_tokens(tok, ids, format_name)

    # Per-layer attention entropy (averaged across heads, measured from last token)
    layer_entropies = []
    for attn in attentions:
        head_entropies = []
        for h in range(n_heads):
            attn_weights = attn[0, h, -1, :].float().cpu().numpy()
            head_entropies.append(entropy(attn_weights))
        layer_entropies.append(sum(head_entropies) / len(head_entropies))

    # Grammar vs payload attention share (last token's attention, all layers/heads)
    grammar_shares = []
    payload_shares = []
    field_name_shares = []

    for attn in attentions:
        for h in range(n_heads):
            attn_weights = attn[0, h, -1, :].float().cpu().numpy()
            g, p, f = 0.0, 0.0, 0.0
            for i, label in enumerate(labels):
                w = float(attn_weights[i])
                if label == "grammar":
                    g += w
                elif label == "field_name":
                    f += w
                elif label == "payload":
                    p += w
            grammar_shares.append(g)
            payload_shares.append(p)
            field_name_shares.append(f)

    grammar_avg = sum(grammar_shares) / len(grammar_shares)
    payload_avg = sum(payload_shares) / len(payload_shares)
    field_avg = sum(field_name_shares) / len(field_name_shares)
    total = grammar_avg + payload_avg + field_avg

    # Token repetition
    unique_ids = len(set(ids))
    repetition_ratio = 1 - (unique_ids / len(ids))

    # Label counts
    n_grammar = labels.count("grammar")
    n_payload = labels.count("payload")
    n_field = labels.count("field_name")

    return {
        "format": format_name,
        "seq_len": seq_len,
        "unique_tokens": unique_ids,
        "repetition_ratio": float(repetition_ratio),
        "n_grammar": n_grammar,
        "n_payload": n_payload,
        "n_field_name": n_field,
        "layer_entropies": [float(e) for e in layer_entropies],
        "mean_entropy": float(sum(layer_entropies) / len(layer_entropies)),
        "grammar_share": float(grammar_avg / total * 100) if total > 0 else 0,
        "payload_share": float(payload_avg / total * 100) if total > 0 else 0,
        "field_share": float(field_avg / total * 100) if total > 0 else 0,
    }


def run_attention_analysis(model, tok, device, name):
    """Run the full attention analysis at multiple scales."""
    sizes = [5, 10, 20, 50, 100]
    results = []

    for n in sizes:
        gcf_text = build_orders_gcf(n)
        json_text = build_orders_json(n)

        gcf_tokens = len(tok.encode(gcf_text).ids)
        json_tokens = len(tok.encode(json_text).ids)

        # Only analyze if fits in context
        gcf_result = analyze_attention(model, tok, gcf_text, "GCF", device) if gcf_tokens <= 2048 else None
        json_result = analyze_attention(model, tok, json_text, "JSON", device) if json_tokens <= 2048 else None

        results.append({
            "n_orders": n,
            "gcf": gcf_result,
            "json": json_result,
        })

    return results


def print_results(name, results):
    """Print formatted results for one model."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    # Entropy scaling
    print(f"\n  Entropy scaling:")
    print(f"  {'Orders':>8} {'GCF entropy':>14} {'JSON entropy':>14} {'Delta':>10}")
    print(f"  {'-'*50}")
    for r in results:
        gcf_e = f"{r['gcf']['mean_entropy']:.2f}" if r['gcf'] else "overflow"
        json_e = f"{r['json']['mean_entropy']:.2f}" if r['json'] else "overflow"
        if r['gcf'] and r['json']:
            delta = r['json']['mean_entropy'] - r['gcf']['mean_entropy']
            print(f"  {r['n_orders']:>8} {gcf_e:>14} {json_e:>14} {delta:>+10.2f}")
        else:
            print(f"  {r['n_orders']:>8} {gcf_e:>14} {json_e:>14}")

    # Grammar attention share
    print(f"\n  Grammar attention share (%):")
    print(f"  {'Orders':>8} {'GCF grammar%':>14} {'GCF payload%':>14} {'JSON grammar%':>14} {'JSON payload%':>14}")
    print(f"  {'-'*68}")
    for r in results:
        if r['gcf'] and r['json']:
            print(f"  {r['n_orders']:>8} {r['gcf']['grammar_share']:>13.1f}% {r['gcf']['payload_share']:>13.1f}% {r['json']['grammar_share']:>13.1f}% {r['json']['payload_share']:>13.1f}%")
        elif r['gcf']:
            print(f"  {r['n_orders']:>8} {r['gcf']['grammar_share']:>13.1f}% {r['gcf']['payload_share']:>13.1f}% {'overflow':>14} {'overflow':>14}")

    # Token repetition
    print(f"\n  Token repetition:")
    print(f"  {'Orders':>8} {'GCF repeat%':>14} {'JSON repeat%':>14} {'GCF tokens':>12} {'JSON tokens':>12}")
    print(f"  {'-'*64}")
    for r in results:
        gcf_rep = f"{r['gcf']['repetition_ratio']:.1%}" if r['gcf'] else "n/a"
        json_rep = f"{r['json']['repetition_ratio']:.1%}" if r['json'] else "n/a"
        gcf_tok = str(r['gcf']['seq_len']) if r['gcf'] else "overflow"
        json_tok = str(r['json']['seq_len']) if r['json'] else "overflow"
        print(f"  {r['n_orders']:>8} {gcf_rep:>14} {json_rep:>14} {gcf_tok:>12} {json_tok:>12}")

    # Per-layer entropy at largest in-context size
    largest = None
    for r in reversed(results):
        if r['gcf'] and r['json']:
            largest = r
            break

    if largest:
        print(f"\n  Per-layer entropy at {largest['n_orders']} orders:")
        print(f"  {'Layer':>8} {'GCF':>10} {'JSON':>10} {'Delta':>10}")
        print(f"  {'-'*42}")
        for i in range(len(largest['gcf']['layer_entropies'])):
            ge = largest['gcf']['layer_entropies'][i]
            je = largest['json']['layer_entropies'][i]
            print(f"  {i:>8} {ge:>10.2f} {je:>10.2f} {je-ge:>+10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Attention decay analysis")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--tokenizer-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--tokenizer-b", required=True)
    parser.add_argument("--size", default="410m")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    name_a = Path(args.tokenizer_a).stem
    name_b = Path(args.tokenizer_b).stem

    print("=" * 80)
    print("ATTENTION DECAY ANALYSIS: structok vs standard BPE")
    print(f"Device: {device}")
    print("=" * 80)

    # Load both models
    model_a, tok_a, step_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_b, tok_b, step_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)

    # Run Model A
    print(f"\nAnalyzing {name_a}...")
    model_a.to(device)
    t0 = time.time()
    results_a = run_attention_analysis(model_a, tok_a, device, name_a)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_results(name_a, results_a)
    model_a.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Run Model B
    print(f"\nAnalyzing {name_b}...")
    model_b.to(device)
    t0 = time.time()
    results_b = run_attention_analysis(model_b, tok_b, device, name_b)
    print(f"  Done in {time.time()-t0:.1f}s")
    print_results(name_b, results_b)
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Side-by-side comparison
    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")

    print(f"\n  GCF entropy (lower = more focused):")
    print(f"  {'Orders':>8} {name_a:>15} {name_b:>15} {'Winner':>15}")
    print(f"  {'-'*58}")
    for ra, rb in zip(results_a, results_b):
        if ra['gcf'] and rb['gcf']:
            winner = name_a if ra['gcf']['mean_entropy'] < rb['gcf']['mean_entropy'] else name_b
            print(f"  {ra['n_orders']:>8} {ra['gcf']['mean_entropy']:>15.2f} {rb['gcf']['mean_entropy']:>15.2f} {winner:>15}")

    print(f"\n  JSON entropy (lower = more focused):")
    print(f"  {'Orders':>8} {name_a:>15} {name_b:>15} {'Winner':>15}")
    print(f"  {'-'*58}")
    for ra, rb in zip(results_a, results_b):
        if ra['json'] and rb['json']:
            winner = name_a if ra['json']['mean_entropy'] < rb['json']['mean_entropy'] else name_b
            print(f"  {ra['n_orders']:>8} {ra['json']['mean_entropy']:>15.2f} {rb['json']['mean_entropy']:>15.2f} {winner:>15}")

    print(f"\n  Grammar attention share on GCF (higher = more structural awareness):")
    print(f"  {'Orders':>8} {name_a:>15} {name_b:>15} {'Winner':>15}")
    print(f"  {'-'*58}")
    for ra, rb in zip(results_a, results_b):
        if ra['gcf'] and rb['gcf']:
            winner = name_a if ra['gcf']['grammar_share'] > rb['gcf']['grammar_share'] else name_b
            print(f"  {ra['n_orders']:>8} {ra['gcf']['grammar_share']:>14.1f}% {rb['gcf']['grammar_share']:>14.1f}% {winner:>15}")

    # Check for grammar attention collapse
    print(f"\n  Grammar attention collapse check:")
    for label, results in [(name_a, results_a), (name_b, results_b)]:
        small = [r for r in results if r['n_orders'] <= 10 and r['json']]
        large = [r for r in results if r['n_orders'] >= 50 and r['json']]
        if small and large:
            small_grammar = sum(r['json']['grammar_share'] for r in small) / len(small)
            large_grammar = sum(r['json']['grammar_share'] for r in large) / len(large)
            change = large_grammar - small_grammar
            print(f"    {label} JSON grammar attention: {small_grammar:.1f}% (small) -> {large_grammar:.1f}% (large), change: {change:+.1f}%")

        small_gcf = [r for r in results if r['n_orders'] <= 10 and r['gcf']]
        large_gcf = [r for r in results if r['n_orders'] >= 50 and r['gcf']]
        if small_gcf and large_gcf:
            small_g = sum(r['gcf']['grammar_share'] for r in small_gcf) / len(small_gcf)
            large_g = sum(r['gcf']['grammar_share'] for r in large_gcf) / len(large_gcf)
            change = large_g - small_g
            print(f"    {label} GCF grammar attention: {small_g:.1f}% (small) -> {large_g:.1f}% (large), change: {change:+.1f}%")

    # Save results
    if args.output:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_a": {"name": name_a, "step": step_a, "results": results_a},
            "model_b": {"name": name_b, "step": step_b, "results": results_b},
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=float)
        print(f"\nResults written to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
