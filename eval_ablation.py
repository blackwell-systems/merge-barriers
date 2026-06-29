#!/usr/bin/env python3
"""
Head ablation experiment: proves delimiter-specialized attention heads
are causally responsible for structured data comprehension.

Procedure:
  1. Load Model A (merge barriers) and identify delimiter-majority heads
  2. Measure baseline PPL on structured data and natural language
  3. Progressively disable delimiter heads (zero attention weights)
  4. Measure PPL after each ablation step
  5. Compare: does structured PPL degrade while NL PPL stays flat?

If ablating delimiter heads crashes structured PPL back to Model B levels
while leaving NL PPL unchanged, that proves:
  - Merge barriers cause the model to develop specialized heads
  - Those heads are specifically responsible for structured data comprehension
  - Without them, the model is no better than standard BPE

This is the causal link between tokenizer design and model architecture
that connects the structok findings to GCF's design rationale.

Usage:
  python eval_ablation.py \
    --checkpoint-a checkpoints/structok/checkpoint.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoints/standard/checkpoint.pt --tokenizer-b standard-64k.json \
    --output runs/ablation-results.json

  # Quick test (fewer ablation steps)
  python eval_ablation.py --checkpoint-a ... --tokenizer-a ... --checkpoint-b ... --tokenizer-b ... --quick
"""

import argparse
import gc
import json
import math
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
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

# =========================================================================
# Test data
# =========================================================================

STRUCTURED_TEXTS = [
    # GCF generic profile
    """## orders [10]{orderId,customer,status,total}
ORD-00001|Alice Chen|pending|29.97
ORD-00002|Bob Smith|processing|42.47
ORD-00003|Carla Rodriguez|shipped|54.97
ORD-00004|David Park|delivered|67.47
ORD-00005|Eva Johansson|cancelled|79.97
ORD-00006|Alice Chen|pending|92.47
ORD-00007|Bob Smith|processing|104.97
ORD-00008|Carla Rodriguez|shipped|117.47
ORD-00009|David Park|delivered|129.97
ORD-00010|Eva Johansson|cancelled|142.47""",
    # GCF graph profile
    """GCF profile=graph tool=context_for_task budget=5000 tokens=1200 symbols=8 edges=6
## targets
@0 fn pkg/auth.ValidateToken 0.95 lsp_resolved
@1 type pkg/server.HTTPHandler 0.88 ast_inferred
@2 fn pkg/db.Connect 0.72 structural
## related
@3 fn pkg/cache.Invalidate 0.65 lsp_resolved
@4 iface pkg/config.LoadYAML 0.50 ast_inferred
## extended
@5 fn pkg/metrics.RecordLatency 0.45 structural
@6 fn pkg/logging.Emit 0.40 lsp_resolved
@7 method pkg/server.Handler.ServeHTTP 0.35 ast_inferred
## edges [6]
@1<@0 calls
@2<@0 calls
@3<@1 references
@4<@1 implements
@5<@1 calls
@6<@2 calls""",
    # JSON (for comparison)
    """{"orders": [
  {"orderId": "ORD-001", "customer": "Alice", "status": "pending", "total": 29.97},
  {"orderId": "ORD-002", "customer": "Bob", "status": "shipped", "total": 42.47},
  {"orderId": "ORD-003", "customer": "Carla", "status": "delivered", "total": 54.97},
  {"orderId": "ORD-004", "customer": "David", "status": "cancelled", "total": 67.47},
  {"orderId": "ORD-005", "customer": "Eva", "status": "pending", "total": 79.97}
]}""",
]

# Per-format text groups for disaggregated analysis
FORMAT_TEXTS = {
    "gcf_generic": [STRUCTURED_TEXTS[0]],
    "gcf_graph": [STRUCTURED_TEXTS[1]],
    "json": [STRUCTURED_TEXTS[2]],
}

NATURAL_LANGUAGE_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet and has been used as a typing exercise for decades. The origins of this pangram date back to the late 19th century when telegraph operators needed a way to test their equipment.",
    "Machine learning models process text by converting it into numerical representations called embeddings. These dense vectors capture semantic relationships between words, allowing the model to understand context and meaning. Transformer architectures revolutionized this field by introducing the self-attention mechanism.",
    "Network engineers monitor traffic patterns across data center fabrics to identify bottlenecks and optimize routing decisions. Modern spine-leaf architectures distribute load evenly across parallel paths, reducing the impact of individual link failures on overall throughput.",
]


# =========================================================================
# Model loading
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
    token_str = id_to_token.get(token_id, "")
    return any(c in token_str for c in BARRIER_CHARS)


# =========================================================================
# PPL measurement
# =========================================================================

def compute_ppl(model, tok, texts, device):
    """Compute perplexity across a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        ids = tok.encode(text).ids
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
        total_loss += loss.item()
        total_tokens += len(ids) - 1

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# =========================================================================
# Head identification
# =========================================================================

def identify_delimiter_heads(model, tok, device, threshold=0.5):
    """Find all attention heads where >threshold of attention goes to delimiters.

    Averages across all structured texts to avoid single-sample bias.
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_scores = {}  # (layer, head) -> list of scores

    for text in STRUCTURED_TEXTS:
        ids = tok.encode(text).ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        attentions = outputs.attentions
        for layer_idx, attn in enumerate(attentions):
            for head_idx in range(n_heads):
                attn_weights = attn[0, head_idx].float().cpu()
                seq_len = attn_weights.shape[0]

                delim_attn = sum(
                    attn_weights[:, d].mean().item()
                    for d in delim_positions
                )
                total_attn = sum(
                    attn_weights[:, p].mean().item()
                    for p in range(seq_len)
                )
                score = delim_attn / max(total_attn, 1e-10)

                key = (layer_idx, head_idx)
                if key not in head_scores:
                    head_scores[key] = []
                head_scores[key].append(score)

    # Average across all texts
    heads = []
    for (layer_idx, head_idx), scores in head_scores.items():
        avg_score = sum(scores) / len(scores)
        if avg_score > threshold:
            heads.append((layer_idx, head_idx, avg_score))

    heads.sort(key=lambda x: x[2], reverse=True)
    return heads


# =========================================================================
# Head ablation (the experiment)
# =========================================================================

def ablate_heads(model, heads_to_ablate):
    """Zero out the output projection weights for specific attention heads.

    This is a permanent modification to the model weights (on the copy).
    Each ablated head produces zero-valued output, effectively disabling it.
    """
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    for layer_idx, head_idx in heads_to_ablate:
        # Zero out the dense (output projection) weights for this head
        dense = model.gpt_neox.layers[layer_idx].attention.dense
        start = head_idx * head_dim
        end = start + head_dim
        dense.weight.data[:, start:end] = 0.0


def run_ablation_experiment(model_a, tok_a, model_b, tok_b, device, quick=False):
    """Progressive ablation of delimiter heads in Model A."""

    print("\n" + "=" * 80)
    print("PHASE 1: Identify delimiter-specialized heads")
    print("=" * 80)

    delimiter_heads = identify_delimiter_heads(model_a, tok_a, device)
    n_total_heads = model_a.config.num_hidden_layers * model_a.config.num_attention_heads
    print(f"\nDelimiter-majority heads (>50%): {len(delimiter_heads)} / {n_total_heads}")
    print(f"Top 10:")
    for layer, head, score in delimiter_heads[:10]:
        print(f"  Layer {layer:>2}, Head {head:>2}: {score:.1%}")

    print("\n" + "=" * 80)
    print("PHASE 2: Baseline PPL (no ablation)")
    print("=" * 80)

    baseline_struct = compute_ppl(model_a, tok_a, STRUCTURED_TEXTS, device)
    baseline_nl = compute_ppl(model_a, tok_a, NATURAL_LANGUAGE_TEXTS, device)
    modelb_struct = compute_ppl(model_b, tok_b, STRUCTURED_TEXTS, device)
    modelb_nl = compute_ppl(model_b, tok_b, NATURAL_LANGUAGE_TEXTS, device)

    # Per-format baselines
    baseline_by_format = {}
    modelb_by_format = {}
    for fmt, texts in FORMAT_TEXTS.items():
        baseline_by_format[fmt] = compute_ppl(model_a, tok_a, texts, device)
        modelb_by_format[fmt] = compute_ppl(model_b, tok_b, texts, device)

    print(f"\nModel A (merge barriers):")
    print(f"  Combined structured PPL: {baseline_struct:.2f}")
    for fmt, ppl in baseline_by_format.items():
        print(f"  {fmt}: {ppl:.2f}")
    print(f"  Natural lang PPL: {baseline_nl:.2f}")
    print(f"\nModel B (standard BPE):")
    print(f"  Combined structured PPL: {modelb_struct:.2f}")
    for fmt, ppl in modelb_by_format.items():
        print(f"  {fmt}: {ppl:.2f}")
    print(f"  Natural lang PPL: {modelb_nl:.2f}")

    print("\n" + "=" * 80)
    print("PHASE 3: Progressive ablation")
    print("=" * 80)

    # Ablation steps: remove N delimiter heads at a time
    n_delim = len(delimiter_heads)
    if quick:
        steps = [0, n_delim // 4, n_delim // 2, 3 * n_delim // 4, n_delim]
    else:
        steps = list(range(0, n_delim + 1, max(1, n_delim // 10)))
        if steps[-1] != n_delim:
            steps.append(n_delim)

    results = []

    print(f"\nAblating {n_delim} delimiter heads in {len(steps)} steps")
    print(f"\n{'Heads':>6} {'GCF Gen':>10} {'GCF Graph':>10} {'JSON':>10} {'NL':>10} {'GCFg Δ':>8} {'JSON Δ':>8} {'NL Δ':>7}")
    print("-" * 76)

    for n_ablate in steps:
        # Deep copy the model for each step
        model_copy = copy.deepcopy(model_a)
        model_copy.to(device)

        if n_ablate > 0:
            heads_to_remove = [(l, h) for l, h, _ in delimiter_heads[:n_ablate]]
            ablate_heads(model_copy, heads_to_remove)

        # Per-format PPL
        fmt_ppl = {}
        for fmt, texts in FORMAT_TEXTS.items():
            fmt_ppl[fmt] = compute_ppl(model_copy, tok_a, texts, device)

        nl_ppl = compute_ppl(model_copy, tok_a, NATURAL_LANGUAGE_TEXTS, device)

        gcfg_delta = ((fmt_ppl["gcf_generic"] - baseline_by_format["gcf_generic"]) / baseline_by_format["gcf_generic"]) * 100
        json_delta = ((fmt_ppl["json"] - baseline_by_format["json"]) / baseline_by_format["json"]) * 100
        nl_delta = ((nl_ppl - baseline_nl) / baseline_nl) * 100

        print(f"{n_ablate:>6} {fmt_ppl['gcf_generic']:>10.1f} {fmt_ppl['gcf_graph']:>10.1f} {fmt_ppl['json']:>10.1f} {nl_ppl:>10.1f} {gcfg_delta:>+7.1f}% {json_delta:>+7.1f}% {nl_delta:>+6.1f}%")

        results.append({
            "heads_ablated": n_ablate,
            "gcf_generic_ppl": round(fmt_ppl["gcf_generic"], 4),
            "gcf_graph_ppl": round(fmt_ppl["gcf_graph"], 4),
            "json_ppl": round(fmt_ppl["json"], 4),
            "natural_language_ppl": round(nl_ppl, 4),
            "gcf_generic_delta_pct": round(gcfg_delta, 2),
            "json_delta_pct": round(json_delta, 2),
            "nl_delta_pct": round(nl_delta, 2),
        })

        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("PHASE 4: Control ablation (random non-delimiter heads)")
    print("=" * 80)

    # Identify non-delimiter heads
    delim_set = {(l, h) for l, h, _ in delimiter_heads}
    n_layers = model_a.config.num_hidden_layers
    n_heads_per_layer = model_a.config.num_attention_heads
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)]
    non_delim_heads = [(l, h) for l, h in all_heads if (l, h) not in delim_set]

    # Ablate same number of random non-delimiter heads
    import random
    random.seed(42)
    control_steps = [0, n_delim // 2, n_delim] if quick else [0, n_delim // 4, n_delim // 2, 3 * n_delim // 4, min(n_delim, len(non_delim_heads))]
    control_results = []

    print(f"\nAblating up to {min(n_delim, len(non_delim_heads))} random NON-delimiter heads as control")
    print(f"\n{'Heads':>6} {'GCF Gen':>10} {'GCF Graph':>10} {'JSON':>10} {'NL':>10} {'GCFg Δ':>8} {'JSON Δ':>8} {'NL Δ':>7}")
    print("-" * 76)

    random.shuffle(non_delim_heads)

    for n_ablate in control_steps:
        model_copy = copy.deepcopy(model_a)
        model_copy.to(device)

        if n_ablate > 0:
            heads_to_remove = non_delim_heads[:n_ablate]
            ablate_heads(model_copy, heads_to_remove)

        fmt_ppl = {}
        for fmt, texts in FORMAT_TEXTS.items():
            fmt_ppl[fmt] = compute_ppl(model_copy, tok_a, texts, device)

        nl_ppl = compute_ppl(model_copy, tok_a, NATURAL_LANGUAGE_TEXTS, device)

        gcfg_delta = ((fmt_ppl["gcf_generic"] - baseline_by_format["gcf_generic"]) / baseline_by_format["gcf_generic"]) * 100
        json_delta = ((fmt_ppl["json"] - baseline_by_format["json"]) / baseline_by_format["json"]) * 100
        nl_delta = ((nl_ppl - baseline_nl) / baseline_nl) * 100

        print(f"{n_ablate:>6} {fmt_ppl['gcf_generic']:>10.1f} {fmt_ppl['gcf_graph']:>10.1f} {fmt_ppl['json']:>10.1f} {nl_ppl:>10.1f} {gcfg_delta:>+7.1f}% {json_delta:>+7.1f}% {nl_delta:>+6.1f}%")

        control_results.append({
            "heads_ablated": n_ablate,
            "gcf_generic_ppl": round(fmt_ppl["gcf_generic"], 4),
            "gcf_graph_ppl": round(fmt_ppl["gcf_graph"], 4),
            "json_ppl": round(fmt_ppl["json"], 4),
            "natural_language_ppl": round(nl_ppl, 4),
            "gcf_generic_delta_pct": round(gcfg_delta, 2),
            "json_delta_pct": round(json_delta, 2),
            "nl_delta_pct": round(nl_delta, 2),
        })

        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    final_ablation = results[-1]
    final_control = control_results[-1]

    print(f"\nBaseline (Model A, no ablation):")
    print(f"  Structured PPL: {baseline_struct:.2f}")
    print(f"  NL PPL: {baseline_nl:.2f}")
    print(f"\nModel B (standard BPE, reference):")
    print(f"  Structured PPL: {modelb_struct:.2f}")
    print(f"  NL PPL: {modelb_nl:.2f}")
    print(f"\nAll delimiter heads ablated ({n_delim} heads):")
    print(f"  GCF generic PPL: {final_ablation['gcf_generic_ppl']:.2f} ({final_ablation['gcf_generic_delta_pct']:+.1f}%)")
    print(f"  GCF graph PPL:   {final_ablation['gcf_graph_ppl']:.2f}")
    print(f"  JSON PPL:        {final_ablation['json_ppl']:.2f} ({final_ablation['json_delta_pct']:+.1f}%)")
    print(f"  NL PPL:          {final_ablation['natural_language_ppl']:.2f} ({final_ablation['nl_delta_pct']:+.1f}%)")
    print(f"\nSame count random non-delimiter heads ablated (control):")
    print(f"  GCF generic PPL: {final_control['gcf_generic_ppl']:.2f} ({final_control['gcf_generic_delta_pct']:+.1f}%)")
    print(f"  GCF graph PPL:   {final_control['gcf_graph_ppl']:.2f}")
    print(f"  JSON PPL:        {final_control['json_ppl']:.2f} ({final_control['json_delta_pct']:+.1f}%)")
    print(f"  NL PPL:          {final_control['natural_language_ppl']:.2f} ({final_control['nl_delta_pct']:+.1f}%)")

    return {
        "delimiter_heads_found": n_delim,
        "total_heads": n_total_heads,
        "baseline": {
            "model_a_structured_ppl": round(baseline_struct, 4),
            "model_a_nl_ppl": round(baseline_nl, 4),
            "model_b_structured_ppl": round(modelb_struct, 4),
            "model_b_nl_ppl": round(modelb_nl, 4),
        },
        "ablation_curve": results,
        "control_curve": control_results,
        "top_delimiter_heads": [
            {"layer": l, "head": h, "score": round(s, 4)}
            for l, h, s in delimiter_heads[:20]
        ],
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Head ablation experiment")
    parser.add_argument("--checkpoint-a", required=True, help="Model A (merge barriers) checkpoint")
    parser.add_argument("--tokenizer-a", required=True, help="Model A tokenizer")
    parser.add_argument("--checkpoint-b", required=True, help="Model B (standard BPE) checkpoint")
    parser.add_argument("--tokenizer-b", required=True, help="Model B tokenizer")
    parser.add_argument("--size", default="410m", help="Model size config")
    parser.add_argument("--device", default=None, help="Device (auto-detected if not set)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--quick", action="store_true", help="Fewer ablation steps (faster)")

    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print("\nLoading Model A (merge barriers)...")
    model_a, tok_a, step_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    print("Loading Model B (standard BPE)...")
    model_b, tok_b, step_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)
    model_b.to(device)

    results = run_ablation_experiment(model_a, tok_a, model_b, tok_b, device, quick=args.quick)
    results["step_a"] = step_a
    results["step_b"] = step_b
    results["device"] = device

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
