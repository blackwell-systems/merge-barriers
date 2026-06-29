#!/usr/bin/env python3
"""
Combined experiment: three remaining ablation studies (#19, #21, #22).

#19: Embedding space analysis under ablation
    - Extract final-layer delimiter token representations before/after ablation
    - Compute intra-class cosine similarity (delimiter cohesion)
    - If representations collapse without heads, the heads maintain embedding structure

#21: Adversarial robustness under ablation
    - Feed corrupted GCF (wrong delimiters, missing fields, swapped values) through Model A and B
    - Measure PPL on clean vs corrupted
    - Ablate delimiter heads and re-measure
    - If ablation removes error detection ability, heads do error detection not just parsing

#22: Scale the sufficiency test
    - Reverse ablation: keep ONLY 70 delimiter heads, zero all others
    - Test at 100 and 200 rows (prior test was 30-50)
    - Does sufficiency hold at scale, or do other heads become necessary?

Usage:
  python eval_remaining_ablation.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoint-b.pt --tokenizer-b standard-64k.json \
    --output remaining-ablation-results.json
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
    "410m": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 2048,
    },
    "410m-llama": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2816,
        "max_position_embeddings": 2048,
        "rope_theta": 500000.0,
    },
    "1.3b-llama": {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 5632,
        "max_position_embeddings": 4096,
        "rope_theta": 500000.0,
    },
}


# =========================================================================
# Test data generators
# =========================================================================

def gen_gcf_generic(n):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson",
             "Fiona Grant", "George Wu", "Hannah Lee", "Ivan Petrov", "Julia Santos"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def gen_gcf_graph(n_symbols=30, n_edges=20):
    packages = ["pkg/auth", "pkg/server", "pkg/db", "pkg/cache", "pkg/config",
                "pkg/handler", "pkg/model", "pkg/service", "pkg/util", "pkg/middleware"]
    names = ["Validate", "Process", "Handle", "Create", "Update", "Delete",
             "Get", "Set", "Check", "Build", "Parse", "Format", "Load", "Save", "Init"]
    kinds = ["fn", "type", "method", "iface"]
    provs = ["lsp_resolved", "ast_inferred", "structural"]

    lines = [f"GCF profile=graph tool=context_for_task budget=5000 tokens={n_symbols*35} symbols={n_symbols} edges={n_edges}"]
    groups = {"targets": [], "related": [], "extended": []}
    for i in range(n_symbols):
        pkg = packages[i % len(packages)]
        name = names[i % len(names)]
        kind = kinds[i % len(kinds)]
        prov = provs[i % len(provs)]
        score = round(max(0.10, 0.95 - i * 0.02), 2)
        if i < n_symbols // 3:
            groups["targets"].append(f"@{i} {kind} {pkg}.{name}{i} {score} {prov}")
        elif i < 2 * n_symbols // 3:
            groups["related"].append(f"@{i} {kind} {pkg}.{name}{i} {score} {prov}")
        else:
            groups["extended"].append(f"@{i} {kind} {pkg}.{name}{i} {score} {prov}")
    for group_name, syms in groups.items():
        if syms:
            lines.append(f"## {group_name}")
            lines.extend(syms)
    edge_types = ["calls", "imports", "implements", "references"]
    lines.append(f"## edges [{n_edges}]")
    for i in range(n_edges):
        src = (i * 3 + 1) % n_symbols
        tgt = (i * 3) % n_symbols
        lines.append(f"@{tgt}<@{src} {edge_types[i % len(edge_types)]}")
    return "\n".join(lines)


def gen_json(n):
    names = ["Alice", "Bob", "Carla", "David", "Eva"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    records = [{"orderId": f"ORD-{i+1:05d}", "customer": names[i % 5],
                "status": statuses[i % 5], "total": round(29.97 + i * 12.50, 2)} for i in range(n)]
    return json.dumps({"orders": records}, indent=2)


def gen_yaml(n):
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    roles = ["admin", "developer", "analyst", "manager", "intern"]
    lines = ["employees:"]
    for i in range(n):
        lines.append(f"  - name: {names[i % len(names)]}")
        lines.append(f"    id: EMP-{i+1:04d}")
        lines.append(f"    role: {roles[i % len(roles)]}")
        lines.append(f"    salary: {50000 + i * 2500}")
    return "\n".join(lines)


NL_TEXT = ("The architecture of modern distributed systems has evolved significantly. "
    "Microservices replaced monolithic applications, bringing independent deployment "
    "and technology diversity, but also complexity in service discovery and tracing. "
    "Container orchestration platforms became the standard deployment target.")


# =========================================================================
# Corrupted GCF generators for adversarial robustness (#21)
# =========================================================================

def gen_gcf_wrong_delimiters(n=50):
    """GCF with pipes replaced by commas (wrong delimiter)."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        # Use commas instead of pipes
        lines.append(f"ORD-{i+1:05d},{names[i % len(names)]},{statuses[i % len(statuses)]},{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def gen_gcf_missing_fields(n=50):
    """GCF with some rows missing fields (inconsistent column count)."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        if i % 5 == 0:
            # Missing total field
            lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}")
        elif i % 7 == 0:
            # Missing status and total
            lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}")
        else:
            lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def gen_gcf_wrong_header(n=50):
    """GCF with wrong count in header (says 100 but has n)."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    # Header claims 100 rows but only n present
    lines = [f"## orders [100]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i % len(names)]}|{statuses[i % len(statuses)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


def gen_gcf_swapped_values(n=50):
    """GCF with values in wrong columns (status and customer swapped)."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        # Swap customer and status positions
        lines.append(f"ORD-{i+1:05d}|{statuses[i % len(statuses)]}|{names[i % len(names)]}|{round(29.97 + i * 12.50, 2)}")
    return "\n".join(lines)


# =========================================================================
# Model helpers
# =========================================================================

def load_model(checkpoint_path, size, tokenizer_path):
    """Load model, auto-detecting architecture from checkpoint config or size name."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"

    # Detect architecture from size name or checkpoint
    is_llama = "llama" in size
    cp_config_path = Path(checkpoint_path).parent / "config.json" if Path(checkpoint_path).is_file() else Path(checkpoint_path) / "config.json"
    if cp_config_path.exists():
        import json as _json
        with open(cp_config_path) as f:
            cp_cfg = _json.load(f)
        if cp_cfg.get("model_type") == "llama":
            is_llama = True

    if is_llama:
        # Add Llama-specific defaults if not in MODEL_CONFIGS
        cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"] // 4)
        cfg.setdefault("rope_theta", 500000.0)
        config = LlamaConfig(**cfg)
        model = LlamaForCausalLM(config)
        arch = "llama"
    else:
        config = GPTNeoXConfig(**cfg)
        model = GPTNeoXForCausalLM(config)
        arch = "neox"

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    step = cp.get("step", 0)
    print(f"Loaded {arch} model from step {step} (tokenizer: {Path(tokenizer_path).stem})")
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


def identify_delimiter_heads(model, tok, device, excess_threshold=0.10):
    """Identify delimiter-specialized heads using excess scores.

    Raw score = fraction of attention on delimiter positions. But if 40% of
    positions ARE delimiters, a uniform-attention head scores 0.40 by chance.
    Excess score = raw_score - base_rate corrects for this.

    Uses 4 probing texts (GCF generic, GCF graph, JSON, TOON) for robustness.
    A head is delimiter-specialized if its mean excess score > excess_threshold.
    """
    test_texts = [
        gen_gcf_generic(50),
        gen_gcf_graph(30, 20),
        gen_json(50),
        gen_yaml(30),
    ]
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
        if device != "cpu":
            torch.cuda.empty_cache()

    heads = []
    for (l, h), scores in head_excess_scores.items():
        avg_excess = sum(scores) / len(scores)
        if avg_excess > excess_threshold:
            heads.append((l, h, avg_excess))
    heads.sort(key=lambda x: x[2], reverse=True)

    # Log base rates for transparency
    print(f"  Base rates per probing text:")
    for i, text in enumerate(test_texts):
        ids = tok.encode(text).ids[:1024]
        delim_pos = sum(1 for tid in ids if is_delimiter_token(tok, tid))
        br = delim_pos / len(ids)
        fmt = ["GCF generic", "GCF graph", "JSON", "YAML"][i]
        print(f"    {fmt}: {delim_pos}/{len(ids)} = {br:.1%} delimiter positions")
    print(f"  Excess threshold: {excess_threshold}")
    print(f"  Heads above threshold: {len(heads)} / {model.config.num_hidden_layers * n_heads}")

    return heads


def _get_output_proj(model, layer_idx):
    """Get the output projection layer for a given layer, architecture-aware."""
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_idx].attention.dense
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn.o_proj
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def ablate_heads(model, heads):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads:
        proj = _get_output_proj(model, layer_idx)
        start = head_idx * head_dim
        end = start + head_dim
        proj.weight.data[:, start:end] = 0.0


def reverse_ablate(model, heads_to_keep):
    """Zero ALL heads EXCEPT heads_to_keep."""
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


def extract_representations(model, tok, texts, device):
    """Extract final-layer hidden states for delimiter and non-delimiter tokens."""
    delim_reps = []
    content_reps = []

    for text in texts:
        ids = tok.encode(text).ids[:1024]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        # Final layer hidden states
        final_hidden = outputs.hidden_states[-1][0].float().cpu()  # [seq_len, hidden_size]

        for i, tid in enumerate(ids):
            rep = final_hidden[i].numpy()
            if is_delimiter_token(tok, tid):
                delim_reps.append(rep)
            else:
                content_reps.append(rep)

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    return delim_reps, content_reps


def compute_cohesion(reps):
    """Compute mean pairwise cosine similarity within a set of representations."""
    import numpy as np
    if len(reps) < 2:
        return 0.0
    # Sample if too many (for speed)
    if len(reps) > 200:
        reps = random.sample(reps, 200)
    reps = np.array(reps)
    # Normalize
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    reps_normed = reps / norms
    # Pairwise cosine similarity
    sim_matrix = reps_normed @ reps_normed.T
    n = len(reps)
    # Mean of upper triangle (excluding diagonal)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i, j]
            count += 1
    return total / max(count, 1)


# =========================================================================
# Experiment #19: Embedding space analysis
# =========================================================================

def run_embedding_analysis(model_a, tok_a, delimiter_heads_a, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT #19: EMBEDDING SPACE ANALYSIS UNDER ABLATION")
    print("=" * 90)
    print("\nQuestion: Do delimiter heads maintain the embedding structure?")
    print("Method: Extract final-layer reps, measure delimiter cohesion before/after ablation\n")

    test_texts = [gen_gcf_generic(50), gen_gcf_graph(30, 20), gen_json(50)]

    # Baseline representations
    print("Extracting baseline representations...")
    delim_reps_base, content_reps_base = extract_representations(model_a, tok_a, test_texts, device)
    cohesion_delim_base = compute_cohesion(delim_reps_base)
    cohesion_content_base = compute_cohesion(content_reps_base)
    ratio_base = cohesion_delim_base / max(cohesion_content_base, 1e-10)
    print(f"  Delimiter cohesion:  {cohesion_delim_base:.4f}")
    print(f"  Content cohesion:    {cohesion_content_base:.4f}")
    print(f"  Delimiter/content:   {ratio_base:.2f}x")

    # After ablation
    print("\nAblating delimiter heads and re-extracting...")
    model_ablated = copy.deepcopy(model_a)
    model_ablated.to(device)
    ablate_heads(model_ablated, [(l, h) for l, h, _ in delimiter_heads_a])

    delim_reps_abl, content_reps_abl = extract_representations(model_ablated, tok_a, test_texts, device)
    cohesion_delim_abl = compute_cohesion(delim_reps_abl)
    cohesion_content_abl = compute_cohesion(content_reps_abl)
    ratio_abl = cohesion_delim_abl / max(cohesion_content_abl, 1e-10)
    print(f"  Delimiter cohesion:  {cohesion_delim_abl:.4f}")
    print(f"  Content cohesion:    {cohesion_content_abl:.4f}")
    print(f"  Delimiter/content:   {ratio_abl:.2f}x")

    # After random ablation (control)
    n_layers = model_a.config.num_hidden_layers
    n_heads_per = model_a.config.num_attention_heads
    delim_set = {(l, h) for l, h, _ in delimiter_heads_a}
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per)]
    non_delim = [(l, h) for l, h in all_heads if (l, h) not in delim_set]

    print("\nRandom ablation control (3 seeds)...")
    random_ratios = []
    for seed in range(3):
        rng = random.Random(seed)
        shuffled = list(non_delim)
        rng.shuffle(shuffled)
        model_rand = copy.deepcopy(model_a)
        model_rand.to(device)
        ablate_heads(model_rand, shuffled[:len(delimiter_heads_a)])
        d_reps, c_reps = extract_representations(model_rand, tok_a, test_texts, device)
        d_coh = compute_cohesion(d_reps)
        c_coh = compute_cohesion(c_reps)
        r = d_coh / max(c_coh, 1e-10)
        random_ratios.append(r)
        print(f"  Seed {seed}: delimiter/content = {r:.2f}x")
        del model_rand
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    rand_ratio_mean = sum(random_ratios) / len(random_ratios)

    del model_ablated
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Summary
    delim_change = ((cohesion_delim_abl - cohesion_delim_base) / max(abs(cohesion_delim_base), 1e-10)) * 100
    content_change = ((cohesion_content_abl - cohesion_content_base) / max(abs(cohesion_content_base), 1e-10)) * 100
    ratio_change = ((ratio_abl - ratio_base) / max(abs(ratio_base), 1e-10)) * 100

    print(f"\n{'Metric':<30} {'Baseline':>10} {'Ablated':>10} {'Change':>10}")
    print("-" * 65)
    print(f"{'Delimiter cohesion':<30} {cohesion_delim_base:>10.4f} {cohesion_delim_abl:>10.4f} {delim_change:>+9.1f}%")
    print(f"{'Content cohesion':<30} {cohesion_content_base:>10.4f} {cohesion_content_abl:>10.4f} {content_change:>+9.1f}%")
    print(f"{'Delimiter/content ratio':<30} {ratio_base:>10.2f}x {ratio_abl:>10.2f}x {ratio_change:>+9.1f}%")
    print(f"{'Random control ratio (mean)':<30} {'':>10} {rand_ratio_mean:>10.2f}x")

    if ratio_change < -10:
        conclusion = "HEADS MAINTAIN STRUCTURE: delimiter cohesion collapses under ablation"
    elif ratio_change > 10:
        conclusion = "HEADS SUPPRESS: delimiter cohesion increases under ablation (unexpected)"
    else:
        conclusion = "NULL RESULT: embedding structure is a whole-model property, not head-controlled"

    print(f"\nConclusion: {conclusion}")

    return {
        "experiment": "embedding_space_ablation",
        "baseline": {
            "delimiter_cohesion": round(cohesion_delim_base, 4),
            "content_cohesion": round(cohesion_content_base, 4),
            "ratio": round(ratio_base, 4),
            "n_delimiter_tokens": len(delim_reps_base),
            "n_content_tokens": len(content_reps_base),
        },
        "ablated": {
            "delimiter_cohesion": round(cohesion_delim_abl, 4),
            "content_cohesion": round(cohesion_content_abl, 4),
            "ratio": round(ratio_abl, 4),
        },
        "random_control": {
            "ratios": [round(r, 4) for r in random_ratios],
            "mean_ratio": round(rand_ratio_mean, 4),
        },
        "changes": {
            "delimiter_cohesion_pct": round(delim_change, 1),
            "content_cohesion_pct": round(content_change, 1),
            "ratio_change_pct": round(ratio_change, 1),
        },
        "conclusion": conclusion,
    }


# =========================================================================
# Experiment #21: Adversarial robustness
# =========================================================================

def run_adversarial_robustness(model_a, tok_a, model_b, tok_b, delimiter_heads_a, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT #21: ADVERSARIAL ROBUSTNESS UNDER ABLATION")
    print("=" * 90)
    print("\nQuestion: Do delimiter heads detect GCF corruption?")
    print("Method: Compare PPL on clean vs corrupted GCF, both models, before/after ablation\n")

    corruptions = {
        "clean": [gen_gcf_generic(50)],
        "wrong_delimiters": [gen_gcf_wrong_delimiters(50)],
        "missing_fields": [gen_gcf_missing_fields(50)],
        "wrong_header": [gen_gcf_wrong_header(50)],
        "swapped_values": [gen_gcf_swapped_values(50)],
    }

    # --- Model A: baseline ---
    print("Model A (merge barriers): baseline PPL on each corruption type")
    a_baseline = {}
    for name, texts in corruptions.items():
        ppl = compute_ppl(model_a, tok_a, texts, device)
        a_baseline[name] = ppl
        print(f"  {name:<20} PPL = {ppl:.1f}")

    # Corruption detection: how much does PPL spike on corrupted vs clean?
    a_clean_ppl = a_baseline["clean"]
    a_detection = {name: ((ppl - a_clean_ppl) / a_clean_ppl * 100) for name, ppl in a_baseline.items()}
    print(f"\n  Detection (PPL spike vs clean):")
    for name, delta in a_detection.items():
        print(f"    {name:<20} {delta:>+7.1f}%")

    # --- Model B: baseline ---
    print(f"\nModel B (standard BPE): baseline PPL on each corruption type")
    b_baseline = {}
    for name, texts in corruptions.items():
        ppl = compute_ppl(model_b, tok_b, texts, device)
        b_baseline[name] = ppl
        print(f"  {name:<20} PPL = {ppl:.1f}")

    b_clean_ppl = b_baseline["clean"]
    b_detection = {name: ((ppl - b_clean_ppl) / b_clean_ppl * 100) for name, ppl in b_baseline.items()}
    print(f"\n  Detection (PPL spike vs clean):")
    for name, delta in b_detection.items():
        print(f"    {name:<20} {delta:>+7.1f}%")

    # --- Model A: after delimiter head ablation ---
    print(f"\nModel A after delimiter head ablation:")
    model_a_abl = copy.deepcopy(model_a)
    model_a_abl.to(device)
    ablate_heads(model_a_abl, [(l, h) for l, h, _ in delimiter_heads_a])

    a_ablated = {}
    for name, texts in corruptions.items():
        ppl = compute_ppl(model_a_abl, tok_a, texts, device)
        a_ablated[name] = ppl
        print(f"  {name:<20} PPL = {ppl:.1f}")

    a_abl_clean_ppl = a_ablated["clean"]
    a_abl_detection = {name: ((ppl - a_abl_clean_ppl) / a_abl_clean_ppl * 100) for name, ppl in a_ablated.items()}
    print(f"\n  Detection after ablation (PPL spike vs ablated clean):")
    for name, delta in a_abl_detection.items():
        print(f"    {name:<20} {delta:>+7.1f}%")

    del model_a_abl
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'Corruption':<20} {'A detect':>10} {'A abl detect':>14} {'B detect':>10} {'A lost?':>10}")
    print("-" * 70)
    for name in corruptions:
        a_det = a_detection[name]
        a_abl_det = a_abl_detection[name]
        b_det = b_detection[name]
        lost = "YES" if abs(a_abl_det) < abs(a_det) * 0.5 and name != "clean" else ("partial" if abs(a_abl_det) < abs(a_det) * 0.8 and name != "clean" else "-")
        print(f"  {name:<20} {a_det:>+9.1f}% {a_abl_det:>+13.1f}% {b_det:>+9.1f}% {lost:>10}")

    # Determine if A detects better than B
    a_avg_detect = sum(abs(v) for k, v in a_detection.items() if k != "clean") / 4
    b_avg_detect = sum(abs(v) for k, v in b_detection.items() if k != "clean") / 4
    a_abl_avg_detect = sum(abs(v) for k, v in a_abl_detection.items() if k != "clean") / 4

    if a_avg_detect > b_avg_detect * 1.2 and a_abl_avg_detect < a_avg_detect * 0.7:
        conclusion = "HEADS DO ERROR DETECTION: A detects better than B, ablation removes the ability"
    elif a_avg_detect > b_avg_detect * 1.2:
        conclusion = "A DETECTS BETTER but ablation doesn't fully remove it (distributed ability)"
    elif a_avg_detect <= b_avg_detect * 1.2:
        conclusion = "NO DETECTION ADVANTAGE: both models respond similarly to corruption"
    else:
        conclusion = "INCONCLUSIVE"

    print(f"\nA avg detection: {a_avg_detect:.1f}%  B avg detection: {b_avg_detect:.1f}%  A ablated avg: {a_abl_avg_detect:.1f}%")
    print(f"Conclusion: {conclusion}")

    return {
        "experiment": "adversarial_robustness_ablation",
        "model_a_baseline": {k: round(v, 2) for k, v in a_baseline.items()},
        "model_a_detection_pct": {k: round(v, 1) for k, v in a_detection.items()},
        "model_b_baseline": {k: round(v, 2) for k, v in b_baseline.items()},
        "model_b_detection_pct": {k: round(v, 1) for k, v in b_detection.items()},
        "model_a_ablated": {k: round(v, 2) for k, v in a_ablated.items()},
        "model_a_ablated_detection_pct": {k: round(v, 1) for k, v in a_abl_detection.items()},
        "summary": {
            "a_avg_detection": round(a_avg_detect, 1),
            "b_avg_detection": round(b_avg_detect, 1),
            "a_ablated_avg_detection": round(a_abl_avg_detect, 1),
        },
        "conclusion": conclusion,
    }


# =========================================================================
# Experiment #22: Sufficiency at scale
# =========================================================================

def run_sufficiency_scaling(model_a, tok_a, delimiter_heads_a, device):
    print("\n" + "=" * 90)
    print("EXPERIMENT #22: SUFFICIENCY TEST AT SCALE")
    print("=" * 90)
    print("\nQuestion: Do 70 delimiter heads still beat 384 at 100/200 row payloads?")
    print("Method: Reverse ablation (keep ONLY delimiter heads) at larger scales\n")

    n_layers = model_a.config.num_hidden_layers
    n_heads_per = model_a.config.num_attention_heads
    total_heads = n_layers * n_heads_per
    delim_set = {(l, h) for l, h, _ in delimiter_heads_a}
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per)]

    sizes = [30, 50, 100, 200]
    formats = ["gcf_generic", "json", "yaml", "nl"]

    print(f"Delimiter heads: {len(delimiter_heads_a)} / {total_heads}")
    print(f"Sizes: {sizes}")
    print(f"Formats: {formats}")

    # Token counts
    print(f"\nToken counts per size:")
    for n in sizes:
        gcf_toks = len(tok_a.encode(gen_gcf_generic(n)).ids)
        json_toks = len(tok_a.encode(gen_json(n)).ids)
        trunc_gcf = " (TRUNC)" if gcf_toks > 2048 else ""
        trunc_json = " (TRUNC)" if json_toks > 2048 else ""
        print(f"  {n:>4} rows: GCF={gcf_toks:>5}{trunc_gcf}  JSON={json_toks:>5}{trunc_json}")

    results = []

    print(f"\n{'Size':>5} {'Format':<12} {'All heads':>10} {'Delim only':>12} {'Delim Δ':>9} {'Random only':>12} {'Rand Δ':>9} {'Sufficient?':>12}")
    print("-" * 100)

    for n in sizes:
        test_data = {
            "gcf_generic": [gen_gcf_generic(n)],
            "json": [gen_json(n)],
            "yaml": [gen_yaml(n)],
            "nl": [NL_TEXT],
        }

        size_result = {"size": n, "formats": {}}

        for fmt in formats:
            # Baseline (all heads)
            baseline = compute_ppl(model_a, tok_a, test_data[fmt], device)

            # Delimiter-only (reverse ablation)
            model_delim_only = copy.deepcopy(model_a)
            model_delim_only.to(device)
            reverse_ablate(model_delim_only, delim_set)
            delim_only_ppl = compute_ppl(model_delim_only, tok_a, test_data[fmt], device)
            delim_delta = ((delim_only_ppl - baseline) / baseline) * 100
            del model_delim_only
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

            # Random-only control (same number of heads, 3 seeds)
            random_deltas = []
            for seed in range(3):
                rng = random.Random(seed + 100)
                shuffled = list(all_heads)
                rng.shuffle(shuffled)
                keep = set(shuffled[:len(delimiter_heads_a)])
                model_rand_only = copy.deepcopy(model_a)
                model_rand_only.to(device)
                reverse_ablate(model_rand_only, keep)
                rand_ppl = compute_ppl(model_rand_only, tok_a, test_data[fmt], device)
                rand_delta = ((rand_ppl - baseline) / baseline) * 100
                random_deltas.append(rand_delta)
                del model_rand_only
                gc.collect()
                if device != "cpu":
                    torch.cuda.empty_cache()

            rand_mean = sum(random_deltas) / len(random_deltas)
            sufficient = "YES" if delim_delta < rand_mean else "NO"

            size_result["formats"][fmt] = {
                "baseline_ppl": round(baseline, 2),
                "delim_only_ppl": round(delim_only_ppl, 2),
                "delim_only_delta_pct": round(delim_delta, 1),
                "random_only_delta_mean_pct": round(rand_mean, 1),
                "random_only_deltas": [round(d, 1) for d in random_deltas],
                "sufficient": sufficient,
            }

            print(f"{n:>5} {fmt:<12} {baseline:>10.1f} {delim_only_ppl:>12.1f} {delim_delta:>+8.1f}% {rand_mean:>+11.1f}% {rand_mean:>+8.1f}% {sufficient:>12}")

        results.append(size_result)
        print()

    # Summary
    print("SUFFICIENCY SUMMARY:")
    print(f"\n{'Size':>5}", end="")
    for fmt in formats:
        print(f" {fmt:>14}", end="")
    print()
    print("-" * (5 + 15 * len(formats)))

    for r in results:
        line = f"{r['size']:>5}"
        for fmt in formats:
            d = r["formats"][fmt]
            gap = d["delim_only_delta_pct"] - d["random_only_delta_mean_pct"]
            marker = "*" if d["sufficient"] == "YES" else ""
            line += f" {gap:>+12.1f}pp{marker}"
        print(line)

    # Check if sufficiency holds at 100/200
    gcf_results = [(r["size"], r["formats"]["gcf_generic"]["sufficient"]) for r in results]
    holds = all(s == "YES" for _, s in gcf_results if _ >= 100)
    if holds:
        conclusion = "SUFFICIENCY HOLDS: delimiter heads outperform random at all scales including 100/200 rows"
    else:
        conclusion = "SUFFICIENCY BREAKS: at larger payloads, other heads become necessary"
    print(f"\nConclusion: {conclusion}")

    return {
        "experiment": "sufficiency_scaling",
        "delimiter_heads": len(delimiter_heads_a),
        "total_heads": total_heads,
        "sizes_tested": sizes,
        "results": results,
        "conclusion": conclusion,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Remaining ablation experiments (#19, #21, #22)")
    parser.add_argument("--checkpoint-a", required=True, help="Model A (merge barriers) checkpoint")
    parser.add_argument("--tokenizer-a", required=True, help="Model A tokenizer")
    parser.add_argument("--checkpoint-b", required=True, help="Model B (standard BPE) checkpoint")
    parser.add_argument("--tokenizer-b", required=True, help="Model B tokenizer")
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

    metadata = {
        "experiment": "remaining_ablation_combined",
        "description": "Three remaining ablation experiments: embedding space (#19), adversarial robustness (#21), sufficiency scaling (#22)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

    print("=" * 90)
    print("REMAINING ABLATION EXPERIMENTS (#19, #21, #22)")
    print("=" * 90)
    print(f"\nTimestamp: {metadata['timestamp_utc']}")
    print(f"Device: {device}")
    if "gpu_name" in metadata:
        print(f"GPU: {metadata['gpu_name']} ({metadata['gpu_memory_mb']} MB)")
    print(f"PyTorch: {metadata['torch_version']}")

    # Load models
    print("\nLoading Model A (merge barriers)...")
    model_a, tok_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    print("Loading Model B (standard BPE)...")
    model_b, tok_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)
    model_b.to(device)

    # Identify delimiter heads (Model A only)
    print("\nIdentifying delimiter heads on Model A...")
    delimiter_heads_a = identify_delimiter_heads(model_a, tok_a, device)
    print(f"Found {len(delimiter_heads_a)} delimiter-majority heads")

    # Run experiments
    results = {"metadata": metadata, "delimiter_heads_count": len(delimiter_heads_a)}

    # #19: Embedding space
    results["embedding_space"] = run_embedding_analysis(model_a, tok_a, delimiter_heads_a, device)

    # Reload model_a (ablation modified it via deepcopy, but just in case)
    # Actually deepcopy means original is untouched

    # #21: Adversarial robustness (needs both models)
    results["adversarial_robustness"] = run_adversarial_robustness(
        model_a, tok_a, model_b, tok_b, delimiter_heads_a, device)

    # Free model B
    del model_b
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # #22: Sufficiency scaling (model A only)
    results["sufficiency_scaling"] = run_sufficiency_scaling(model_a, tok_a, delimiter_heads_a, device)

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nResults saved to {args.output}")

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
        if args.output and os.path.exists(args.output):
            s3.upload_file(args.output, "structok-training",
                          "logs/run-002-ablation/remaining-ablation-results.json")
            print("  Uploaded remaining-ablation-results.json", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training",
                          "logs/run-002-ablation/remaining-ablation-log.txt")
            print("  Uploaded remaining-ablation-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)

    print("\n" + "=" * 90)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 90)

    # Print final summary
    print(f"\n#19 Embedding space: {results['embedding_space']['conclusion']}")
    print(f"#21 Adversarial:     {results['adversarial_robustness']['conclusion']}")
    print(f"#22 Sufficiency:     {results['sufficiency_scaling']['conclusion']}")


if __name__ == "__main__":
    main()
