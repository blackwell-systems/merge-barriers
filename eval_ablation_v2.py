#!/usr/bin/env python3
"""
Head ablation v2: larger test set, multiple control seeds, per-format PPL.

Addresses the concern that v1 results (removing any heads improves PPL)
may be a small-sample regularization artifact.

Changes from v1:
  - 10x more test data per format (50-100 rows instead of 5-10)
  - 5 random seeds for control ablation (not just 1)
  - Reports mean and std across control runs
  - Adds YAML and code test data alongside GCF/JSON

Usage:
  python eval_ablation_v2.py \
    --checkpoint-a /root/checkpoint-a.pt --tokenizer-a /root/structok-64k.json \
    --checkpoint-b /root/checkpoint-b.pt --tokenizer-b /root/standard-64k.json \
    --output /root/ablation-v2-results.json
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
# Generate larger test data programmatically
# =========================================================================

def generate_gcf_generic(n_rows=50):
    """Generate a GCF generic profile with n_rows."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson",
             "Fiona Grant", "George Wu", "Hannah Lee", "Ivan Petrov", "Julia Santos"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = [f"## orders [{n_rows}]{{orderId,customer,status,total}}"]
    for i in range(n_rows):
        name = names[i % len(names)]
        status = statuses[i % len(statuses)]
        total = round(29.97 + i * 12.50, 2)
        lines.append(f"ORD-{i+1:05d}|{name}|{status}|{total}")
    return "\n".join(lines)


def generate_gcf_graph(n_symbols=30, n_edges=20):
    """Generate a GCF graph profile."""
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
        et = edge_types[i % len(edge_types)]
        lines.append(f"@{tgt}<@{src} {et}")

    return "\n".join(lines)


def generate_json(n_rows=50):
    """Generate JSON array with n_rows."""
    names = ["Alice", "Bob", "Carla", "David", "Eva",
             "Fiona", "George", "Hannah", "Ivan", "Julia"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    records = []
    for i in range(n_rows):
        records.append({
            "orderId": f"ORD-{i+1:05d}",
            "customer": names[i % len(names)],
            "status": statuses[i % len(statuses)],
            "total": round(29.97 + i * 12.50, 2),
        })
    return json.dumps({"orders": records}, indent=2)


def generate_yaml(n_rows=50):
    """Generate YAML-like structured data."""
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    roles = ["admin", "developer", "analyst", "manager", "intern"]
    lines = ["employees:"]
    for i in range(n_rows):
        lines.append(f"  - name: {names[i % len(names)]}")
        lines.append(f"    id: EMP-{i+1:04d}")
        lines.append(f"    role: {roles[i % len(roles)]}")
        lines.append(f"    salary: {50000 + i * 2500}")
        lines.append(f"    active: {'true' if i % 3 != 0 else 'false'}")
    return "\n".join(lines)


def generate_code():
    """Generate a realistic code snippet."""
    return '''def process_batch(items, config):
    results = []
    for item in items:
        if item.status == "pending":
            validated = validate_item(item, config.rules)
            if validated.is_valid:
                result = transform_item(validated, config.mappings)
                results.append(result)
            else:
                log_error(f"Validation failed for {item.id}: {validated.errors}")
        elif item.status == "retry":
            retried = retry_with_backoff(item, max_attempts=config.max_retries)
            if retried.success:
                results.append(retried.result)

    summary = BatchSummary(
        total=len(items),
        processed=len(results),
        failed=len(items) - len(results),
        duration_ms=timer.elapsed(),
    )
    return results, summary


class BatchProcessor:
    def __init__(self, config, db_client, cache):
        self.config = config
        self.db = db_client
        self.cache = cache
        self._lock = threading.Lock()

    async def run(self, batch_id):
        items = await self.db.fetch_batch(batch_id)
        cached = self.cache.get(batch_id)
        if cached and cached.version == items.version:
            return cached.results

        with self._lock:
            results, summary = process_batch(items.records, self.config)
            self.cache.set(batch_id, results, ttl=3600)
            await self.db.update_status(batch_id, "completed")

        return results
'''


NL_TEXTS = [
    "The architecture of modern distributed systems has evolved significantly over the past decade. "
    "Microservices replaced monolithic applications, bringing benefits like independent deployment and "
    "technology diversity, but also introducing complexity in service discovery, circuit breaking, and "
    "distributed tracing. The industry has largely converged on container orchestration platforms as "
    "the standard deployment target, with service meshes providing cross-cutting concerns like mutual "
    "TLS, load balancing, and observability without requiring changes to application code.",

    "Natural language processing has been transformed by the introduction of transformer architectures. "
    "Pre-training on large corpora followed by task-specific fine-tuning has become the dominant paradigm. "
    "Recent work on instruction tuning and reinforcement learning from human feedback has improved the "
    "alignment of language models with human preferences, enabling more natural conversational interactions "
    "and more reliable adherence to complex instructions.",

    "The economic implications of artificial intelligence adoption vary significantly across industries. "
    "Manufacturing and logistics have seen productivity gains from automation and predictive maintenance, "
    "while knowledge work sectors are experiencing a more nuanced transformation. Creative industries "
    "are grappling with questions of intellectual property and the role of human judgment in an "
    "increasingly automated production pipeline. The labor market is adjusting, with demand shifting "
    "toward roles that combine domain expertise with AI literacy.",
]

FORMAT_TEXTS = {}


def build_test_data():
    global FORMAT_TEXTS
    FORMAT_TEXTS = {
        "gcf_generic": [generate_gcf_generic(50), generate_gcf_generic(30)],
        "gcf_graph": [generate_gcf_graph(30, 20), generate_gcf_graph(20, 15)],
        "json": [generate_json(50), generate_json(30)],
        "yaml": [generate_yaml(30)],
        "code": [generate_code()],
    }


# =========================================================================
# Model loading and helpers (same as v1)
# =========================================================================

def load_model(checkpoint_path, size, tokenizer_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"

    is_llama = "llama" in size
    if is_llama:
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
    return model, tok, step


def is_delimiter_token(tok, token_id):
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    token_str = id_to_token.get(token_id, "")
    return any(c in token_str for c in BARRIER_CHARS)


def compute_ppl(model, tok, texts, device):
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        ids = tok.encode(text).ids
        # Truncate to max_position_embeddings
        ids = ids[:2048]
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
    return math.exp(min(avg_loss, 20))  # cap to avoid overflow


def identify_delimiter_heads(model, tok, device, excess_threshold=0.15):
    """Identify delimiter-specialized heads using excess scores."""
    all_texts = []
    for texts in FORMAT_TEXTS.values():
        all_texts.extend(texts)

    n_heads = model.config.num_attention_heads
    head_excess_scores = {}

    for text in all_texts[:4]:
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
        avg = sum(scores) / len(scores)
        if avg > excess_threshold:
            heads.append((l, h, avg))

    heads.sort(key=lambda x: x[2], reverse=True)
    print(f"  Excess threshold: {excess_threshold}")
    print(f"  Heads above threshold: {len(heads)} / {model.config.num_hidden_layers * n_heads}")
    return heads


def _get_output_proj(model, layer_idx):
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_idx].attention.dense
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn.o_proj
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")


def ablate_heads(model, heads_to_ablate):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads_to_ablate:
        proj = _get_output_proj(model, layer_idx)
        start = head_idx * head_dim
        end = start + head_dim
        proj.weight.data[:, start:end] = 0.0


def measure_all_formats(model, tok, device):
    result = {}
    for fmt, texts in FORMAT_TEXTS.items():
        result[fmt] = compute_ppl(model, tok, texts, device)
    result["nl"] = compute_ppl(model, tok, NL_TEXTS, device)
    return result


# =========================================================================
# Main experiment
# =========================================================================

def collect_metadata(args, device):
    """Collect full experiment metadata for reproducibility."""
    import platform
    import datetime
    meta = {
        "experiment": "head_ablation_v2",
        "description": "Progressive ablation of delimiter-specialized attention heads to test causal role in structured data comprehension",
        "hypothesis": "Delimiter-majority heads are causally responsible for structured data comprehension in merge-barrier models",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "args": vars(args),
        "device": device,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    try:
        import transformers
        meta["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return meta


def main():
    parser = argparse.ArgumentParser(description="Head ablation v2")
    parser.add_argument("--checkpoint-a", required=True, help="Model A (merge barriers) checkpoint")
    parser.add_argument("--tokenizer-a", required=True, help="Model A tokenizer")
    parser.add_argument("--checkpoint-b", required=True, help="Model B (standard BPE) checkpoint")
    parser.add_argument("--tokenizer-b", required=True, help="Model B tokenizer")
    parser.add_argument("--size", default="410m", help="Model size config")
    parser.add_argument("--device", default=None, help="Device (auto-detected)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--control-seeds", type=int, default=5, help="Number of random seeds for control ablation")
    parser.add_argument("--ablate-model", default="a", choices=["a", "b", "both"], help="Which model to ablate (a=merge barriers, b=standard, both)")
    parser.add_argument("--delimiter-threshold", type=float, default=0.5, help="Threshold for delimiter-majority head classification")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    metadata = collect_metadata(args, device)

    print("=" * 90)
    print("HEAD ABLATION EXPERIMENT v2")
    print("=" * 90)
    print(f"\nTimestamp: {metadata['timestamp_utc']}")
    print(f"Device: {device}")
    if 'gpu_name' in metadata:
        print(f"GPU: {metadata['gpu_name']} ({metadata['gpu_memory_mb']} MB)")
    print(f"PyTorch: {metadata['torch_version']}, CUDA: {metadata.get('cuda_version', 'N/A')}")
    print(f"Ablating model: {args.ablate_model}")
    print(f"Control seeds: {args.control_seeds}")
    print(f"Delimiter threshold: {args.delimiter_threshold}")

    build_test_data()

    # Print test data stats
    print("\nTest data:")
    for fmt, texts in FORMAT_TEXTS.items():
        total_chars = sum(len(t) for t in texts)
        print(f"  {fmt}: {len(texts)} texts, {total_chars:,} chars")
    print(f"  nl: {len(NL_TEXTS)} texts, {sum(len(t) for t in NL_TEXTS):,} chars")

    print("\nLoading Model A (merge barriers)...")
    model_a, tok_a, step_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    print("Loading Model B (standard BPE)...")
    model_b, tok_b, step_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)
    model_b.to(device)

    # Phase 1: identify heads
    print("\n" + "=" * 90)
    print("PHASE 1: Identify delimiter-specialized heads")
    print("=" * 90)

    delimiter_heads = identify_delimiter_heads(model_a, tok_a, device)
    n_total = model_a.config.num_hidden_layers * model_a.config.num_attention_heads
    print(f"\nDelimiter-majority heads (>50%): {len(delimiter_heads)} / {n_total}")
    for layer, head, score in delimiter_heads[:10]:
        print(f"  Layer {layer:>2}, Head {head:>2}: {score:.1%}")

    # Phase 2: baselines
    print("\n" + "=" * 90)
    print("PHASE 2: Baselines")
    print("=" * 90)

    baseline_a = measure_all_formats(model_a, tok_a, device)
    baseline_b = measure_all_formats(model_b, tok_b, device)

    print("\nModel A (merge barriers):")
    for fmt, ppl in baseline_a.items():
        print(f"  {fmt}: {ppl:.2f}")
    print("\nModel B (standard BPE):")
    for fmt, ppl in baseline_b.items():
        print(f"  {fmt}: {ppl:.2f}")

    # Move model B off GPU
    model_b.to("cpu")
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Phase 3: delimiter head ablation
    print("\n" + "=" * 90)
    print("PHASE 3: Progressive delimiter head ablation")
    print("=" * 90)

    n_delim = len(delimiter_heads)
    steps = list(range(0, n_delim + 1, max(1, n_delim // 10)))
    if steps[-1] != n_delim:
        steps.append(n_delim)

    fmt_names = list(FORMAT_TEXTS.keys()) + ["nl"]
    header = f"{'Heads':>6}"
    for fmt in fmt_names:
        header += f" {fmt:>12}"
    print(f"\n{header}")
    print("-" * (6 + 13 * len(fmt_names)))

    ablation_results = []
    for n_ablate in steps:
        model_copy = copy.deepcopy(model_a)
        model_copy.to(device)

        if n_ablate > 0:
            heads_to_remove = [(l, h) for l, h, _ in delimiter_heads[:n_ablate]]
            ablate_heads(model_copy, heads_to_remove)

        ppls = measure_all_formats(model_copy, tok_a, device)
        row = {"heads_ablated": n_ablate}
        line = f"{n_ablate:>6}"
        for fmt in fmt_names:
            ppl = ppls[fmt]
            delta = ((ppl - baseline_a[fmt]) / baseline_a[fmt]) * 100
            row[f"{fmt}_ppl"] = round(ppl, 2)
            row[f"{fmt}_delta"] = round(delta, 1)
            line += f" {ppl:>10.1f}{'':>2}"
        print(line)
        ablation_results.append(row)

        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Phase 4: multiple control runs
    print("\n" + "=" * 90)
    print(f"PHASE 4: Control ablation ({args.control_seeds} random seeds)")
    print("=" * 90)

    delim_set = {(l, h) for l, h, _ in delimiter_heads}
    n_layers = model_a.config.num_hidden_layers
    n_heads_per_layer = model_a.config.num_attention_heads
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)]
    non_delim_heads = [(l, h) for l, h in all_heads if (l, h) not in delim_set]

    control_steps = [0, n_delim // 2, n_delim]
    all_control_results = []

    for seed in range(args.control_seeds):
        print(f"\n  Seed {seed}:")
        rng = random.Random(seed)
        shuffled = list(non_delim_heads)
        rng.shuffle(shuffled)

        seed_results = []
        for n_ablate in control_steps:
            model_copy = copy.deepcopy(model_a)
            model_copy.to(device)

            if n_ablate > 0:
                ablate_heads(model_copy, shuffled[:n_ablate])

            ppls = measure_all_formats(model_copy, tok_a, device)
            row = {"seed": seed, "heads_ablated": n_ablate}
            line = f"    {n_ablate:>4} heads:"
            for fmt in fmt_names:
                ppl = ppls[fmt]
                delta = ((ppl - baseline_a[fmt]) / baseline_a[fmt]) * 100
                row[f"{fmt}_ppl"] = round(ppl, 2)
                row[f"{fmt}_delta"] = round(delta, 1)
                line += f"  {fmt}={ppl:.0f}({delta:+.0f}%)"
            print(line)
            seed_results.append(row)

            del model_copy
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()

        all_control_results.append(seed_results)

    # Phase 5: Reverse ablation (keep ONLY delimiter heads)
    print("\n" + "=" * 90)
    print("PHASE 5: Reverse ablation (keep ONLY delimiter heads)")
    print("=" * 90)
    print(f"\nRemoving all {len(non_delim_heads)} non-delimiter heads, keeping {n_delim} delimiter heads.")
    print(f"Tests sufficiency: are delimiter heads alone enough for GCF comprehension?")

    model_copy = copy.deepcopy(model_a)
    model_copy.to(device)
    ablate_heads(model_copy, non_delim_heads)
    reverse_ppls = measure_all_formats(model_copy, tok_a, device)

    print(f"\n{'Format':<14} {'Baseline':>10} {'Delim only':>10} {'Delta':>8}")
    print("-" * 48)

    reverse_results = {}
    for fmt in fmt_names:
        base = baseline_a[fmt]
        ppl = reverse_ppls[fmt]
        delta = ((ppl - base) / base) * 100
        reverse_results[f"{fmt}_ppl"] = round(ppl, 2)
        reverse_results[f"{fmt}_delta"] = round(delta, 1)
        print(f"{fmt:<14} {base:>10.1f} {ppl:>10.1f} {delta:>+7.1f}%")

    del model_copy
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Also test reverse on random heads (keep only N random non-delimiter heads)
    print(f"\nControl: keep {n_delim} random NON-delimiter heads, remove all others")
    rng = random.Random(42)
    shuffled_non_delim = list(non_delim_heads)
    rng.shuffle(shuffled_non_delim)
    keep_random = shuffled_non_delim[:n_delim]
    remove_for_control = [(l, h) for l, h in all_heads if (l, h) not in set(keep_random)]

    model_copy = copy.deepcopy(model_a)
    model_copy.to(device)
    ablate_heads(model_copy, remove_for_control)
    reverse_control_ppls = measure_all_formats(model_copy, tok_a, device)

    reverse_control_results = {}
    print(f"\n{'Format':<14} {'Baseline':>10} {'Rand only':>10} {'Delta':>8}")
    print("-" * 48)
    for fmt in fmt_names:
        base = baseline_a[fmt]
        ppl = reverse_control_ppls[fmt]
        delta = ((ppl - base) / base) * 100
        reverse_control_results[f"{fmt}_ppl"] = round(ppl, 2)
        reverse_control_results[f"{fmt}_delta"] = round(delta, 1)
        print(f"{fmt:<14} {base:>10.1f} {ppl:>10.1f} {delta:>+7.1f}%")

    del model_copy
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Phase 6: Layer-wise ablation
    print("\n" + "=" * 90)
    print("PHASE 6: Layer-wise delimiter head ablation")
    print("=" * 90)

    layer_groups = [
        ("early (0-7)", 0, 7),
        ("middle (8-15)", 8, 15),
        ("late (16-23)", 16, 23),
    ]

    layer_results = {}
    for group_name, layer_start, layer_end in layer_groups:
        group_heads = [(l, h) for l, h, _ in delimiter_heads if layer_start <= l <= layer_end]
        print(f"\n  {group_name}: {len(group_heads)} delimiter heads")

        if len(group_heads) == 0:
            print("    (no delimiter heads in this range)")
            layer_results[group_name] = {"n_heads": 0}
            continue

        model_copy = copy.deepcopy(model_a)
        model_copy.to(device)
        ablate_heads(model_copy, group_heads)
        ppls = measure_all_formats(model_copy, tok_a, device)

        layer_results[group_name] = {"n_heads": len(group_heads)}
        line = "    "
        for fmt in fmt_names:
            ppl = ppls[fmt]
            delta = ((ppl - baseline_a[fmt]) / baseline_a[fmt]) * 100
            layer_results[group_name][f"{fmt}_ppl"] = round(ppl, 2)
            layer_results[group_name][f"{fmt}_delta"] = round(delta, 1)
            line += f"  {fmt}={ppl:.0f}({delta:+.0f}%)"
        print(line)

        del model_copy
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Phase 7: Attention heatmap data
    print("\n" + "=" * 90)
    print("PHASE 7: Attention pattern analysis (top 5 delimiter heads)")
    print("=" * 90)

    top5_heads = delimiter_heads[:5]
    attn_analysis = {}

    for fmt_name, texts in [("gcf_generic", FORMAT_TEXTS["gcf_generic"][:1]), ("json", FORMAT_TEXTS["json"][:1])]:
        text = texts[0]
        ids = tok_a.encode(text).ids[:512]  # cap for memory
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Classify each position
        positions = []
        for i, tid in enumerate(ids):
            is_delim = is_delimiter_token(tok_a, tid)
            positions.append({"pos": i, "is_delimiter": is_delim})

        n_delim_pos = sum(1 for p in positions if p["is_delimiter"])
        n_content_pos = len(positions) - n_delim_pos

        with torch.no_grad():
            outputs = model_a(input_ids=input_ids, output_attentions=True)

        fmt_analysis = {"n_tokens": len(ids), "n_delimiter_positions": n_delim_pos, "n_content_positions": n_content_pos, "heads": []}

        print(f"\n  {fmt_name}: {len(ids)} tokens ({n_delim_pos} delimiter, {n_content_pos} content)")

        for layer, head, score in top5_heads:
            attn_weights = outputs.attentions[layer][0, head].float().cpu()  # (seq, seq)

            # For each query position type, where does attention go?
            delim_to_delim = 0.0
            delim_to_content = 0.0
            content_to_delim = 0.0
            content_to_content = 0.0
            n_d = 0
            n_c = 0

            for q_pos in range(len(ids)):
                q_is_delim = positions[q_pos]["is_delimiter"]
                for k_pos in range(len(ids)):
                    k_is_delim = positions[k_pos]["is_delimiter"]
                    w = attn_weights[q_pos, k_pos].item()

                    if q_is_delim and k_is_delim:
                        delim_to_delim += w
                    elif q_is_delim and not k_is_delim:
                        delim_to_content += w
                    elif not q_is_delim and k_is_delim:
                        content_to_delim += w
                    else:
                        content_to_content += w

                if q_is_delim:
                    n_d += 1
                else:
                    n_c += 1

            # Normalize by query count
            if n_d > 0:
                delim_to_delim /= n_d
                delim_to_content /= n_d
            if n_c > 0:
                content_to_delim /= n_c
                content_to_content /= n_c

            head_data = {
                "layer": layer,
                "head": head,
                "overall_delimiter_score": round(score, 4),
                "delim_query_to_delim_key": round(delim_to_delim, 4),
                "delim_query_to_content_key": round(delim_to_content, 4),
                "content_query_to_delim_key": round(content_to_delim, 4),
                "content_query_to_content_key": round(content_to_content, 4),
            }
            fmt_analysis["heads"].append(head_data)

            print(f"    L{layer}H{head} (score {score:.1%}): d->d={delim_to_delim:.3f} d->c={delim_to_content:.3f} c->d={content_to_delim:.3f} c->c={content_to_content:.3f}")

        attn_analysis[fmt_name] = fmt_analysis

        del outputs
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    # Phase 8: Summary
    print("\n" + "=" * 90)
    print("SUMMARY: Delimiter ablation vs Control (all heads ablated)")
    print("=" * 90)

    final_delim = ablation_results[-1]

    # Average control results at max ablation
    control_at_max = [seed_results[-1] for seed_results in all_control_results]

    print(f"\n{'Format':<14} {'Baseline':>10} {'Delim abl':>10} {'Delim Δ':>8}  {'Ctrl mean':>10} {'Ctrl Δ':>8} {'Ctrl std':>8}")
    print("-" * 78)

    for fmt in fmt_names:
        base = baseline_a[fmt]
        delim_ppl = final_delim[f"{fmt}_ppl"]
        delim_delta = final_delim[f"{fmt}_delta"]

        ctrl_ppls = [r[f"{fmt}_ppl"] for r in control_at_max]
        ctrl_mean = sum(ctrl_ppls) / len(ctrl_ppls)
        ctrl_delta = ((ctrl_mean - base) / base) * 100
        ctrl_std = (sum((p - ctrl_mean) ** 2 for p in ctrl_ppls) / len(ctrl_ppls)) ** 0.5

        print(f"{fmt:<14} {base:>10.1f} {delim_ppl:>10.1f} {delim_delta:>+7.1f}%  {ctrl_mean:>10.1f} {ctrl_delta:>+7.1f}% {ctrl_std:>8.1f}")

    print(f"\nIf delimiter Δ is WORSE than control Δ on GCF but SIMILAR on JSON/NL,")
    print(f"the delimiter heads are specifically important for GCF comprehension.")
    print(f"If both are similar, the heads aren't special.")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "config": {
                "model_size": args.size,
                "delimiter_threshold": args.delimiter_threshold,
                "control_seeds": args.control_seeds,
                "ablate_model": args.ablate_model,
                "checkpoint_a": args.checkpoint_a,
                "checkpoint_b": args.checkpoint_b,
                "tokenizer_a": args.tokenizer_a,
                "tokenizer_b": args.tokenizer_b,
            },
            "test_data_stats": {fmt: {"n_texts": len(texts), "total_chars": sum(len(t) for t in texts)} for fmt, texts in FORMAT_TEXTS.items()},
            "delimiter_heads": {
                "count": len(delimiter_heads),
                "total_heads": n_total,
                "threshold": args.delimiter_threshold,
                "top_20": [{"layer": l, "head": h, "score": round(s, 4)} for l, h, s in delimiter_heads[:20]],
            },
            "baseline_a": {k: round(v, 4) for k, v in baseline_a.items()},
            "baseline_b": {k: round(v, 4) for k, v in baseline_b.items()},
            "ablation_curve": ablation_results,
            "control_results": all_control_results,
            "reverse_ablation": {
                "delimiter_only": reverse_results,
                "random_only_control": reverse_control_results,
            },
            "layer_wise_ablation": layer_results,
            "attention_patterns": attn_analysis,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
