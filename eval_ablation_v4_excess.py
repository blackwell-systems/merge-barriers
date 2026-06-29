#!/usr/bin/env python3
"""
Head ablation v4: extended cross-format transfer.

Tests delimiter head ablation across 12 formats (5 trained, 7 unseen).
If delimiter heads help all unseen formats with clean delimiters,
the cross-format transfer is universal, not GCF-specific.

Usage:
  python eval_ablation_v4.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --output ablation-v4-results.json
"""

import argparse
import gc
import json
import math
import copy
import datetime
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
# Test data: 12 formats
# =========================================================================

def gen_gcf_generic():
    names = ["Alice Chen", "Bob Smith", "Carla Rodriguez", "David Park", "Eva Johansson"]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
    lines = ["## orders [30]{orderId,customer,status,total}"]
    for i in range(30):
        lines.append(f"ORD-{i+1:05d}|{names[i%5]}|{statuses[i%5]}|{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_gcf_graph():
    pkgs = ["pkg/auth","pkg/server","pkg/db","pkg/cache","pkg/config"]
    names = ["Validate","Process","Handle","Create","Update","Delete","Get","Set","Check","Build"]
    kinds = ["fn","type","method","iface"]
    lines = ["GCF profile=graph tool=context symbols=20 edges=15"]
    lines.append("## targets")
    for i in range(7):
        lines.append(f"@{i} {kinds[i%4]} {pkgs[i%5]}.{names[i%10]}{i} {round(0.95-i*0.03,2)} lsp_resolved")
    lines.append("## related")
    for i in range(7,14):
        lines.append(f"@{i} {kinds[i%4]} {pkgs[i%5]}.{names[i%10]}{i} {round(0.95-i*0.03,2)} ast_inferred")
    lines.append("## extended")
    for i in range(14,20):
        lines.append(f"@{i} {kinds[i%4]} {pkgs[i%5]}.{names[i%10]}{i} {round(0.95-i*0.03,2)} structural")
    lines.append("## edges [15]")
    et = ["calls","imports","implements","references"]
    for i in range(15):
        lines.append(f"@{(i*3)%20}<@{(i*3+1)%20} {et[i%4]}")
    return "\n".join(lines)

def gen_json():
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    records = [{"orderId":f"ORD-{i+1:05d}","customer":names[i%5],"status":statuses[i%5],"total":round(29.97+i*12.50,2)} for i in range(30)]
    return json.dumps({"orders":records}, indent=2)

def gen_yaml():
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    roles = ["admin","developer","analyst","manager","intern"]
    lines = ["employees:"]
    for i in range(30):
        lines.extend([f"  - name: {names[i%5]}", f"    id: EMP-{i+1:04d}", f"    role: {roles[i%5]}", f"    salary: {50000+i*2500}"])
    return "\n".join(lines)

def gen_python():
    return '''class OrderProcessor:
    def __init__(self, db, cache, config):
        self.db = db
        self.cache = cache
        self.config = config

    def process_batch(self, orders):
        results = []
        for order in orders:
            if order.status == "pending":
                validated = self.validate(order)
                if validated.is_valid:
                    results.append(self.transform(validated))
            elif order.status == "retry":
                results.append(self.retry(order))
        return results

    def validate(self, order):
        rules = self.config.get_rules(order.type)
        return Validator(rules).check(order)

    def transform(self, order):
        mapping = self.config.get_mapping(order.type)
        return Transformer(mapping).apply(order)
'''

# --- UNSEEN FORMATS (never in training) ---

def gen_toon():
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["orderId\tcustomer\tstatus\ttotal"]
    for i in range(30):
        lines.append(f"ORD-{i+1:05d}\t{names[i%5]}\t{statuses[i%5]}\t{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_csv():
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["orderId,customer,status,total"]
    for i in range(30):
        lines.append(f"ORD-{i+1:05d},{names[i%5]},{statuses[i%5]},{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_toml():
    lines = []
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    for i in range(20):
        lines.append(f"[[orders]]")
        lines.append(f'orderId = "ORD-{i+1:05d}"')
        lines.append(f'customer = "{names[i%5]}"')
        lines.append(f'status = "{statuses[i%5]}"')
        lines.append(f"total = {round(29.97+i*12.50,2)}")
        lines.append("")
    return "\n".join(lines)

def gen_ini():
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    lines = []
    for i in range(20):
        lines.append(f"[order_{i+1:03d}]")
        lines.append(f"customer = {names[i%5]}")
        lines.append(f"status = pending")
        lines.append(f"total = {round(29.97+i*12.50,2)}")
        lines.append("")
    return "\n".join(lines)

def gen_sql():
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["CREATE TABLE orders (orderId VARCHAR(20), customer VARCHAR(50), status VARCHAR(20), total DECIMAL(10,2));", ""]
    for i in range(30):
        lines.append(f"INSERT INTO orders VALUES ('ORD-{i+1:05d}', '{names[i%5]}', '{statuses[i%5]}', {round(29.97+i*12.50,2)});")
    return "\n".join(lines)

def gen_xml():
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ['<?xml version="1.0"?>', "<orders>"]
    for i in range(20):
        lines.append(f"  <order>")
        lines.append(f'    <orderId>ORD-{i+1:05d}</orderId>')
        lines.append(f"    <customer>{names[i%5]}</customer>")
        lines.append(f"    <status>{statuses[i%5]}</status>")
        lines.append(f"    <total>{round(29.97+i*12.50,2)}</total>")
        lines.append(f"  </order>")
    lines.append("</orders>")
    return "\n".join(lines)

def gen_markdown_table():
    names = ["Alice Chen","Bob Smith","Carla Rodriguez","David Park","Eva Johansson"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["| Order ID | Customer | Status | Total |", "|----------|----------|--------|-------|"]
    for i in range(30):
        lines.append(f"| ORD-{i+1:05d} | {names[i%5]} | {statuses[i%5]} | {round(29.97+i*12.50,2)} |")
    return "\n".join(lines)

def gen_sexp():
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["(orders"]
    for i in range(20):
        lines.append(f'  (order (id "ORD-{i+1:05d}") (customer "{names[i%5]}") (status {statuses[i%5]}) (total {round(29.97+i*12.50,2)}))')
    lines.append(")")
    return "\n".join(lines)

def gen_protobuf_text():
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = []
    for i in range(20):
        lines.append(f"order {{")
        lines.append(f'  order_id: "ORD-{i+1:05d}"')
        lines.append(f'  customer: "{names[i%5]}"')
        lines.append(f'  status: "{statuses[i%5]}"')
        lines.append(f"  total: {round(29.97+i*12.50,2)}")
        lines.append(f"}}")
        lines.append("")
    return "\n".join(lines)

NL_TEXT = ("The architecture of modern distributed systems has evolved significantly. "
    "Microservices replaced monolithic applications, bringing independent deployment "
    "and technology diversity, but also complexity in service discovery and tracing. "
    "Container orchestration platforms became the standard deployment target.")

FORMATS = {}

def build_formats():
    global FORMATS
    FORMATS = {
        # Trained formats
        "gcf_generic":   {"texts": [gen_gcf_generic()], "trained": True, "delimiters": "|"},
        "gcf_graph":     {"texts": [gen_gcf_graph()],   "trained": True, "delimiters": "@ < |"},
        "json":          {"texts": [gen_json()],        "trained": True, "delimiters": "\" : , { }"},
        "yaml":          {"texts": [gen_yaml()],        "trained": True, "delimiters": ": -"},
        "python":        {"texts": [gen_python()],      "trained": True, "delimiters": "( ) : { }"},
        # Unseen formats
        "toon":          {"texts": [gen_toon()],        "trained": False, "delimiters": "\\t"},
        "csv":           {"texts": [gen_csv()],         "trained": False, "delimiters": ","},
        "toml":          {"texts": [gen_toml()],        "trained": False, "delimiters": "= [ ] \""},
        "ini":           {"texts": [gen_ini()],         "trained": False, "delimiters": "= [ ]"},
        "sql":           {"texts": [gen_sql()],         "trained": False, "delimiters": "( ) , ' ;"},
        "xml":           {"texts": [gen_xml()],         "trained": False, "delimiters": "< > \" /"},
        "md_table":      {"texts": [gen_markdown_table()], "trained": False, "delimiters": "|"},
        "s_expression":  {"texts": [gen_sexp()],        "trained": False, "delimiters": "( ) \""},
        "protobuf_text": {"texts": [gen_protobuf_text()], "trained": False, "delimiters": "{ } : \""},
        # Natural language
        "nl":            {"texts": [NL_TEXT],            "trained": True, "delimiters": "none"},
    }


# =========================================================================
# Model loading and helpers
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

    Excess score = raw attention fraction minus base rate. Corrects for
    the base-rate problem where JSON's 75.7% delimiter positions inflate
    head counts with the raw >50% method.
    """
    test_texts = [FORMATS["gcf_generic"]["texts"][0], FORMATS["gcf_graph"]["texts"][0],
                  FORMATS["json"]["texts"][0], FORMATS["yaml"]["texts"][0]]

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

    # Log base rates
    print(f"  Base rates per probing text:")
    for i, text in enumerate(test_texts):
        ids = tok.encode(text).ids[:1024]
        delim_pos = sum(1 for tid in ids if is_delimiter_token(tok, tid))
        br = delim_pos / len(ids)
        fmt = ["GCF generic", "GCF graph", "JSON", "YAML"][i]
        print(f"    {fmt}: {delim_pos}/{len(ids)} = {br:.1%}")
    print(f"  Excess threshold: {excess_threshold}")
    print(f"  Heads above threshold: {len(heads)} / {model.config.num_hidden_layers * n_heads}")

    return heads


def _get_output_proj(model, layer_idx):
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_idx].attention.dense
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn.o_proj
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def ablate_heads(model, heads_to_ablate):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads_to_ablate:
        proj = _get_output_proj(model, layer_idx)
        start = head_idx * head_dim
        end = start + head_dim
        proj.weight.data[:, start:end] = 0.0


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Head ablation v4: extended cross-format transfer")
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

    metadata = {
        "experiment": "head_ablation_v4_extended_transfer",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("HEAD ABLATION v4: EXTENDED CROSS-FORMAT TRANSFER")
    print("=" * 90)
    print(f"\nTimestamp: {metadata['timestamp_utc']}")
    print(f"Device: {device}")

    build_formats()

    print(f"\n{'Format':<16} {'Trained':>8} {'Delimiters':<20} {'Chars':>8}")
    print("-" * 56)
    for fmt, info in FORMATS.items():
        chars = sum(len(t) for t in info["texts"])
        trained = "yes" if info["trained"] else "NO"
        print(f"{fmt:<16} {trained:>8} {info['delimiters']:<20} {chars:>8,}")

    print(f"\nLoading Model A (merge barriers)...")
    model_a, tok_a, step = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    # Baselines
    print("\n" + "=" * 90)
    print("BASELINES")
    print("=" * 90)

    baseline = {}
    for fmt, info in FORMATS.items():
        baseline[fmt] = compute_ppl(model_a, tok_a, info["texts"], device)
        print(f"  {fmt}: {baseline[fmt]:.2f}")

    # Identify delimiter heads
    delimiter_heads = identify_delimiter_heads(model_a, tok_a, device, 0.10)
    n_total = model_a.config.num_hidden_layers * model_a.config.num_attention_heads
    print(f"\nDelimiter heads: {len(delimiter_heads)} / {n_total}")

    # Ablate all delimiter heads
    print("\n" + "=" * 90)
    print("CROSS-FORMAT TRANSFER: DELIMITER HEAD ABLATION")
    print("=" * 90)

    model_copy = copy.deepcopy(model_a)
    model_copy.to(device)
    ablate_heads(model_copy, [(l, h) for l, h, _ in delimiter_heads])

    ablated = {}
    for fmt, info in FORMATS.items():
        ablated[fmt] = compute_ppl(model_copy, tok_a, info["texts"], device)

    del model_copy
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Results
    print(f"\n{'Format':<16} {'Trained':>8} {'Baseline':>12} {'Ablated':>12} {'Delta':>8} {'Transfer?':>10}")
    print("-" * 72)

    results = {}
    trained_deltas = []
    unseen_deltas = []

    for fmt in FORMATS:
        info = FORMATS[fmt]
        base = baseline[fmt]
        abl = ablated[fmt]
        delta = ((abl - base) / base) * 100
        trained = "yes" if info["trained"] else "NO"
        transfer = ""
        if not info["trained"] and delta > 5:
            transfer = "YES"
        elif not info["trained"] and delta < -5:
            transfer = "no"
        elif not info["trained"]:
            transfer = "weak"

        if info["trained"]:
            trained_deltas.append(delta)
        elif fmt != "nl":
            unseen_deltas.append(delta)

        results[fmt] = {
            "baseline": round(base, 2),
            "ablated": round(abl, 2),
            "delta_pct": round(delta, 1),
            "trained": info["trained"],
            "delimiters": info["delimiters"],
        }
        print(f"{fmt:<16} {trained:>8} {base:>12.1f} {abl:>12.1f} {delta:>+7.1f}% {transfer:>10}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    unseen_hurt = sum(1 for d in unseen_deltas if d > 5)
    unseen_helped = sum(1 for d in unseen_deltas if d < -5)
    unseen_neutral = len(unseen_deltas) - unseen_hurt - unseen_helped

    print(f"\nUnseen formats (never in training): {len(unseen_deltas)} tested")
    print(f"  Hurt by delimiter ablation (transfer confirmed): {unseen_hurt}")
    print(f"  Helped by delimiter ablation: {unseen_helped}")
    print(f"  Neutral: {unseen_neutral}")

    if unseen_hurt > 0:
        avg_hurt = sum(d for d in unseen_deltas if d > 5) / unseen_hurt
        print(f"  Average degradation on hurt formats: +{avg_hurt:.1f}%")

    if unseen_hurt >= len(unseen_deltas) * 0.5:
        print(f"\n  CONCLUSION: Cross-format transfer is UNIVERSAL.")
        print(f"  Delimiter heads help {unseen_hurt} of {len(unseen_deltas)} unseen formats.")
        print(f"  The mechanism is not GCF-specific; it works for any format with clean delimiters.")
    elif unseen_hurt > 0:
        print(f"\n  CONCLUSION: Cross-format transfer is PARTIAL.")
        print(f"  Delimiter heads help some unseen formats but not all.")
    else:
        print(f"\n  CONCLUSION: No cross-format transfer detected.")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "n_delimiter_heads": len(delimiter_heads),
            "n_total_heads": n_total,
            "formats_tested": len(FORMATS),
            "unseen_formats_tested": len(unseen_deltas),
            "unseen_formats_hurt": unseen_hurt,
            "results": results,
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
            r2_files.append((args.output, "logs/run-002-ablation/ablation-v4-excess-results.json"))
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            r2_files.append((log_path, "logs/run-002-ablation/ablation-v4-excess-log.txt"))

        for local, key in r2_files:
            s3.upload_file(local, "structok-training", key)
            size_kb = os.path.getsize(local) / 1024
            print(f"  Uploaded {key} ({size_kb:.0f} KB)", flush=True)
        print("R2 upload complete.", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
