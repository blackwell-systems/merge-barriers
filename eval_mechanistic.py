#!/usr/bin/env python3
"""
Mechanistic analysis: per-token loss, head specialization, corruption detection,
embedding space, cross-format transfer, confidence calibration, few-shot generation.

Usage:
  python eval_mechanistic.py \
    --checkpoint-a checkpoints/structok/checkpoint.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoints/standard/checkpoint.pt --tokenizer-b standard-64k.json \
    --output mechanistic-results.json
"""

import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict

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
    """Check if a token contains barrier characters."""
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    token_str = id_to_token.get(token_id, "")
    return any(c in token_str for c in BARRIER_CHARS)


def get_token_str(tok, token_id):
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    return id_to_token.get(token_id, f"<{token_id}>")


# =========================================================================
# 1. Per-token loss heatmap
# =========================================================================
def eval_per_token_loss(model, tok, device, name):
    """Compute loss at every token position, classify as delimiter vs content."""
    text = """## orders [10]{orderId,customer,status,total}
ORD-00001|Alice Chen|pending|29.97
ORD-00002|Bob Smith|processing|42.47
ORD-00003|Carla Rodriguez|shipped|54.97
ORD-00004|David Park|delivered|67.47
ORD-00005|Eva Johansson|cancelled|79.97
ORD-00006|Alice Chen|pending|92.47
ORD-00007|Bob Smith|processing|104.97
ORD-00008|Carla Rodriguez|shipped|117.47
ORD-00009|David Park|delivered|129.97
ORD-00010|Eva Johansson|cancelled|142.47"""

    ids = tok.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Per-token cross-entropy loss
    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    per_token_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')

    delimiter_losses = []
    content_losses = []
    token_details = []

    for i in range(len(per_token_loss)):
        target_id = ids[i + 1]
        loss_val = per_token_loss[i].item()
        token_str = get_token_str(tok, target_id)
        is_delim = is_delimiter_token(tok, target_id)

        if is_delim:
            delimiter_losses.append(loss_val)
        else:
            content_losses.append(loss_val)

        token_details.append({
            "pos": i + 1,
            "token": token_str[:20],
            "loss": round(loss_val, 4),
            "type": "delimiter" if is_delim else "content",
        })

    avg_delim = sum(delimiter_losses) / max(len(delimiter_losses), 1)
    avg_content = sum(content_losses) / max(len(content_losses), 1)

    print(f"    Avg delimiter loss: {avg_delim:.4f} (n={len(delimiter_losses)})")
    print(f"    Avg content loss:   {avg_content:.4f} (n={len(content_losses)})")
    print(f"    Delimiter/content ratio: {avg_delim / max(avg_content, 1e-10):.2f}x")

    # Show top-5 highest loss tokens
    sorted_tokens = sorted(token_details, key=lambda x: x["loss"], reverse=True)
    print(f"    Top-5 highest loss:")
    for t in sorted_tokens[:5]:
        print(f"      pos {t['pos']:>3}: loss {t['loss']:.4f} [{t['type']}] '{t['token']}'")

    return {
        "avg_delimiter_loss": avg_delim,
        "avg_content_loss": avg_content,
        "n_delimiter": len(delimiter_losses),
        "n_content": len(content_losses),
        "top5_loss": sorted_tokens[:5],
    }


# =========================================================================
# 2. Head specialization
# =========================================================================
def eval_head_specialization(model, tok, device, name):
    """Find which attention heads specialize in delimiter tokens."""
    text = """## products [5]{productId,name,category,price,inStock}
PROD-44821|Monitor|electronics|299.99|true
PROD-18374|Keyboard|peripherals|89.50|false
PROD-92651|Mouse|peripherals|45.00|true
PROD-37284|Headset|accessories|129.99|true
PROD-56738|Stand|accessories|34.99|true"""

    ids = tok.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Find delimiter positions
    delim_positions = [i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid)]
    content_positions = [i for i, tid in enumerate(ids) if not is_delimiter_token(tok, tid)]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    attentions = outputs.attentions
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    # For each head, compute fraction of attention going to delimiters
    head_delimiter_scores = {}
    delimiter_heads = []

    delim_set = set(delim_positions)
    for layer_idx, attn in enumerate(attentions):
        for head_idx in range(n_heads):
            # Average attention to delimiters across all query positions
            attn_weights = attn[0, head_idx].float().cpu()  # (seq_len, seq_len)
            delim_attn = sum(attn_weights[:, d].mean().item() for d in delim_positions) if delim_positions else 0
            total_attn = delim_attn + sum(attn_weights[:, c].mean().item() for c in content_positions) if content_positions else 1

            score = delim_attn / max(total_attn, 1e-10)
            head_delimiter_scores[(layer_idx, head_idx)] = score

            if score > 0.5:  # Head allocates majority to delimiters
                delimiter_heads.append((layer_idx, head_idx, score))

    # Top 10 delimiter-focused heads
    sorted_heads = sorted(head_delimiter_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"    Total heads: {n_layers * n_heads}")
    print(f"    Delimiter-majority heads (>50%): {len(delimiter_heads)}")
    print(f"    Top-10 delimiter-focused heads:")
    for (l, h), score in sorted_heads[:10]:
        print(f"      Layer {l:>2}, Head {h:>2}: {score:.1%} attention to delimiters")

    # Bottom 5 (least delimiter attention)
    print(f"    Bottom-5 (least delimiter attention):")
    for (l, h), score in sorted_heads[-5:]:
        print(f"      Layer {l:>2}, Head {h:>2}: {score:.1%}")

    return {
        "n_delimiter_heads": len(delimiter_heads),
        "top10": [{"layer": l, "head": h, "score": s} for (l, h), s in sorted_heads[:10]],
        "avg_delimiter_score": sum(head_delimiter_scores.values()) / len(head_delimiter_scores),
    }


# =========================================================================
# 3. Corruption detection
# =========================================================================
def eval_corruption_detection(model, tok, device, name):
    """Feed corrupted GCF, check if loss spikes at corruption point."""
    clean = """## items [5]{id,name,price}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003|Sensor|125.00
ITEM-004|Module|67.25
ITEM-005|Adapter|9.99"""

    corruptions = {
        "wrong_delimiter": """## items [5]{id,name,price}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003,Sensor,125.00
ITEM-004|Module|67.25
ITEM-005|Adapter|9.99""",

        "missing_field": """## items [5]{id,name,price}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003|Sensor
ITEM-004|Module|67.25
ITEM-005|Adapter|9.99""",

        "extra_pipe": """## items [5]{id,name,price}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003|Sensor||125.00
ITEM-004|Module|67.25
ITEM-005|Adapter|9.99""",

        "wrong_count": """## items [5]{id,name,price}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003|Sensor|125.00""",

        "broken_header": """## items [5]{id,name price}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003|Sensor|125.00
ITEM-004|Module|67.25
ITEM-005|Adapter|9.99""",
    }

    def compute_ppl(text):
        ids = tok.encode(text).ids
        if len(ids) > 2048:
            ids = ids[:2048]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
        return math.exp(min(outputs.loss.item(), 20))

    clean_ppl = compute_ppl(clean)
    print(f"    Clean PPL: {clean_ppl:.1f}")

    results = {"clean_ppl": clean_ppl, "corruptions": {}}
    for corruption_type, text in corruptions.items():
        ppl = compute_ppl(text)
        spike = ppl / clean_ppl
        results["corruptions"][corruption_type] = {"ppl": ppl, "spike_ratio": spike}
        detected = "DETECTED" if spike > 1.5 else "missed"
        print(f"    {corruption_type:<20} PPL {ppl:>10.1f} (spike: {spike:.2f}x) {detected}")

    return results


# =========================================================================
# 4. Embedding space analysis
# =========================================================================
def eval_embedding_space(model, tok, device, name):
    """Analyze how delimiter vs content token embeddings are organized."""
    # Get embedding matrix
    embeddings = model.gpt_neox.embed_in.weight.detach().cpu().float().numpy()
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    # Classify all tokens
    delimiter_ids = []
    content_ids = []
    for tid in range(min(len(embeddings), tok.get_vocab_size())):
        token_str = id_to_token.get(tid, "")
        if any(c in token_str for c in BARRIER_CHARS):
            delimiter_ids.append(tid)
        elif token_str.strip():
            content_ids.append(tid)

    if not delimiter_ids or not content_ids:
        print(f"    No delimiter/content split found")
        return {}

    delim_embeds = embeddings[delimiter_ids]
    content_sample = random.sample(content_ids, min(500, len(content_ids)))
    content_embeds = embeddings[content_sample]

    # Compute within-group and between-group cosine similarity
    def avg_cosine_sim(vecs):
        if len(vecs) < 2:
            return 0.0
        norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        sims = norms @ norms.T
        n = len(vecs)
        return (sims.sum() - n) / (n * (n - 1)) if n > 1 else 0.0

    delim_internal_sim = avg_cosine_sim(delim_embeds[:100])  # sample for speed
    content_internal_sim = avg_cosine_sim(content_embeds[:100])

    # Between-group: avg cosine sim between delimiter and content embeddings
    delim_norms = delim_embeds[:100] / (np.linalg.norm(delim_embeds[:100], axis=1, keepdims=True) + 1e-10)
    content_norms = content_embeds[:100] / (np.linalg.norm(content_embeds[:100], axis=1, keepdims=True) + 1e-10)
    cross_sims = delim_norms @ content_norms.T
    cross_sim = float(cross_sims.mean())

    # Embedding norms
    delim_norms_mag = np.linalg.norm(delim_embeds, axis=1)
    content_norms_mag = np.linalg.norm(embeddings[content_sample], axis=1)

    print(f"    Delimiter tokens: {len(delimiter_ids)}")
    print(f"    Delimiter internal cosine sim: {delim_internal_sim:.4f}")
    print(f"    Content internal cosine sim:   {content_internal_sim:.4f}")
    print(f"    Cross-group cosine sim:        {cross_sim:.4f}")
    print(f"    Delimiter avg embedding norm:  {delim_norms_mag.mean():.4f}")
    print(f"    Content avg embedding norm:    {content_norms_mag.mean():.4f}")

    # Separation metric: (internal_sim - cross_sim), higher = more separated
    separation = delim_internal_sim - cross_sim
    print(f"    Separation metric (internal - cross): {separation:.4f}")

    return {
        "n_delimiter_tokens": len(delimiter_ids),
        "delim_internal_sim": float(delim_internal_sim),
        "content_internal_sim": float(content_internal_sim),
        "cross_sim": float(cross_sim),
        "delim_avg_norm": float(delim_norms_mag.mean()),
        "content_avg_norm": float(content_norms_mag.mean()),
        "separation": float(separation),
    }


# =========================================================================
# 5. Cross-format transfer (TOON)
# =========================================================================
def eval_cross_format(model, tok, device, name):
    """Test on TOON (tab-separated), never seen in training."""
    from eval_model import compute_perplexity

    # Same data in 3 formats
    toon_text = "orderId\tcustomer\tstatus\ttotal\nORD-00001\tAlice Chen\tpending\t29.97\nORD-00002\tBob Smith\tprocessing\t42.47\nORD-00003\tCarla Rodriguez\tshipped\t54.97\nORD-00004\tDavid Park\tdelivered\t67.47\nORD-00005\tEva Johansson\tcancelled\t79.97"

    gcf_text = """## orders [5]{orderId,customer,status,total}
ORD-00001|Alice Chen|pending|29.97
ORD-00002|Bob Smith|processing|42.47
ORD-00003|Carla Rodriguez|shipped|54.97
ORD-00004|David Park|delivered|67.47
ORD-00005|Eva Johansson|cancelled|79.97"""

    json_text = json.dumps([
        {"orderId": "ORD-00001", "customer": "Alice Chen", "status": "pending", "total": 29.97},
        {"orderId": "ORD-00002", "customer": "Bob Smith", "status": "processing", "total": 42.47},
        {"orderId": "ORD-00003", "customer": "Carla Rodriguez", "status": "shipped", "total": 54.97},
        {"orderId": "ORD-00004", "customer": "David Park", "status": "delivered", "total": 67.47},
        {"orderId": "ORD-00005", "customer": "Eva Johansson", "status": "cancelled", "total": 79.97},
    ], indent=2)

    toon_ppl = compute_perplexity(model, tok, toon_text, device)
    gcf_ppl = compute_perplexity(model, tok, gcf_text, device)
    json_ppl = compute_perplexity(model, tok, json_text, device)

    print(f"    GCF PPL:  {gcf_ppl:>10.1f} ({len(tok.encode(gcf_text).ids)} tokens)")
    print(f"    TOON PPL: {toon_ppl:>10.1f} ({len(tok.encode(toon_text).ids)} tokens)")
    print(f"    JSON PPL: {json_ppl:>10.1f} ({len(tok.encode(json_text).ids)} tokens)")

    return {
        "gcf_ppl": gcf_ppl,
        "toon_ppl": toon_ppl,
        "json_ppl": json_ppl,
    }


# =========================================================================
# 6. Confidence calibration
# =========================================================================
def eval_confidence(model, tok, device, name):
    """Measure model confidence (softmax probability) on delimiter vs content predictions."""
    text = """## products [5]{productId,name,category,price,inStock}
PROD-44821|Monitor|electronics|299.99|true
PROD-18374|Keyboard|peripherals|89.50|false
PROD-92651|Mouse|peripherals|45.00|true
PROD-37284|Headset|accessories|129.99|true
PROD-56738|Stand|accessories|34.99|true"""

    ids = tok.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    probs = F.softmax(logits, dim=-1)

    delim_confidences = []
    content_confidences = []
    correct_delim_conf = []
    correct_content_conf = []

    for i in range(len(ids) - 1):
        target_id = ids[i + 1]
        target_prob = probs[i, target_id].item()
        pred_id = logits[i].argmax().item()
        is_delim = is_delimiter_token(tok, target_id)

        if is_delim:
            delim_confidences.append(target_prob)
            if pred_id == target_id:
                correct_delim_conf.append(target_prob)
        else:
            content_confidences.append(target_prob)
            if pred_id == target_id:
                correct_content_conf.append(target_prob)

    avg_delim_conf = sum(delim_confidences) / max(len(delim_confidences), 1)
    avg_content_conf = sum(content_confidences) / max(len(content_confidences), 1)
    avg_correct_delim = sum(correct_delim_conf) / max(len(correct_delim_conf), 1) if correct_delim_conf else 0
    avg_correct_content = sum(correct_content_conf) / max(len(correct_content_conf), 1) if correct_content_conf else 0

    print(f"    Avg delimiter confidence:  {avg_delim_conf:.4f} (n={len(delim_confidences)})")
    print(f"    Avg content confidence:    {avg_content_conf:.4f} (n={len(content_confidences)})")
    print(f"    Correct delimiter conf:    {avg_correct_delim:.4f} (n={len(correct_delim_conf)})")
    print(f"    Correct content conf:      {avg_correct_content:.4f} (n={len(correct_content_conf)})")

    return {
        "avg_delimiter_confidence": avg_delim_conf,
        "avg_content_confidence": avg_content_conf,
        "avg_correct_delimiter_conf": avg_correct_delim,
        "avg_correct_content_conf": avg_correct_content,
        "n_delimiter": len(delim_confidences),
        "n_content": len(content_confidences),
    }


# =========================================================================
# 7. Few-shot generation
# =========================================================================
def eval_few_shot(model, tok, device, name):
    """Few-shot GCF generation: give examples, ask for a new one."""

    def generate(model, tok, prompt, max_tokens=150, temperature=0.5, device="cpu"):
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
        return tok.decode(generated[len(ids):])

    prompt = """Example 1:
## users [3]{userId,email,role}
USR-001|alice@example.com|admin
USR-002|bob@example.com|editor
USR-003|carla@example.com|viewer

Example 2:
## products [3]{productId,name,price}
PROD-100|Widget|29.99
PROD-101|Gadget|49.99
PROD-102|Sensor|19.99

Example 3:
## """

    results = []
    for i in range(5):
        output = generate(model, tok, prompt, max_tokens=150, temperature=0.5, device=device)
        # Check validity
        lines = output.strip().split('\n')
        has_header = any('[' in l and '{' in l for l in lines[:2])
        has_pipes = sum(1 for l in lines if '|' in l)
        valid = has_header and has_pipes >= 2

        results.append({
            "output": output[:200],
            "has_header": has_header,
            "pipe_rows": has_pipes,
            "valid": valid,
        })
        preview = output[:80].replace('\n', '\\n')
        status = "VALID" if valid else "invalid"
        print(f"    [{i}] {status}: {preview}")

    valid_count = sum(1 for r in results if r["valid"])
    print(f"    Valid: {valid_count}/5")

    return {"valid": valid_count, "total": 5, "samples": results}


def main():
    parser = argparse.ArgumentParser(description="Mechanistic model analysis")
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
    print("MECHANISTIC ANALYSIS")
    print(f"Device: {device}")
    print("=" * 80)

    model_a, tok_a, step_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_b, tok_b, step_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)

    all_results = {}

    analyses = [
        ("1. PER-TOKEN LOSS HEATMAP", eval_per_token_loss),
        ("2. HEAD SPECIALIZATION", eval_head_specialization),
        ("3. CORRUPTION DETECTION", eval_corruption_detection),
        ("4. EMBEDDING SPACE", eval_embedding_space),
        ("5. CROSS-FORMAT TRANSFER", eval_cross_format),
        ("6. CONFIDENCE CALIBRATION", eval_confidence),
        ("7. FEW-SHOT GENERATION", eval_few_shot),
    ]

    for title, eval_fn in analyses:
        print(f"\n{'='*80}")
        print(title)
        print("=" * 80)

        # Model A
        model_a.to(device)
        print(f"\n  {name_a}:")
        result_a = eval_fn(model_a, tok_a, device, name_a)
        model_a.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Model B
        model_b.to(device)
        print(f"\n  {name_b}:")
        result_b = eval_fn(model_b, tok_b, device, name_b)
        model_b.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        key = title.split(". ")[1].lower().replace(" ", "_")
        all_results[key] = {"model_a": result_a, "model_b": result_b}

    # Summary
    print(f"\n{'='*80}")
    print("MECHANISTIC ANALYSIS SUMMARY")
    print("=" * 80)

    ptl = all_results.get("per-token_loss_heatmap", {})
    if ptl:
        a = ptl["model_a"]
        b = ptl["model_b"]
        print(f"\n  Per-token loss:")
        print(f"    {name_a}: delimiter {a['avg_delimiter_loss']:.4f}, content {a['avg_content_loss']:.4f}")
        print(f"    {name_b}: delimiter {b['avg_delimiter_loss']:.4f}, content {b['avg_content_loss']:.4f}")

    hs = all_results.get("head_specialization", {})
    if hs:
        print(f"\n  Head specialization:")
        print(f"    {name_a}: {hs['model_a']['n_delimiter_heads']} delimiter-majority heads, avg score {hs['model_a']['avg_delimiter_score']:.3f}")
        print(f"    {name_b}: {hs['model_b']['n_delimiter_heads']} delimiter-majority heads, avg score {hs['model_b']['avg_delimiter_score']:.3f}")

    cd = all_results.get("corruption_detection", {})
    if cd:
        a_detected = sum(1 for v in cd["model_a"]["corruptions"].values() if v["spike_ratio"] > 1.5)
        b_detected = sum(1 for v in cd["model_b"]["corruptions"].values() if v["spike_ratio"] > 1.5)
        print(f"\n  Corruption detection:")
        print(f"    {name_a}: {a_detected}/5 corruptions detected (>1.5x spike)")
        print(f"    {name_b}: {b_detected}/5 corruptions detected")

    es = all_results.get("embedding_space", {})
    if es:
        print(f"\n  Embedding space separation:")
        print(f"    {name_a}: {es['model_a'].get('separation', 0):.4f}")
        print(f"    {name_b}: {es['model_b'].get('separation', 0):.4f}")

    cf = all_results.get("cross-format_transfer", {})
    if cf:
        print(f"\n  Cross-format transfer (TOON PPL):")
        print(f"    {name_a}: {cf['model_a']['toon_ppl']:.1f}")
        print(f"    {name_b}: {cf['model_b']['toon_ppl']:.1f}")

    cc = all_results.get("confidence_calibration", {})
    if cc:
        print(f"\n  Delimiter confidence:")
        print(f"    {name_a}: {cc['model_a']['avg_delimiter_confidence']:.4f}")
        print(f"    {name_b}: {cc['model_b']['avg_delimiter_confidence']:.4f}")

    fs = all_results.get("few-shot_generation", {})
    if fs:
        print(f"\n  Few-shot generation:")
        print(f"    {name_a}: {fs['model_a']['valid']}/5 valid")
        print(f"    {name_b}: {fs['model_b']['valid']}/5 valid")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results written to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
