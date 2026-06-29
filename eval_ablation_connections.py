#!/usr/bin/env python3
"""
Connect ablation findings to the original paper's key numbers.

Experiment 17: Per-token loss under ablation
  - Does ablating delimiter heads spike delimiter prediction loss back to Model B levels?

Experiment 18: Attention entropy under ablation
  - Does ablating delimiter heads increase attention entropy (more diffuse)?

Usage:
  python eval_ablation_connections.py \
    --checkpoint-a checkpoint-a.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoint-b.pt --tokenizer-b standard-64k.json \
    --output ablation-connections-results.json
"""

import argparse
import copy
import datetime
import gc
import json
import math
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

GCF_TEXT = """## orders [10]{orderId,customer,status,total}
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


def load_model(checkpoint_path, size, tokenizer_path):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from transformers import LlamaConfig, LlamaForCausalLM
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    cfg = MODEL_CONFIGS[size].copy()
    cfg["vocab_size"] = vocab_size
    cfg["_attn_implementation"] = "eager"

    if "llama" in size:
        cfg.setdefault("num_key_value_heads", cfg["num_attention_heads"] // 4)
        cfg.setdefault("rope_theta", 500000.0)
        config = LlamaConfig(**cfg)
        model = LlamaForCausalLM(config)
    else:
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


def identify_delimiter_heads(model, tok, device, excess_threshold=0.15):
    ids = tok.encode(GCF_TEXT).ids[:512]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))
    base_rate = len(delim_positions) / max(len(ids), 1)

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
            excess = raw - base_rate
            if excess > excess_threshold:
                heads.append((layer_idx, head_idx, excess))

    del outputs
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    heads.sort(key=lambda x: x[2], reverse=True)
    print(f"  Excess threshold: {excess_threshold}, heads: {len(heads)} / {model.config.num_hidden_layers * model.config.num_attention_heads}")
    return heads


def _get_output_proj(model, layer_idx):
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer_idx].attention.dense
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx].self_attn.o_proj
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")


def ablate_heads(model, heads):
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    for layer_idx, head_idx in heads:
        proj = _get_output_proj(model, layer_idx)
        start = head_idx * head_dim
        end = start + head_dim
        proj.weight.data[:, start:end] = 0.0


# =========================================================================
# Experiment 17: Per-token loss under ablation
# =========================================================================

def per_token_loss(model, tok, device):
    """Compute average loss for delimiter vs content tokens."""
    ids = tok.encode(GCF_TEXT).ids[:512]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    losses = F.cross_entropy(shift_logits, shift_labels, reduction='none')

    delim_losses = []
    content_losses = []

    for i in range(len(losses)):
        target_id = ids[i + 1]
        loss_val = losses[i].item()
        if is_delimiter_token(tok, target_id):
            delim_losses.append(loss_val)
        else:
            content_losses.append(loss_val)

    avg_delim = sum(delim_losses) / max(len(delim_losses), 1)
    avg_content = sum(content_losses) / max(len(content_losses), 1)

    return {
        "avg_delimiter_loss": round(avg_delim, 4),
        "avg_content_loss": round(avg_content, 4),
        "delimiter_content_ratio": round(avg_delim / max(avg_content, 1e-10), 4),
        "n_delimiter": len(delim_losses),
        "n_content": len(content_losses),
    }


# =========================================================================
# Experiment 18: Attention entropy under ablation
# =========================================================================

def attention_entropy(model, tok, device):
    """Compute average attention entropy across all heads and positions."""
    ids = tok.encode(GCF_TEXT).ids[:512]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    total_entropy = 0.0
    n_heads_total = 0

    # Also compute grammar attention share
    delim_positions = set(i for i, tid in enumerate(ids) if is_delimiter_token(tok, tid))
    total_grammar_attn = 0.0
    total_attn = 0.0

    for layer_idx, attn in enumerate(outputs.attentions):
        for head_idx in range(attn.shape[1]):
            w = attn[0, head_idx].float().cpu()
            # Entropy per query position, averaged
            # H = -sum(p * log(p))
            eps = 1e-10
            h = -(w * (w + eps).log()).sum(dim=-1).mean().item()
            total_entropy += h
            n_heads_total += 1

            # Grammar attention share
            seq_len = w.shape[0]
            for d in delim_positions:
                total_grammar_attn += w[:, d].mean().item()
            for p in range(seq_len):
                total_attn += w[:, p].mean().item()

    avg_entropy = total_entropy / max(n_heads_total, 1)
    grammar_share = total_grammar_attn / max(total_attn, 1e-10)

    del outputs
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "avg_entropy": round(avg_entropy, 4),
        "grammar_attention_share": round(grammar_share, 4),
        "n_heads": n_heads_total,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Connect ablation to original paper numbers")
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
        "experiment": "ablation_connections",
        "description": "Connect ablation findings to per-token loss (2.4x) and attention entropy (30% to 8.6%)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": device,
    }
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    print("=" * 90)
    print("CONNECTING ABLATION TO ORIGINAL PAPER NUMBERS")
    print("=" * 90)
    print(f"\nDevice: {device}")

    # Load both models
    print("\nLoading Model A (merge barriers)...", flush=True)
    model_a, tok_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_a.to(device)

    print("Loading Model B (standard BPE)...", flush=True)
    model_b, tok_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)
    model_b.to(device)

    # Identify delimiter heads in Model A
    print("\nIdentifying delimiter heads in Model A...", flush=True)
    delimiter_heads = identify_delimiter_heads(model_a, tok_a, device)
    print(f"Found {len(delimiter_heads)} delimiter heads")

    # =====================================================================
    # EXPERIMENT 17: Per-token loss
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 17: PER-TOKEN LOSS UNDER ABLATION")
    print("=" * 90)
    print("\nQuestion: does ablating delimiter heads spike delimiter prediction")
    print("loss back to Model B levels?")
    print(f"\nOriginal finding: Model A delimiter loss 6.1, Model B 14.8 (2.4x)")

    # Model A baseline
    ptl_a = per_token_loss(model_a, tok_a, device)
    print(f"\nModel A (baseline):")
    print(f"  Delimiter loss: {ptl_a['avg_delimiter_loss']:.4f}")
    print(f"  Content loss:   {ptl_a['avg_content_loss']:.4f}")
    print(f"  Ratio:          {ptl_a['delimiter_content_ratio']:.4f}x")

    # Model B baseline
    ptl_b = per_token_loss(model_b, tok_b, device)
    print(f"\nModel B (baseline):")
    print(f"  Delimiter loss: {ptl_b['avg_delimiter_loss']:.4f}")
    print(f"  Content loss:   {ptl_b['avg_content_loss']:.4f}")
    print(f"  Ratio:          {ptl_b['delimiter_content_ratio']:.4f}x")

    # Model A with delimiter heads ablated
    model_a_ablated = copy.deepcopy(model_a)
    model_a_ablated.to(device)
    ablate_heads(model_a_ablated, [(l, h) for l, h, _ in delimiter_heads])

    ptl_a_ablated = per_token_loss(model_a_ablated, tok_a, device)
    print(f"\nModel A (delimiter heads ablated):")
    print(f"  Delimiter loss: {ptl_a_ablated['avg_delimiter_loss']:.4f}")
    print(f"  Content loss:   {ptl_a_ablated['avg_content_loss']:.4f}")
    print(f"  Ratio:          {ptl_a_ablated['delimiter_content_ratio']:.4f}x")

    delim_change = ((ptl_a_ablated['avg_delimiter_loss'] - ptl_a['avg_delimiter_loss']) / ptl_a['avg_delimiter_loss']) * 100
    content_change = ((ptl_a_ablated['avg_content_loss'] - ptl_a['avg_content_loss']) / ptl_a['avg_content_loss']) * 100
    print(f"\n  Delimiter loss change: {delim_change:+.1f}%")
    print(f"  Content loss change:   {content_change:+.1f}%")

    if ptl_a_ablated['avg_delimiter_loss'] > ptl_b['avg_delimiter_loss'] * 0.7:
        print(f"\n  CONFIRMED: ablation brings delimiter loss near Model B levels")
        print(f"  ({ptl_a_ablated['avg_delimiter_loss']:.1f} vs Model B's {ptl_b['avg_delimiter_loss']:.1f})")
    else:
        print(f"\n  Delimiter loss increased but didn't reach Model B levels")
        print(f"  ({ptl_a_ablated['avg_delimiter_loss']:.1f} vs Model B's {ptl_b['avg_delimiter_loss']:.1f})")

    # =====================================================================
    # EXPERIMENT 18: Attention entropy
    # =====================================================================
    print("\n" + "=" * 90)
    print("EXPERIMENT 18: ATTENTION ENTROPY UNDER ABLATION")
    print("=" * 90)
    print("\nQuestion: does ablating delimiter heads increase attention entropy")
    print("(more diffuse, like the grammar attention collapse)?")

    # Model A baseline
    ent_a = attention_entropy(model_a, tok_a, device)
    print(f"\nModel A (baseline):")
    print(f"  Avg entropy:          {ent_a['avg_entropy']:.4f}")
    print(f"  Grammar attn share:   {ent_a['grammar_attention_share']:.1%}")

    # Model B baseline
    model_b.to(device)
    ent_b = attention_entropy(model_b, tok_b, device)
    print(f"\nModel B (baseline):")
    print(f"  Avg entropy:          {ent_b['avg_entropy']:.4f}")
    print(f"  Grammar attn share:   {ent_b['grammar_attention_share']:.1%}")

    # Model A ablated
    ent_a_ablated = attention_entropy(model_a_ablated, tok_a, device)
    print(f"\nModel A (delimiter heads ablated):")
    print(f"  Avg entropy:          {ent_a_ablated['avg_entropy']:.4f}")
    print(f"  Grammar attn share:   {ent_a_ablated['grammar_attention_share']:.1%}")

    entropy_change = ((ent_a_ablated['avg_entropy'] - ent_a['avg_entropy']) / ent_a['avg_entropy']) * 100
    grammar_change = ((ent_a_ablated['grammar_attention_share'] - ent_a['grammar_attention_share']) / ent_a['grammar_attention_share']) * 100
    print(f"\n  Entropy change:        {entropy_change:+.1f}%")
    print(f"  Grammar share change:  {grammar_change:+.1f}%")

    if ent_a_ablated['avg_entropy'] > ent_a['avg_entropy']:
        print(f"\n  CONFIRMED: ablation increases entropy (more diffuse attention)")
    if abs(ent_a_ablated['grammar_attention_share'] - ent_b['grammar_attention_share']) < abs(ent_a['grammar_attention_share'] - ent_b['grammar_attention_share']):
        print(f"  CONFIRMED: ablated grammar share ({ent_a_ablated['grammar_attention_share']:.1%}) moves toward Model B ({ent_b['grammar_attention_share']:.1%})")

    del model_a_ablated
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\nPer-token loss:")
    print(f"  Model A baseline:  delim={ptl_a['avg_delimiter_loss']:.1f}, content={ptl_a['avg_content_loss']:.1f}")
    print(f"  Model A ablated:   delim={ptl_a_ablated['avg_delimiter_loss']:.1f}, content={ptl_a_ablated['avg_content_loss']:.1f}")
    print(f"  Model B:           delim={ptl_b['avg_delimiter_loss']:.1f}, content={ptl_b['avg_content_loss']:.1f}")
    print(f"  Ablation effect:   delimiter loss {delim_change:+.1f}%, content loss {content_change:+.1f}%")
    print(f"\nAttention entropy:")
    print(f"  Model A baseline:  entropy={ent_a['avg_entropy']:.4f}, grammar={ent_a['grammar_attention_share']:.1%}")
    print(f"  Model A ablated:   entropy={ent_a_ablated['avg_entropy']:.4f}, grammar={ent_a_ablated['grammar_attention_share']:.1%}")
    print(f"  Model B:           entropy={ent_b['avg_entropy']:.4f}, grammar={ent_b['grammar_attention_share']:.1%}")
    print(f"  Ablation effect:   entropy {entropy_change:+.1f}%, grammar share {grammar_change:+.1f}%")

    # Save
    if args.output:
        out = {
            "metadata": metadata,
            "delimiter_heads": len(delimiter_heads),
            "per_token_loss": {
                "model_a_baseline": ptl_a,
                "model_b_baseline": ptl_b,
                "model_a_ablated": ptl_a_ablated,
                "delimiter_loss_change_pct": round(delim_change, 1),
                "content_loss_change_pct": round(content_change, 1),
            },
            "attention_entropy": {
                "model_a_baseline": ent_a,
                "model_b_baseline": ent_b,
                "model_a_ablated": ent_a_ablated,
                "entropy_change_pct": round(entropy_change, 1),
                "grammar_share_change_pct": round(grammar_change, 1),
            },
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
            (args.output, "logs/run-002-ablation/ablation-connections-results.json"),
        ]:
            if local and os.path.exists(local):
                s3.upload_file(local, "structok-training", key)
                print(f"  Uploaded {key}", flush=True)
        log_path = args.output.replace("-results.json", "-log.txt") if args.output else None
        if log_path and os.path.exists(log_path):
            s3.upload_file(log_path, "structok-training", "logs/run-002-ablation/ablation-connections-log.txt")
            print(f"  Uploaded ablation-connections-log.txt", flush=True)
    except Exception as e:
        print(f"R2 upload failed: {e}", flush=True)


if __name__ == "__main__":
    main()
