#!/usr/bin/env python3
"""
Deep evaluation: generation quality, delimiter accuracy, attention patterns,
scaling curves, and adversarial inputs. Compares two models side-by-side.

Usage:
  python eval_deep.py \
    --checkpoint-a checkpoints/structok/checkpoint.pt --tokenizer-a structok-64k.json \
    --checkpoint-b checkpoints/standard/checkpoint.pt --tokenizer-b standard-64k.json \
    --output deep-eval-results.json
"""

import argparse
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F


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
    config = GPTNeoXConfig(**cfg)
    model = GPTNeoXForCausalLM(config)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    step = cp.get("step", 0)
    print(f"Loaded model from step {step} (tokenizer: {Path(tokenizer_path).stem})")

    model.eval()
    return model, tok, step


def generate_text(model, tok, prompt, max_new_tokens=200, temperature=0.8, device="cpu"):
    """Generate text continuation from a prompt."""
    ids = tok.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated = list(ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(generated) >= 2048:
                break
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    return tok.decode(generated[len(ids):])


def eval_generation_quality(model, tok, device, name):
    """Test if the model can generate valid structured data continuations."""
    print(f"\n  {name}:")

    prompts = {
        "gcf_tabular": '## orders [5]{orderId,customer,status,total}\nORD-12345|Alice Chen|shipped|',
        "gcf_graph": '## symbols [3]{id,kind,qname,score,provenance}\n@0|function|auth.',
        "json_object": '{"users": [{"name": "Alice", "role": "admin", "active": ',
        "python_func": 'def calculate_total(items, tax_rate):\n    """Calculate total with tax."""\n    subtotal = sum(',
        "go_func": 'func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {\n\tif r.Method != ',
    }

    results = {}
    for key, prompt in prompts.items():
        # Generate 3 samples
        samples = []
        for i in range(3):
            output = generate_text(model, tok, prompt, max_new_tokens=100,
                                   temperature=0.7, device=device)
            samples.append(output)

        # Analyze quality
        valid_count = 0
        for s in samples:
            if key == "gcf_tabular":
                # Check: contains pipes, has numeric-looking value
                valid_count += 1 if '|' in s and any(c.isdigit() for c in s) else 0
            elif key == "gcf_graph":
                valid_count += 1 if '|' in s else 0
            elif key == "json_object":
                valid_count += 1 if any(c in s for c in ['}', '"', ':']) else 0
            elif key.startswith("python") or key.startswith("go"):
                valid_count += 1 if len(s.strip()) > 5 else 0

        results[key] = {
            "valid": valid_count,
            "total": 3,
            "samples": [s[:150] for s in samples],
        }
        print(f"    {key}: {valid_count}/3 valid")
        for i, s in enumerate(samples):
            preview = s[:80].replace('\n', '\\n')
            print(f"      [{i}] {preview}")

    return results


def eval_delimiter_accuracy(model, tok, device, name):
    """Measure next-token prediction accuracy specifically on delimiter positions."""
    test_texts = {
        "gcf_tabular": """## products [10]{productId,name,category,price,inStock}
PROD-44821|Monitor|electronics|299.99|true
PROD-18374|Keyboard|peripherals|89.50|false
PROD-92651|Mouse|peripherals|45.00|true
PROD-37284|Headset|accessories|129.99|true
PROD-61029|Webcam|electronics|79.99|false
PROD-85412|Dock|electronics|199.00|true
PROD-23847|Cable|accessories|12.99|true
PROD-70193|Adapter|accessories|24.50|false
PROD-49281|Speaker|electronics|149.99|true
PROD-56738|Stand|accessories|34.99|true""",

        "gcf_graph": """## symbols [5]{id,kind,qname,score,provenance}
@0|function|auth.validate|0.95|definition
@1|class|api.Handler|0.88|definition
@2|method|db.connect|0.72|ast_inferred
@3|interface|service.Config|0.91|definition
@4|function|utils.parse|0.65|reference

## edges [4]{target,source,type}
@1<@0|calls
@2<@1|imports
@3<@1|implements
@4<@0|calls""",

        "json": json.dumps([
            {"productId": "PROD-44821", "name": "Monitor", "category": "electronics", "price": 299.99},
            {"productId": "PROD-18374", "name": "Keyboard", "category": "peripherals", "price": 89.50},
            {"productId": "PROD-92651", "name": "Mouse", "category": "peripherals", "price": 45.00},
        ], indent=2),
    }

    print(f"\n  {name}:")
    results = {}

    for key, text in test_texts.items():
        ids = tok.encode(text).ids
        if len(ids) > 2048:
            ids = ids[:2048]

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        predictions = logits[0, :-1].argmax(dim=-1)
        targets = input_ids[0, 1:]

        # Classify each position
        delimiter_correct = 0
        delimiter_total = 0
        content_correct = 0
        content_total = 0

        vocab = tok.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}

        for i in range(len(targets)):
            target_id = targets[i].item()
            pred_id = predictions[i].item()
            target_token = id_to_token.get(target_id, "")

            is_delimiter = any(c in target_token for c in BARRIER_CHARS)

            if is_delimiter:
                delimiter_total += 1
                if pred_id == target_id:
                    delimiter_correct += 1
            else:
                content_total += 1
                if pred_id == target_id:
                    content_correct += 1

        delim_acc = delimiter_correct / max(delimiter_total, 1)
        content_acc = content_correct / max(content_total, 1)

        results[key] = {
            "delimiter_acc": delim_acc,
            "delimiter_n": delimiter_total,
            "content_acc": content_acc,
            "content_n": content_total,
        }
        print(f"    {key}: delimiter acc {delim_acc:.1%} ({delimiter_correct}/{delimiter_total}), content acc {content_acc:.1%} ({content_correct}/{content_total})")

    return results


def eval_attention_patterns(model, tok, device, name):
    """Extract attention weights and analyze how models attend to delimiters."""
    text = """## orders [5]{orderId,customer,status,total}
ORD-12345|Alice Chen|shipped|299.99
ORD-67890|Bob Smith|pending|45.50
ORD-24680|Carla Rodriguez|delivered|1250.00
ORD-13579|David Park|processing|89.99
ORD-11223|Eva Johansson|shipped|567.25"""

    ids = tok.encode(text).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    # Find delimiter positions
    delimiter_positions = []
    content_positions = []
    for i, tid in enumerate(ids):
        token_str = id_to_token.get(tid, "")
        if any(c in token_str for c in BARRIER_CHARS):
            delimiter_positions.append(i)
        else:
            content_positions.append(i)

    print(f"\n  {name} ({len(ids)} tokens, {len(delimiter_positions)} delimiters, {len(content_positions)} content):")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True)

    attentions = outputs.attentions  # tuple of (1, num_heads, seq_len, seq_len) per layer

    results = {"layers": []}

    # Analyze attention to delimiters vs content at each layer
    for layer_idx, attn in enumerate(attentions):
        attn_weights = attn[0]  # (num_heads, seq_len, seq_len)
        avg_attn = attn_weights.mean(dim=0)  # (seq_len, seq_len) averaged over heads

        # How much does each position attend to delimiters vs content?
        if delimiter_positions and content_positions:
            delim_tensor = torch.tensor(delimiter_positions, device=device)
            content_tensor = torch.tensor(content_positions, device=device)

            attn_to_delimiters = avg_attn[:, delim_tensor].sum(dim=-1).mean().item()
            attn_to_content = avg_attn[:, content_tensor].sum(dim=-1).mean().item()

            # Entropy of attention distribution
            entropy = -(avg_attn * (avg_attn + 1e-10).log()).sum(dim=-1).mean().item()

            results["layers"].append({
                "layer": layer_idx,
                "attn_to_delimiters": attn_to_delimiters,
                "attn_to_content": attn_to_content,
                "delimiter_ratio": attn_to_delimiters / max(attn_to_delimiters + attn_to_content, 1e-10),
                "entropy": entropy,
            })

    # Summary: average across layers
    if results["layers"]:
        avg_delim_ratio = sum(l["delimiter_ratio"] for l in results["layers"]) / len(results["layers"])
        avg_entropy = sum(l["entropy"] for l in results["layers"]) / len(results["layers"])
        early_ratio = sum(l["delimiter_ratio"] for l in results["layers"][:6]) / 6
        late_ratio = sum(l["delimiter_ratio"] for l in results["layers"][-6:]) / 6

        results["summary"] = {
            "avg_delimiter_attention_ratio": avg_delim_ratio,
            "avg_entropy": avg_entropy,
            "early_layers_delimiter_ratio": early_ratio,
            "late_layers_delimiter_ratio": late_ratio,
        }
        print(f"    Avg delimiter attention ratio: {avg_delim_ratio:.3f}")
        print(f"    Early layers (0-5): {early_ratio:.3f}")
        print(f"    Late layers (18-23): {late_ratio:.3f}")
        print(f"    Avg attention entropy: {avg_entropy:.3f}")

    return results


def eval_scaling_curve(model, tok, device, name, test_data_dir=None):
    """Test comprehension at fine-grained sizes to map the scaling curve."""
    from eval_model import compute_perplexity

    print(f"\n  {name}:")

    results = []

    # Generate test data at various sizes
    random.seed(88888)
    sizes = [1, 2, 3, 5, 10, 20, 50, 100, 150, 200]

    for n in sizes:
        records = []
        for _ in range(n):
            records.append({
                "productId": f"PROD-{random.randint(10000,99999)}",
                "name": random.choice(["Laptop", "Monitor", "Keyboard", "Mouse", "Headset"]),
                "category": random.choice(["electronics", "peripherals", "accessories"]),
                "price": round(random.uniform(5, 2000), 2),
                "inStock": random.choice([True, False]),
            })

        json_str = json.dumps(records, indent=2)
        gcf_str = f"## products [{n}]{{productId,name,category,price,inStock}}\n" + "\n".join(
            f"{r['productId']}|{r['name']}|{r['category']}|{r['price']}|{str(r['inStock']).lower()}" for r in records
        )

        json_tokens = len(tok.encode(json_str).ids)
        gcf_tokens = len(tok.encode(gcf_str).ids)

        # Only compute if within context window
        gcf_ppl = compute_perplexity(model, tok, gcf_str, device) if gcf_tokens <= 2048 else None
        json_ppl = compute_perplexity(model, tok, json_str, device) if json_tokens <= 2048 else None

        results.append({
            "records": n,
            "gcf_ppl": gcf_ppl,
            "json_ppl": json_ppl,
            "gcf_tokens": gcf_tokens,
            "json_tokens": json_tokens,
        })

        gcf_str_fmt = f"{gcf_ppl:.1f}" if gcf_ppl else "overflow"
        json_str_fmt = f"{json_ppl:.1f}" if json_ppl else "overflow"
        print(f"    {n:>4} records: GCF PPL {gcf_str_fmt:>12} ({gcf_tokens} tok), JSON PPL {json_str_fmt:>12} ({json_tokens} tok)")

    return results


def eval_adversarial(model, tok, device, name):
    """Test on GCF with adversarial/ambiguous content values."""
    from eval_model import compute_perplexity, compute_next_token_accuracy

    print(f"\n  {name}:")

    tests = {
        "normal": """## items [5]{id,name,value}
ITEM-001|Widget|42.50
ITEM-002|Gadget|18.99
ITEM-003|Sensor|125.00
ITEM-004|Module|67.25
ITEM-005|Adapter|9.99""",

        "pipe_in_values": """## items [5]{id,description,value}
ITEM-001|Widget - size A/B|42.50
ITEM-002|Gadget (v2.0)|18.99
ITEM-003|Sensor: temp+humidity|125.00
ITEM-004|Module [rev-3]|67.25
ITEM-005|Adapter <USB-C>|9.99""",

        "json_like_values": """## items [5]{id,config,count}
ITEM-001|{"key": "value"}|42
ITEM-002|[1, 2, 3]|18
ITEM-003|{"nested": {"a": 1}}|125
ITEM-004|true|67
ITEM-005|null|9""",

        "numeric_heavy": """## metrics [5]{id,cpu,mem,disk,latency,errors}
SRV-001|0.847|0.623|0.412|12.5|0
SRV-002|0.234|0.891|0.156|45.2|3
SRV-003|0.956|0.445|0.789|8.1|0
SRV-004|0.112|0.967|0.334|123.7|15
SRV-005|0.678|0.234|0.901|3.2|1""",

        "empty_fields": """## items [5]{id,name,optional1,optional2,value}
ITEM-001|Widget|||42.50
ITEM-002|Gadget|extra||18.99
ITEM-003|||data|125.00
ITEM-004|Module|a|b|67.25
ITEM-005||||9.99""",
    }

    results = {}
    for key, text in tests.items():
        ppl = compute_perplexity(model, tok, text, device)
        acc, total = compute_next_token_accuracy(model, tok, text, device)
        tokens = len(tok.encode(text).ids)

        results[key] = {"ppl": ppl, "acc": acc, "tokens": tokens}
        print(f"    {key:<25} PPL {ppl:>10.1f}  acc {acc:>6.1%}  ({tokens} tokens)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Deep model evaluation")
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
    print(f"Device: {device}")

    name_a = Path(args.tokenizer_a).stem
    name_b = Path(args.tokenizer_b).stem

    model_a, tok_a, step_a = load_model(args.checkpoint_a, args.size, args.tokenizer_a)
    model_b, tok_b, step_b = load_model(args.checkpoint_b, args.size, args.tokenizer_b)

    all_results = {}

    # =========================================================================
    # 1. Generation Quality
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. GENERATION QUALITY")
    print("=" * 80)

    model_a.to(device)
    gen_a = eval_generation_quality(model_a, tok_a, device, name_a)
    model_a.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_b.to(device)
    gen_b = eval_generation_quality(model_b, tok_b, device, name_b)
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results["generation"] = {"model_a": gen_a, "model_b": gen_b}

    # Summary
    print(f"\n  Summary:")
    total_a = sum(v["valid"] for v in gen_a.values())
    total_b = sum(v["valid"] for v in gen_b.values())
    print(f"    {name_a}: {total_a}/{sum(v['total'] for v in gen_a.values())} valid generations")
    print(f"    {name_b}: {total_b}/{sum(v['total'] for v in gen_b.values())} valid generations")

    # =========================================================================
    # 2. Delimiter Prediction Accuracy
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. DELIMITER PREDICTION ACCURACY")
    print("=" * 80)

    model_a.to(device)
    delim_a = eval_delimiter_accuracy(model_a, tok_a, device, name_a)
    model_a.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_b.to(device)
    delim_b = eval_delimiter_accuracy(model_b, tok_b, device, name_b)
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results["delimiter_accuracy"] = {"model_a": delim_a, "model_b": delim_b}

    # Comparison table
    print(f"\n  Comparison:")
    print(f"  {'Test':<20} {name_a+' delim':>18} {name_b+' delim':>18} {name_a+' content':>18} {name_b+' content':>18}")
    print(f"  {'-'*95}")
    for key in delim_a:
        print(f"  {key:<20} {delim_a[key]['delimiter_acc']:>17.1%} {delim_b[key]['delimiter_acc']:>17.1%} {delim_a[key]['content_acc']:>17.1%} {delim_b[key]['content_acc']:>17.1%}")

    # =========================================================================
    # 3. Attention Patterns
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. ATTENTION PATTERN ANALYSIS")
    print("=" * 80)

    model_a.to(device)
    attn_a = eval_attention_patterns(model_a, tok_a, device, name_a)
    model_a.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_b.to(device)
    attn_b = eval_attention_patterns(model_b, tok_b, device, name_b)
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results["attention"] = {"model_a": attn_a, "model_b": attn_b}

    if attn_a.get("summary") and attn_b.get("summary"):
        print(f"\n  Comparison:")
        print(f"  {'Metric':<35} {name_a:>18} {name_b:>18}")
        print(f"  {'-'*75}")
        print(f"  {'Avg delimiter attn ratio':<35} {attn_a['summary']['avg_delimiter_attention_ratio']:>18.3f} {attn_b['summary']['avg_delimiter_attention_ratio']:>18.3f}")
        print(f"  {'Early layers (0-5) delim ratio':<35} {attn_a['summary']['early_layers_delimiter_ratio']:>18.3f} {attn_b['summary']['early_layers_delimiter_ratio']:>18.3f}")
        print(f"  {'Late layers (18-23) delim ratio':<35} {attn_a['summary']['late_layers_delimiter_ratio']:>18.3f} {attn_b['summary']['late_layers_delimiter_ratio']:>18.3f}")
        print(f"  {'Avg attention entropy':<35} {attn_a['summary']['avg_entropy']:>18.3f} {attn_b['summary']['avg_entropy']:>18.3f}")

    # =========================================================================
    # 4. Scaling Curve
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. SCALING CURVE")
    print("=" * 80)

    model_a.to(device)
    scale_a = eval_scaling_curve(model_a, tok_a, device, name_a)
    model_a.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_b.to(device)
    scale_b = eval_scaling_curve(model_b, tok_b, device, name_b)
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results["scaling"] = {"model_a": scale_a, "model_b": scale_b}

    # Comparison table
    print(f"\n  GCF PPL scaling comparison:")
    print(f"  {'Records':>8} {name_a:>15} {name_b:>15} {'Ratio':>10}")
    print(f"  {'-'*52}")
    for sa, sb in zip(scale_a, scale_b):
        if sa["gcf_ppl"] and sb["gcf_ppl"]:
            ratio = sb["gcf_ppl"] / sa["gcf_ppl"]
            print(f"  {sa['records']:>8} {sa['gcf_ppl']:>15.1f} {sb['gcf_ppl']:>15.1f} {ratio:>9.1f}x")
        elif sa["gcf_ppl"]:
            print(f"  {sa['records']:>8} {sa['gcf_ppl']:>15.1f} {'overflow':>15}")
        else:
            print(f"  {sa['records']:>8} {'overflow':>15} {'overflow':>15}")

    # =========================================================================
    # 5. Adversarial Inputs
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. ADVERSARIAL INPUTS")
    print("=" * 80)

    model_a.to(device)
    adv_a = eval_adversarial(model_a, tok_a, device, name_a)
    model_a.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model_b.to(device)
    adv_b = eval_adversarial(model_b, tok_b, device, name_b)
    model_b.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_results["adversarial"] = {"model_a": adv_a, "model_b": adv_b}

    # Comparison
    print(f"\n  Comparison (PPL, lower is better):")
    print(f"  {'Test':<25} {name_a:>15} {name_b:>15} {'Winner':>15}")
    print(f"  {'-'*75}")
    for key in adv_a:
        winner = name_a if adv_a[key]["ppl"] < adv_b[key]["ppl"] else name_b
        print(f"  {key:<25} {adv_a[key]['ppl']:>15.1f} {adv_b[key]['ppl']:>15.1f} {winner:>15}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("DEEP EVAL SUMMARY")
    print("=" * 80)
    print(f"\n  1. Generation: {name_a} {total_a}/15, {name_b} {total_b}/15")

    avg_delim_a = sum(d["delimiter_acc"] for d in delim_a.values()) / len(delim_a)
    avg_delim_b = sum(d["delimiter_acc"] for d in delim_b.values()) / len(delim_b)
    print(f"  2. Delimiter accuracy: {name_a} {avg_delim_a:.1%}, {name_b} {avg_delim_b:.1%}")

    if attn_a.get("summary") and attn_b.get("summary"):
        print(f"  3. Attention: {name_a} delim ratio {attn_a['summary']['avg_delimiter_attention_ratio']:.3f}, {name_b} {attn_b['summary']['avg_delimiter_attention_ratio']:.3f}")

    # Count scaling wins
    scale_wins_a = sum(1 for sa, sb in zip(scale_a, scale_b) if sa["gcf_ppl"] and sb["gcf_ppl"] and sa["gcf_ppl"] < sb["gcf_ppl"])
    scale_total = sum(1 for sa, sb in zip(scale_a, scale_b) if sa["gcf_ppl"] and sb["gcf_ppl"])
    print(f"  4. Scaling: {name_a} wins {scale_wins_a}/{scale_total} sizes")

    adv_wins_a = sum(1 for k in adv_a if adv_a[k]["ppl"] < adv_b[k]["ppl"])
    print(f"  5. Adversarial: {name_a} wins {adv_wins_a}/{len(adv_a)} tests")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results written to {args.output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
