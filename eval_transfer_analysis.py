"""
Transfer hypothesis analysis for merge-barrier tokenizer.
Tests three hypotheses about what predicts cross-format transfer.
"""

import json
import math
from collections import Counter
from tokenizers import Tokenizer
from scipy import stats
import numpy as np

# --- Load tokenizer ---
tok = Tokenizer.from_file("/Users/dayna.blackwell/code/structok/structok-64k.json")

BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')

# Build lookup once
_vocab = tok.get_vocab()
_id_to_token = {v: k for k, v in _vocab.items()}

def is_delimiter_token(token_id):
    token_str = _id_to_token.get(token_id, "")
    return any(c in token_str for c in BARRIER_CHARS)


# --- Data generators ---

def gen_csv(n=50):
    lines = ["orderId,customer,status,total"]
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d},{names[i%5]},{statuses[i%5]},{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_ini(n=50):
    lines = []
    for i in range(n):
        lines.append(f"[server_{i}]")
        lines.append(f"host = 10.0.{i//256}.{i%256}")
        lines.append(f"port = {8000+i}")
        lines.append(f"enabled = {'true' if i%3!=0 else 'false'}")
        lines.append("")
    return "\n".join(lines)

def gen_sql(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["INSERT INTO orders (id, customer, status, total) VALUES"]
    vals = []
    for i in range(n):
        vals.append(f"  ('ORD-{i+1:05d}', '{names[i%5]}', '{statuses[i%5]}', {round(29.97+i*12.50,2)})")
    lines.append(",\n".join(vals) + ";")
    return "\n".join(lines)

def gen_markdown_table(n=50):
    lines = ["| orderId | customer | status | total |", "|---------|----------|--------|-------|"]
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    for i in range(n):
        lines.append(f"| ORD-{i+1:05d} | {names[i%5]} | {statuses[i%5]} | {round(29.97+i*12.50,2)} |")
    return "\n".join(lines)

def gen_sexpr(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    exprs = ["(orders"]
    for i in range(n):
        exprs.append(f'  (order (id "ORD-{i+1:05d}") (customer "{names[i%5]}") (status "{statuses[i%5]}") (total {round(29.97+i*12.50,2)}))')
    exprs.append(")")
    return "\n".join(exprs)

def gen_protobuf(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = []
    for i in range(n):
        lines.append("order {")
        lines.append(f'  id: "ORD-{i+1:05d}"')
        lines.append(f'  customer: "{names[i%5]}"')
        lines.append(f'  status: "{statuses[i%5]}"')
        lines.append(f"  total: {round(29.97+i*12.50,2)}")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)

def gen_toml(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = []
    for i in range(n):
        lines.append(f"[[orders]]")
        lines.append(f'id = "ORD-{i+1:05d}"')
        lines.append(f'customer = "{names[i%5]}"')
        lines.append(f'status = "{statuses[i%5]}"')
        lines.append(f"total = {round(29.97+i*12.50,2)}")
        lines.append("")
    return "\n".join(lines)

def gen_toon(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["orderId\tcustomer\tstatus\ttotal"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}\t{names[i%5]}\t{statuses[i%5]}\t{round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_xml(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = ["<orders>"]
    for i in range(n):
        lines.append(f"  <order>")
        lines.append(f"    <id>ORD-{i+1:05d}</id>")
        lines.append(f"    <customer>{names[i%5]}</customer>")
        lines.append(f"    <status>{statuses[i%5]}</status>")
        lines.append(f"    <total>{round(29.97+i*12.50,2)}</total>")
        lines.append(f"  </order>")
    lines.append("</orders>")
    return "\n".join(lines)

def gen_yaml(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    lines = ["orders:"]
    for i in range(n):
        lines.append(f"  - id: ORD-{i+1:05d}")
        lines.append(f"    customer: {names[i%5]}")
        lines.append(f"    status: pending")
        lines.append(f"    total: {round(29.97+i*12.50,2)}")
    return "\n".join(lines)

def gen_json_data(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    records = [{"id":f"ORD-{i+1:05d}","customer":names[i%5],"status":"pending","total":round(29.97+i*12.50,2)} for i in range(n)]
    return json.dumps({"orders":records}, indent=2)

def gen_gcf(n=50):
    names = ["Alice","Bob","Carla","David","Eva"]
    statuses = ["pending","processing","shipped","delivered","cancelled"]
    lines = [f"## orders [{n}]{{orderId,customer,status,total}}"]
    for i in range(n):
        lines.append(f"ORD-{i+1:05d}|{names[i%5]}|{statuses[i%5]}|{round(29.97+i*12.50,2)}")
    return "\n".join(lines)


# --- Ablation data ---

ablation_data = {
    "CSV":            +30.0,
    "INI":            +36.4,
    "SQL":            +57.2,
    "Markdown table": +30.4,
    "S-expression":   +38.5,
    "Protobuf text":  +102.4,
    "TOML":           +3.5,
    "TOON":           -15.8,
    "XML":            -31.5,
    # Trained formats (for reference, included in correlation)
    "YAML":           +32.7,
    "JSON":           -61.8,
    "GCF generic":    -17.0,
}

generators = {
    "CSV":            gen_csv,
    "INI":            gen_ini,
    "SQL":            gen_sql,
    "Markdown table": gen_markdown_table,
    "S-expression":   gen_sexpr,
    "Protobuf text":  gen_protobuf,
    "TOML":           gen_toml,
    "TOON":           gen_toon,
    "XML":            gen_xml,
    "YAML":           gen_yaml,
    "JSON":           gen_json_data,
    "GCF generic":    gen_gcf,
}


# --- Metric computation ---

def tokenize_text(text):
    """Return list of token IDs."""
    encoding = tok.encode(text)
    return encoding.ids


def get_delimiter_positions(token_ids):
    """Return sorted list of positions where delimiter tokens appear."""
    return [i for i, tid in enumerate(token_ids) if is_delimiter_token(tid)]


def inter_delimiter_distances(delimiter_positions):
    """Compute distances between consecutive delimiter tokens."""
    if len(delimiter_positions) < 2:
        return []
    return [delimiter_positions[i+1] - delimiter_positions[i]
            for i in range(len(delimiter_positions) - 1)]


def hypothesis1_boundary_clarity(token_ids):
    """Mean inter-delimiter token span."""
    delim_pos = get_delimiter_positions(token_ids)
    distances = inter_delimiter_distances(delim_pos)
    if not distances:
        return 0.0
    return float(np.mean(distances))


def hypothesis2_positional_distribution(token_ids):
    """Fraction of delimiter tokens at boundary positions (near newlines)."""
    delim_pos = get_delimiter_positions(token_ids)
    if not delim_pos:
        return 0.0

    # Find newline positions
    newline_pos = set()
    for i, tid in enumerate(token_ids):
        token_str = _id_to_token.get(tid, "")
        if "\n" in token_str:
            newline_pos.add(i)

    # A delimiter is at a "boundary position" if it is within 2 tokens of a newline
    boundary_count = 0
    for pos in delim_pos:
        near_newline = any(abs(pos - nl) <= 2 for nl in newline_pos)
        if near_newline:
            boundary_count += 1

    return boundary_count / len(delim_pos)


def hypothesis3_spacing_regularity(token_ids):
    """Shannon entropy of inter-delimiter distance distribution (lower = more regular)."""
    delim_pos = get_delimiter_positions(token_ids)
    distances = inter_delimiter_distances(delim_pos)
    if len(distances) < 2:
        return 0.0

    # Bin distances (use raw integer distances as bins)
    counts = Counter(distances)
    total = len(distances)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# --- Run analysis ---

print("=" * 100)
print("CROSS-FORMAT TRANSFER HYPOTHESIS ANALYSIS")
print("Tokenizer: structok-64k with merge barriers")
print("=" * 100)
print()

# Compute metrics for all formats
results = {}
for fmt_name, gen_func in generators.items():
    text = gen_func(50)
    token_ids = tokenize_text(text)
    delim_pos = get_delimiter_positions(token_ids)

    h1 = hypothesis1_boundary_clarity(token_ids)
    h2 = hypothesis2_positional_distribution(token_ids)
    h3 = hypothesis3_spacing_regularity(token_ids)
    delta = ablation_data[fmt_name]

    results[fmt_name] = {
        "total_tokens": len(token_ids),
        "delimiter_tokens": len(delim_pos),
        "delim_fraction": len(delim_pos) / len(token_ids) if token_ids else 0,
        "h1_mean_span": h1,
        "h2_boundary_frac": h2,
        "h3_entropy": h3,
        "delta": delta,
    }

# --- Print results table ---

print(f"{'Format':<18} {'Tokens':>6} {'Delims':>6} {'Delim%':>7} {'H1:MeanSpan':>12} {'H2:BndryFrac':>13} {'H3:Entropy':>11} {'Delta%':>8} {'Transfer':>10}")
print("-" * 100)

for fmt_name in generators:
    r = results[fmt_name]
    transfer = "YES" if r["delta"] > 20 else ("weak" if r["delta"] > 0 else "NO")
    print(f"{fmt_name:<18} {r['total_tokens']:>6} {r['delimiter_tokens']:>6} {r['delim_fraction']:>7.1%} {r['h1_mean_span']:>12.2f} {r['h2_boundary_frac']:>13.3f} {r['h3_entropy']:>11.3f} {r['delta']:>+8.1f} {transfer:>10}")

print()

# --- Correlation analysis ---

fmt_names = list(generators.keys())
deltas = [results[f]["delta"] for f in fmt_names]
h1_vals = [results[f]["h1_mean_span"] for f in fmt_names]
h2_vals = [results[f]["h2_boundary_frac"] for f in fmt_names]
h3_vals = [results[f]["h3_entropy"] for f in fmt_names]

print("=" * 100)
print("CORRELATION ANALYSIS (Pearson r, all 12 formats)")
print("=" * 100)
print()

for name, vals in [("H1: Mean inter-delimiter span", h1_vals),
                    ("H2: Boundary position fraction", h2_vals),
                    ("H3: Spacing entropy (lower = more regular)", h3_vals)]:
    r, p = stats.pearsonr(vals, deltas)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(not significant)"
    print(f"  {name}")
    print(f"    Pearson r = {r:+.4f},  p = {p:.4f}  {sig}")
    print()

# --- Untrained formats only (the real transfer test) ---
untrained = ["CSV", "INI", "SQL", "Markdown table", "S-expression", "Protobuf text", "TOML", "TOON", "XML"]
deltas_u = [results[f]["delta"] for f in untrained]
h1_u = [results[f]["h1_mean_span"] for f in untrained]
h2_u = [results[f]["h2_boundary_frac"] for f in untrained]
h3_u = [results[f]["h3_entropy"] for f in untrained]

print("=" * 100)
print("CORRELATION ANALYSIS (Pearson r, 9 untrained formats only)")
print("=" * 100)
print()

for name, vals in [("H1: Mean inter-delimiter span", h1_u),
                    ("H2: Boundary position fraction", h2_u),
                    ("H3: Spacing entropy (lower = more regular)", h3_u)]:
    r, p = stats.pearsonr(vals, deltas_u)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(not significant)"
    print(f"  {name}")
    print(f"    Pearson r = {r:+.4f},  p = {p:.4f}  {sig}")
    print()

# --- Summary ---
print("=" * 100)
print("HYPOTHESIS VERDICTS")
print("=" * 100)
print()

# Re-compute for verdict
for label, vals_all, vals_untr in [
    ("H1: Boundary Clarity (mean inter-delimiter span)", h1_vals, h1_u),
    ("H2: Positional Distribution (boundary fraction)", h2_vals, h2_u),
    ("H3: Spacing Regularity (entropy, lower = more regular)", h3_vals, h3_u),
]:
    r_all, p_all = stats.pearsonr(vals_all, deltas)
    r_u, p_u = stats.pearsonr(vals_untr, deltas_u)

    if label.startswith("H3"):
        # For entropy, prediction is NEGATIVE correlation (lower entropy = higher delta)
        supported_all = r_all < -0.3 and p_all < 0.10
        supported_u = r_u < -0.3 and p_u < 0.10
        direction = "negative"
    else:
        supported_all = r_all > 0.3 and p_all < 0.10
        supported_u = r_u > 0.3 and p_u < 0.10
        direction = "positive"

    verdict = "SUPPORTED" if supported_u else "NOT SUPPORTED"
    print(f"  {label}")
    print(f"    All formats:      r = {r_all:+.4f}, p = {p_all:.4f}")
    print(f"    Untrained only:   r = {r_u:+.4f}, p = {p_u:.4f}")
    print(f"    Expected direction: {direction}")
    print(f"    Verdict: {verdict}")
    print()
