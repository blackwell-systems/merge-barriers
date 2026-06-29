#!/usr/bin/env python3
"""
Cross-format transfer predictability analysis.
Computes delimiter density for each format, plots density vs ablation delta.
"""

import json
import sys

from tokenizers import Tokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Load ablation results ──
with open("/Users/dayna.blackwell/code/structok/runs/run-002-ablation-v4-results.json") as f:
    ablation = json.load(f)

results = ablation["results"]

# ── Load tokenizer ──
tok = Tokenizer.from_file("/Users/dayna.blackwell/code/structok/structok-64k.json")

# ── Barrier characters (same as eval_ablation_v4.py) ──
BARRIER_CHARS = set('|@<>"\',:;\t\n{}[]()')

# ── Generators (copied from eval_ablation_v4.py to avoid torch import) ──

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

GENERATORS = {
    "gcf_generic": gen_gcf_generic,
    "gcf_graph": gen_gcf_graph,
    "json": gen_json,
    "yaml": gen_yaml,
    "python": gen_python,
    "toon": gen_toon,
    "csv": gen_csv,
    "toml": gen_toml,
    "ini": gen_ini,
    "sql": gen_sql,
    "xml": gen_xml,
    "md_table": gen_markdown_table,
    "s_expression": gen_sexp,
    "protobuf_text": gen_protobuf_text,
    "nl": lambda: NL_TEXT,
}

# ── Compute delimiter density ──
def delimiter_density(text):
    """Fraction of tokens that contain at least one barrier character."""
    encoded = tok.encode(text)
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    total = len(encoded.ids)
    if total == 0:
        return 0.0

    delim_count = 0
    for tid in encoded.ids:
        token_str = id_to_token.get(tid, "")
        if any(c in token_str for c in BARRIER_CHARS):
            delim_count += 1

    return (delim_count / total) * 100.0

densities = {}
for fmt, gen_fn in GENERATORS.items():
    text = gen_fn()
    densities[fmt] = delimiter_density(text)
    print(f"  {fmt:<16} density={densities[fmt]:5.1f}%  delta={results[fmt]['delta_pct']:+6.1f}%  trained={results[fmt]['trained']}")

# ── Prepare plot data ──
formats = list(results.keys())
x = [densities[f] for f in formats]
y = [results[f]["delta_pct"] for f in formats]

colors = []
for f in formats:
    r = results[f]
    if r["trained"]:
        colors.append("#00e5ff")  # cyan for trained
    elif r["delta_pct"] > 5:
        colors.append("#4caf50")  # green = transfer confirmed (ablation hurts)
    elif r["delta_pct"] < -5:
        colors.append("#f44336")  # red = no transfer (ablation helps)
    else:
        colors.append("#888888")  # gray = neutral

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

ax.scatter(x, y, c=colors, s=100, zorder=5, edgecolors="white", linewidths=0.5)

# Label each point
for i, fmt in enumerate(formats):
    offset_x = 0.8
    offset_y = 2.5
    # Adjust overlapping labels
    if fmt == "gcf_graph":
        offset_y = -4
    elif fmt == "yaml":
        offset_y = -4
    elif fmt == "csv":
        offset_x = -3
        offset_y = -4

    ax.annotate(fmt, (x[i], y[i]),
                textcoords="offset points", xytext=(offset_x * 5, offset_y),
                fontsize=7, color="white", alpha=0.9)

# Reference lines
ax.axhline(y=5, color="#4caf50", linestyle="--", alpha=0.3, linewidth=0.8)
ax.axhline(y=-5, color="#f44336", linestyle="--", alpha=0.3, linewidth=0.8)
ax.axhline(y=0, color="white", linestyle="-", alpha=0.15, linewidth=0.5)

# Styling
ax.set_xlabel("Delimiter Density (%)", color="white", fontsize=11)
ax.set_ylabel("Ablation Delta (%)", color="white", fontsize=11)
ax.set_title("Cross-Format Transfer Predictability:\nDelimiter Density vs Ablation Impact",
             color="white", fontsize=11, fontweight="bold")

ax.tick_params(colors="white", labelsize=9)
ax.spines["bottom"].set_color("#333")
ax.spines["left"].set_color("#333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.1, color="white")

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='#0a0a0a', markerfacecolor='#00e5ff',
           markersize=8, label='Trained format'),
    Line2D([0], [0], marker='o', color='#0a0a0a', markerfacecolor='#4caf50',
           markersize=8, label='Unseen: transfer confirmed (delta > +5%)'),
    Line2D([0], [0], marker='o', color='#0a0a0a', markerfacecolor='#f44336',
           markersize=8, label='Unseen: no transfer (delta < -5%)'),
    Line2D([0], [0], marker='o', color='#0a0a0a', markerfacecolor='#888888',
           markersize=8, label='Unseen: neutral'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
          facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")

# Correlation annotation
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax.annotate(f"r = {r_value:.3f}, p = {p_value:.3f}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9, color="#888",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor="#333"))

plt.tight_layout()
plt.savefig("/Users/dayna.blackwell/code/structok/charts/cross-format-density-vs-delta.png",
            facecolor="#0a0a0a", dpi=150)
print(f"\nSaved to charts/cross-format-density-vs-delta.png")

# ── Analysis output ──
print("\n" + "=" * 70)
print("ANALYSIS: Delimiter Density vs Transfer")
print("=" * 70)
print(f"\nLinear regression: r={r_value:.3f}, p={p_value:.3f}, slope={slope:.2f}")

# Check for threshold
unseen = [(densities[f], results[f]["delta_pct"], f) for f in formats if not results[f]["trained"]]
unseen.sort(key=lambda t: t[0])

print(f"\nUnseen formats sorted by density:")
print(f"  {'Format':<16} {'Density':>8} {'Delta':>8} {'Transfer?':>10}")
for d, delta, f in unseen:
    xfer = "YES" if delta > 5 else ("no" if delta < -5 else "neutral")
    print(f"  {f:<16} {d:>7.1f}% {delta:>+7.1f}% {xfer:>10}")

# Find threshold
transfer_densities = [d for d, delta, f in unseen if delta > 5]
no_transfer_densities = [d for d, delta, f in unseen if delta < -5]

if transfer_densities and no_transfer_densities:
    min_transfer = min(transfer_densities)
    max_no_transfer = max(no_transfer_densities)
    if min_transfer > max_no_transfer:
        threshold = (min_transfer + max_no_transfer) / 2
        print(f"\nThreshold found: ~{threshold:.1f}% density separates transfer from no-transfer")
        print(f"  Formats above {threshold:.1f}% density: transfer confirmed")
        print(f"  Formats below {threshold:.1f}% density: no transfer")
    else:
        print(f"\nNo clean threshold: transfer and no-transfer density ranges overlap")
        print(f"  Transfer range: {min(transfer_densities):.1f}% - {max(transfer_densities):.1f}%")
        print(f"  No-transfer range: {min(no_transfer_densities):.1f}% - {max(no_transfer_densities):.1f}%")
elif no_transfer_densities:
    print(f"\nOnly XML shows no-transfer (delta < -5%). Density: {no_transfer_densities[0]:.1f}%")
else:
    print(f"\nAll unseen formats show transfer or neutral. No threshold needed.")
