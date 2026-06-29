#!/usr/bin/env python3
"""Generate 6 ablation experiment charts from run-002 data."""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Colors
BG = "#0a0a0a"
TEXT = "white"
GRID = "#333333"
CYAN = "#18befc"
GREEN = "#00ff88"
RED = "#ff4444"
ORANGE = "#ff9944"
GRAY = "#888888"

FIGSIZE = (10, 6)
DPI = 150
OUTDIR = "/Users/dayna.blackwell/code/structok/charts"

# Load data
with open("/Users/dayna.blackwell/code/structok/runs/run-002-ablation-full-results.json") as f:
    full = json.load(f)
with open("/Users/dayna.blackwell/code/structok/runs/run-002-ablation-v3-results.json") as f:
    v3 = json.load(f)
with open("/Users/dayna.blackwell/code/structok/runs/run-002-ablation-v4-results.json") as f:
    v4 = json.load(f)


def setup_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, color=GRID)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    path = f"{OUTDIR}/{name}"
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Chart 1: Progressive Ablation Curve ──
fig, ax = plt.subplots(figsize=FIGSIZE)
setup_ax(ax, "Progressive Delimiter Head Ablation", "Delimiter heads removed", "PPL (% change from baseline)")

curve = full["ablation_curve"]
x = [p["heads_ablated"] for p in curve]
gcf = [p["gcf_generic_delta"] for p in curve]
yaml_d = [p["yaml_delta"] for p in curve]
json_d = [p["json_delta"] for p in curve]
nl_d = [p["nl_delta"] for p in curve]

ax.plot(x, gcf, color=CYAN, marker="o", markersize=5, linewidth=2, label="GCF generic")
ax.plot(x, yaml_d, color=GREEN, marker="s", markersize=5, linewidth=2, label="YAML")
ax.plot(x, json_d, color=ORANGE, marker="^", markersize=5, linewidth=2, label="JSON")
ax.plot(x, nl_d, color=GRAY, marker="D", markersize=4, linewidth=2, label="Natural language")
ax.axhline(y=0, color=TEXT, linewidth=0.8, alpha=0.5, linestyle="--")
ax.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TEXT, fontsize=10)
save(fig, "ablation-progressive-curve.png")


# ── Chart 2: Delimiter vs Random Control ──
fig, ax = plt.subplots(figsize=FIGSIZE)
setup_ax(ax, "Delimiter Head Ablation vs Random Control", "Format", "PPL (% change from baseline)")

# Delimiter: final step (40 heads)
delim_final = full["ablation_curve"][-1]
delim_vals = [delim_final["gcf_generic_delta"], delim_final["yaml_delta"],
              delim_final["json_delta"], delim_final["nl_delta"]]

# Random control: average of seed final steps (40 heads)
ctrl_gcf, ctrl_yaml, ctrl_json, ctrl_nl = [], [], [], []
for seed_data in full["control_results"]:
    final = seed_data[-1]  # last entry is 40 heads
    ctrl_gcf.append(final["gcf_generic_delta"])
    ctrl_yaml.append(final["yaml_delta"])
    ctrl_json.append(final["json_delta"])
    ctrl_nl.append(final["nl_delta"])

ctrl_vals = [np.mean(ctrl_gcf), np.mean(ctrl_yaml), np.mean(ctrl_json), np.mean(ctrl_nl)]
labels = ["GCF generic", "YAML", "JSON", "NL"]

x_pos = np.arange(len(labels))
width = 0.35
bars1 = ax.bar(x_pos - width/2, delim_vals, width, color=CYAN, label="Delimiter heads", zorder=3)
bars2 = ax.bar(x_pos + width/2, ctrl_vals, width, color=GRAY, label="Random heads (mean)", zorder=3)

for bar, val in zip(bars1, delim_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (2 if val >= 0 else -4),
            f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", color=TEXT, fontsize=9)
for bar, val in zip(bars2, ctrl_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (2 if val >= 0 else -4),
            f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", color=TEXT, fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, color=TEXT)
ax.axhline(y=0, color=TEXT, linewidth=0.8, alpha=0.5, linestyle="--")
ax.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TEXT, fontsize=10)
save(fig, "ablation-delimiter-vs-control.png")


# ── Chart 3: Reverse Ablation (Sufficiency Test) ──
fig, ax = plt.subplots(figsize=FIGSIZE)
setup_ax(ax, "Sufficiency Test: 70 Heads vs All 384", "Format", "Perplexity (log scale)")

rev = full["reverse_ablation"]
baseline = full["baseline_a"]
formats = ["gcf_generic", "yaml", "nl", "code"]
format_labels = ["GCF generic", "YAML", "NL", "Code"]

baseline_vals = [baseline[f] for f in formats]
delim_only = [rev["delimiter_only"][f"{f}_ppl"] for f in formats]
random_only = [rev["random_only_control"][f"{f}_ppl"] for f in formats]

x_pos = np.arange(len(formats))
width = 0.25
ax.bar(x_pos - width, baseline_vals, width, color=GRAY, label="All 384 heads", zorder=3)
ax.bar(x_pos, delim_only, width, color=CYAN, label="70 delimiter heads only", zorder=3)
ax.bar(x_pos + width, random_only, width, color=ORANGE, label="70 random heads only", zorder=3)

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(format_labels, color=TEXT)
ax.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TEXT, fontsize=10)
save(fig, "ablation-reverse.png")


# ── Chart 4: Layer-wise Ablation ──
fig, ax = plt.subplots(figsize=FIGSIZE)
setup_ax(ax, "Layer-wise Delimiter Head Ablation", "Layer group", "PPL (% change from baseline)")

lw = full["layer_wise_ablation"]
groups = ["early (0-7)", "middle (8-15)", "late (16-23)"]
gcf_deltas = [lw[g]["gcf_generic_delta"] for g in groups]
json_deltas = [lw[g]["json_delta"] for g in groups]
head_counts = [lw[g]["n_heads"] for g in groups]

group_labels = [f"Early (0-7)\n{head_counts[0]} heads",
                f"Middle (8-15)\n{head_counts[1]} heads",
                f"Late (16-23)\n{head_counts[2]} heads"]

x_pos = np.arange(len(groups))
width = 0.35
bars1 = ax.bar(x_pos - width/2, gcf_deltas, width, color=CYAN, label="GCF generic", zorder=3)
bars2 = ax.bar(x_pos + width/2, json_deltas, width, color=ORANGE, label="JSON", zorder=3)

for bar, val in zip(list(bars1) + list(bars2), gcf_deltas + json_deltas):
    offset = 2 if val >= 0 else -2
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", color=TEXT, fontsize=10, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(group_labels, color=TEXT, fontsize=10)
ax.axhline(y=0, color=TEXT, linewidth=0.8, alpha=0.5, linestyle="--")
ax.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TEXT, fontsize=10)
save(fig, "ablation-layer-wise.png")


# ── Chart 5: Cross-Format Transfer ──
fig, ax = plt.subplots(figsize=(10, 8))
setup_ax(ax, "Cross-Format Transfer: 15 Formats Tested", "PPL change (%)", "")

results = v4["results"]
# Sort by delta (most degradation = highest positive at top)
items = sorted(results.items(), key=lambda x: x[1]["delta_pct"], reverse=True)

format_names = [k for k, _ in items]
deltas = [v["delta_pct"] for _, v in items]
trained = [v["trained"] for _, v in items]

colors = []
for d, t in zip(deltas, trained):
    if t:
        colors.append(CYAN)
    elif d > 0:
        colors.append(GREEN)  # unseen, degraded = transfer confirmed
    else:
        colors.append(RED)  # unseen, improved = no transfer

y_pos = np.arange(len(format_names))
ax.barh(y_pos, deltas, color=colors, zorder=3, height=0.7)
ax.axvline(x=0, color=TEXT, linewidth=0.8, alpha=0.5, linestyle="--")

# Labels
nice_names = {
    "gcf_generic": "GCF generic", "gcf_graph": "GCF graph", "json": "JSON",
    "yaml": "YAML", "python": "Python", "toon": "TOON", "csv": "CSV",
    "toml": "TOML", "ini": "INI", "sql": "SQL", "xml": "XML",
    "md_table": "Markdown table", "s_expression": "S-expression",
    "protobuf_text": "Protobuf text", "nl": "Natural language"
}
display_names = []
for name, vals in items:
    label = nice_names.get(name, name)
    tag = "TRAINED" if vals["trained"] else "UNSEEN"
    display_names.append(f"{label}  [{tag}]")

ax.set_yticks(y_pos)
ax.set_yticklabels(display_names, color=TEXT, fontsize=9)

# Value labels on bars
for i, (d, c) in enumerate(zip(deltas, colors)):
    offset = 2 if d >= 0 else -2
    ax.text(d + offset, i, f"{d:+.1f}%", va="center",
            ha="left" if d >= 0 else "right", color=TEXT, fontsize=9)

ax.invert_yaxis()
save(fig, "ablation-cross-format-transfer.png")


# ── Chart 6: Individual Head Importance ──
fig, ax = plt.subplots(figsize=FIGSIZE)
setup_ax(ax, "Individual Head Importance (GCF Generic)", "Head rank", "GCF generic PPL change (%)")

heads = v3["experiment_2_head_importance"]
# Already sorted by gcf_delta_pct descending in the data
gcf_deltas = [h["gcf_delta_pct"] for h in heads]
bar_colors = [CYAN if d >= 0 else RED for d in gcf_deltas]

x_pos = np.arange(len(gcf_deltas))
ax.bar(x_pos, gcf_deltas, color=bar_colors, zorder=3, width=0.8)
ax.axhline(y=0, color=TEXT, linewidth=0.8, alpha=0.5, linestyle="--")

n_positive = sum(1 for d in gcf_deltas if d > 0)
n_negative = sum(1 for d in gcf_deltas if d < 0)
n_zero = sum(1 for d in gcf_deltas if d == 0)

annotation = f"{n_positive} heads hurt GCF when removed, {n_negative} heads help"
if n_zero:
    annotation += f", {n_zero} neutral"
ax.annotate(annotation, xy=(0.5, 0.95), xycoords="axes fraction",
            ha="center", va="top", fontsize=10, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a", edgecolor=GRID))

# Minimal x-axis ticks
ax.set_xticks(range(0, len(gcf_deltas), 5))
save(fig, "ablation-head-importance.png")


print("\nAll 6 charts generated successfully.")
