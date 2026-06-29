#!/usr/bin/env python3
"""
Attention heatmap: top 5 delimiter heads, GCF vs JSON.
2x5 grid with 2x2 mini-heatmaps showing attention flow.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Load data ──
with open("/Users/dayna.blackwell/code/structok/runs/run-002-ablation-full-results.json") as f:
    data = json.load(f)

attn = data["attention_patterns"]
gcf_heads = attn["gcf_generic"]["heads"]
json_heads = attn["json"]["heads"]

# Both have 5 heads, same layers/heads
assert len(gcf_heads) == 5 and len(json_heads) == 5

# ── Custom colormap (dark theme) ──
cmap = LinearSegmentedColormap.from_list("structok",
    ["#0a0a0a", "#1a3a5c", "#00e5ff", "#ffffff"], N=256)

# ── Build figure ──
fig, axes = plt.subplots(2, 5, figsize=(12, 8), dpi=150)
fig.patch.set_facecolor("#0a0a0a")

formats_data = [
    ("GCF Generic", gcf_heads),
    ("JSON", json_heads),
]

flow_labels = ["d\u2192d", "d\u2192c", "c\u2192d", "c\u2192c"]
flow_keys = [
    "delim_query_to_delim_key",
    "delim_query_to_content_key",
    "content_query_to_delim_key",
    "content_query_to_content_key",
]

for row_idx, (fmt_name, heads) in enumerate(formats_data):
    for col_idx, head in enumerate(heads):
        ax = axes[row_idx, col_idx]
        ax.set_facecolor("#0a0a0a")

        # Build 2x2 matrix
        matrix = np.array([
            [head["delim_query_to_delim_key"], head["delim_query_to_content_key"]],
            [head["content_query_to_delim_key"], head["content_query_to_content_key"]],
        ])

        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                color = "black" if val > 0.7 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        # Axis labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["delim", "content"], fontsize=7, color="white")
        ax.set_yticklabels(["delim Q", "content Q"], fontsize=7, color="white")
        ax.tick_params(length=0)

        # Title
        layer, head_num = head["layer"], head["head"]
        score = head["overall_delimiter_score"]
        title = f"L{layer}H{head_num} ({score:.2f})"
        ax.set_title(title, fontsize=9, color="white", fontweight="bold", pad=6)

        # Add format label on far left
        if col_idx == 0:
            ax.set_ylabel(fmt_name, fontsize=11, color="#00e5ff",
                          fontweight="bold", labelpad=10)

        # Border
        for spine in ax.spines.values():
            spine.set_color("#333")
            spine.set_linewidth(0.5)

# Suptitle
fig.suptitle("Attention Flow in Top 5 Delimiter Heads: GCF vs JSON",
             color="white", fontsize=11, fontweight="bold", y=0.98)

# Subtitle
fig.text(0.5, 0.93,
         "Each cell: fraction of attention from query type (row) to key type (column)",
         ha="center", fontsize=9, color="#888")

# Column headers: "Key target"
fig.text(0.5, 0.01, "Key target (columns: delimiter vs content tokens)",
         ha="center", fontsize=9, color="#666")

plt.tight_layout(rect=[0, 0.03, 1, 0.91])
plt.savefig("/Users/dayna.blackwell/code/structok/charts/attention-heatmap-gcf-vs-json.png",
            facecolor="#0a0a0a", dpi=150)
print("Saved to charts/attention-heatmap-gcf-vs-json.png")

# ── Print summary ──
print("\nAttention flow comparison (top 5 heads):")
print(f"  {'Head':<12} {'GCF d->d':>10} {'JSON d->d':>10} {'GCF c->d':>10} {'JSON c->d':>10}")
print("  " + "-" * 52)
for g, j in zip(gcf_heads, json_heads):
    label = f"L{g['layer']}H{g['head']}"
    print(f"  {label:<12} {g['delim_query_to_delim_key']:>10.3f} {j['delim_query_to_delim_key']:>10.3f} "
          f"{g['content_query_to_delim_key']:>10.3f} {j['content_query_to_delim_key']:>10.3f}")

# Key finding
gcf_avg_dd = np.mean([h["delim_query_to_delim_key"] for h in gcf_heads])
json_avg_dd = np.mean([h["delim_query_to_delim_key"] for h in json_heads])
gcf_avg_cd = np.mean([h["content_query_to_delim_key"] for h in gcf_heads])
json_avg_cd = np.mean([h["content_query_to_delim_key"] for h in json_heads])

print(f"\nAverage d->d: GCF={gcf_avg_dd:.3f}, JSON={json_avg_dd:.3f}")
print(f"Average c->d: GCF={gcf_avg_cd:.3f}, JSON={json_avg_cd:.3f}")
print(f"\nJSON delimiter heads are {json_avg_dd/gcf_avg_dd:.1f}x more focused on d->d than GCF")
print(f"JSON content-to-delimiter: {json_avg_cd/gcf_avg_cd:.1f}x stronger than GCF")
