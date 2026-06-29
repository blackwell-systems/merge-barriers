#!/usr/bin/env python3
"""Generate charts from run-003 Llama architecture independence data."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Colors (match structok dark theme)
BG = "#0a0a0a"
TEXT = "white"
GRID = "#333333"
CYAN = "#18befc"
GREEN = "#00ff88"
RED = "#ff4444"
ORANGE = "#ff9944"
GRAY = "#888888"
PURPLE = "#a78bfa"
LEGEND_BG = "#1a1a1a"

DPI = 150
OUTDIR = "/Users/dayna.blackwell/code/structok/charts"


def setup_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    if ylabel:
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
    print(f"  {name}")


# ── Chart 1: Cross-Format Transfer NeoX vs Llama ──

def chart_transfer_comparison():
    formats = ["CSV", "INI", "SQL", "Md table", "S-expr", "Protobuf", "TOML", "TOON", "XML"]

    # NeoX (50 heads, excess 0.10)
    neox = [38.0, 41.0, 72.1, 20.9, 25.3, 120.4, 74.8, -2.7, 12.2]

    # Llama (56 heads, excess 0.15)
    llama = [27.0, 1.5, 13.5, 38.0, 23.8, 17.9, 21.3, 15.3, 34.0]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(formats))
    width = 0.35

    ax.bar(x - width/2, neox, width, label="GPT-NeoX 410M", color=CYAN, edgecolor=BG, linewidth=0.5)
    ax.bar(x + width/2, llama, width, label="Llama 410M", color=ORANGE, edgecolor=BG, linewidth=0.5)

    setup_ax(ax, "Cross-Format Transfer: Architecture Independence\nAblation delta on unseen formats (positive = heads helped)",
             ylabel="PPL change when delimiter heads removed (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, color=TEXT, fontsize=10, rotation=15, ha="right")
    ax.axhline(y=0, color=TEXT, linewidth=0.5, alpha=0.3, linestyle="--")
    ax.legend(facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=11, loc="upper right")

    # Annotate transfer count
    neox_transfer = sum(1 for v in neox if v > 5)
    llama_transfer = sum(1 for v in llama if v > 5)
    ax.text(0.02, 0.02, f"NeoX: {neox_transfer}/9 transfer.  Llama: {llama_transfer}/9 transfer.\nMechanism replicates across architectures.",
            transform=ax.transAxes, fontsize=9, color=GRAY, va="bottom")

    save(fig, "run003-transfer-comparison.png")


# ── Chart 2: Emergence Comparison ──

def chart_emergence_comparison():
    # NeoX (run-002)
    neox_steps = [1000, 1500, 2000, 2500, 3500, 4000, 4500, 5000]
    neox_heads = [107, 96, 110, 105, 70, 60, 66, 61]

    # Llama (run-003)
    llama_steps = [15000, 20000, 25000, 30000, 35000, 40000]
    llama_heads = [67, 71, 65, 57, 49, 66]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})

    # NeoX
    setup_ax(ax1, "GPT-NeoX 410M\nSteps 1K-5K", xlabel="Training step", ylabel="Delimiter heads")
    ax1.plot(neox_steps, neox_heads, color=CYAN, marker="D", markersize=8, linewidth=2.5)
    ax1.fill_between(neox_steps, neox_heads, alpha=0.1, color=CYAN)
    for s, h in zip(neox_steps, neox_heads):
        ax1.annotate(str(h), xy=(s, h), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT)
    ax1.set_ylim(40, 120)

    # Llama
    setup_ax(ax2, "Llama 410M\nSteps 15K-40K", xlabel="Training step", ylabel="Delimiter heads")
    ax2.plot(llama_steps, llama_heads, color=ORANGE, marker="D", markersize=8, linewidth=2.5)
    ax2.fill_between(llama_steps, llama_heads, alpha=0.1, color=ORANGE)
    for s, h in zip(llama_steps, llama_heads):
        ax2.annotate(str(h), xy=(s, h), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT)
    ax2.set_ylim(40, 120)

    fig.suptitle("Head Count Narrowing During Training\nSame pattern on both architectures: many heads emerge early, training prunes to a stable core",
                 fontsize=11, fontweight="bold", color=TEXT, y=1.03)

    save(fig, "run003-emergence-comparison.png")


# ── Chart 3: KV-Group Ablation Gaps ──

def chart_kvgroup_gaps():
    formats = ["GCF", "JSON", "YAML", "NL"]
    delim_delta = [-15.5, 67.8, 34.0, 13.8]
    random_delta = [-63.9, -96.2, -18.0, 28.5]
    gaps = [d - r for d, r in zip(delim_delta, random_delta)]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(formats))
    width = 0.3

    ax.bar(x - width, delim_delta, width, label="Delimiter KV groups", color=CYAN, edgecolor=BG)
    ax.bar(x, random_delta, width, label="Random KV groups", color=ORANGE, edgecolor=BG)
    ax.bar(x + width, gaps, width, label="Gap (delim - random)", color=GREEN, edgecolor=BG)

    for i, g in enumerate(gaps):
        ax.annotate(f"{g:+.0f}pp", xy=(x[i] + width, max(g, 0)),
                   xytext=(0, 5), textcoords="offset points", ha="center",
                   fontsize=10, fontweight="bold", color=GREEN)

    setup_ax(ax, "KV-Group Ablation: Delimiter vs Random on Llama\nThe gap is the causal signal (positive = delimiter groups more important)",
             ylabel="PPL change (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, color=TEXT, fontsize=12)
    ax.axhline(y=0, color=TEXT, linewidth=0.5, alpha=0.3, linestyle="--")
    ax.legend(facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper left")

    save(fig, "run003-kvgroup-gaps.png")


# ── Chart 4: Layer-wise Comparison ──

def chart_layer_comparison():
    groups = ["Early\n(0-7)", "Middle\n(8-15)", "Late\n(16-23)"]

    # GCF delta when ablating each layer group
    neox_gcf = [-10, 4, 63]
    llama_gcf = [-41, 20.1, 6.4]

    # Head counts per group
    neox_count = [6, 14, 20]
    llama_count = [25, 25, 16]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(groups))
    width = 0.35

    # Left: GCF delta
    setup_ax(ax1, "GCF PPL Delta by Layer Group\nWhere structural reasoning happens",
             ylabel="GCF PPL change when group ablated (%)")
    ax1.bar(x - width/2, neox_gcf, width, label="GPT-NeoX", color=CYAN, edgecolor=BG)
    ax1.bar(x + width/2, llama_gcf, width, label="Llama", color=ORANGE, edgecolor=BG)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, color=TEXT, fontsize=11)
    ax1.axhline(y=0, color=TEXT, linewidth=0.5, alpha=0.3, linestyle="--")
    ax1.legend(facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    for i, (n, l) in enumerate(zip(neox_gcf, llama_gcf)):
        ax1.annotate(f"{n:+.0f}%", xy=(x[i] - width/2, max(n, 0)),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=CYAN)
        ax1.annotate(f"{l:+.0f}%", xy=(x[i] + width/2, max(l, 0)),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, color=ORANGE)

    # Right: Head distribution
    setup_ax(ax2, "Delimiter Head Distribution by Layer Group\nGQA pushes specialization to earlier layers",
             ylabel="Delimiter heads in group")
    ax2.bar(x - width/2, neox_count, width, label="GPT-NeoX", color=CYAN, edgecolor=BG)
    ax2.bar(x + width/2, llama_count, width, label="Llama", color=ORANGE, edgecolor=BG)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, color=TEXT, fontsize=11)
    ax2.legend(facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    for i, (n, l) in enumerate(zip(neox_count, llama_count)):
        ax2.annotate(str(n), xy=(x[i] - width/2, n), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=10, color=CYAN, fontweight="bold")
        ax2.annotate(str(l), xy=(x[i] + width/2, l), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=10, color=ORANGE, fontweight="bold")

    save(fig, "run003-layer-comparison.png")


# ── Chart 5: B0 vs A0 Head Count ──

def chart_b0_vs_a0():
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ["NeoX A\n(structok)", "NeoX B\n(standard)", "Llama A0\n(structok)", "Llama B0\n(standard)"]
    heads = [50, 3, 66, 35]
    colors = [CYAN, GRAY, ORANGE, GRAY]
    functional = ["YES", "NO", "YES", "YES"]

    bars = ax.bar(models, heads, color=colors, edgecolor=BG, width=0.6)

    for bar, h, f in zip(bars, heads, functional):
        label = f"{h} heads"
        if f == "NO":
            label += "\n(non-functional)"
        else:
            label += "\n(functional)"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 8), textcoords="offset points", ha="center",
                   fontsize=11, fontweight="bold", color=TEXT)

    setup_ax(ax, "Delimiter Head Specialization by Model and Architecture\nGQA enables partial specialization even without merge barriers",
             ylabel="Delimiter-specialized heads (excess > 0.15)")

    ax.text(0.5, -0.15,
            "NeoX: binary (structok specializes, standard doesn't). Llama: spectrum (both specialize, structok more).",
            transform=ax.transAxes, fontsize=9, color=GRAY, ha="center")

    save(fig, "run003-b0-vs-a0-heads.png")


# ── Generate all ──

if __name__ == "__main__":
    print("Generating run-003 charts...")
    chart_transfer_comparison()
    chart_emergence_comparison()
    chart_kvgroup_gaps()
    chart_layer_comparison()
    chart_b0_vs_a0()
    print("Done.")
