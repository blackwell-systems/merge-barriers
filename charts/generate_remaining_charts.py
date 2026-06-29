#!/usr/bin/env python3
"""Generate 3 charts from remaining ablation experiments (#19, #21, #22)."""

import json
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

FIGSIZE = (10, 6)
DPI = 150
OUTDIR = "/Users/dayna.blackwell/code/structok/charts"

# Load data
with open("/Users/dayna.blackwell/code/structok/runs/run-002-remaining-ablation-results.json") as f:
    data = json.load(f)


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


# ── Chart 1: Sufficiency at Scale (#22) ──

def chart_sufficiency_scaling():
    suf = data["sufficiency_scaling"]
    sizes = [r["size"] for r in suf["results"]]
    delim_deltas = [r["formats"]["gcf_generic"]["delim_only_delta_pct"] for r in suf["results"]]
    rand_deltas = [r["formats"]["gcf_generic"]["random_only_delta_mean_pct"] for r in suf["results"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    setup_ax(ax, "Sufficiency at Scale: 50 Delimiter Heads vs 50 Random",
             "Payload size (rows)", "PPL change from baseline (%)")

    ax.plot(sizes, delim_deltas, color=CYAN, marker="D", markersize=8, linewidth=2.5,
            label="Delimiter heads only (50)", zorder=5)
    ax.plot(sizes, rand_deltas, color=ORANGE, marker="o", markersize=8, linewidth=2.5,
            label="Random heads only (50)", zorder=5)

    # Fill the gap
    ax.fill_between(sizes, delim_deltas, rand_deltas, alpha=0.08, color=CYAN)

    # Annotate the gap at each size
    for s, d, r in zip(sizes, delim_deltas, rand_deltas):
        gap = d - r
        mid = (d + r) / 2
        ax.annotate(f"{gap:+.0f}pp", xy=(s, mid), fontsize=9, fontweight="bold",
                   color=CYAN, ha="left", va="center",
                   xytext=(8, 0), textcoords="offset points")

    # Annotate key values
    ax.annotate(f"{delim_deltas[0]:+.1f}%", xy=(sizes[0], delim_deltas[0]),
               xytext=(-15, -18), textcoords="offset points", fontsize=9, color=CYAN)
    ax.annotate(f"{delim_deltas[-1]:+.1f}%", xy=(sizes[-1], delim_deltas[-1]),
               xytext=(8, -12), textcoords="offset points", fontsize=9, color=CYAN)
    ax.annotate(f"{rand_deltas[0]:+.1f}%", xy=(sizes[0], rand_deltas[0]),
               xytext=(-15, 10), textcoords="offset points", fontsize=9, color=ORANGE)
    ax.annotate(f"{rand_deltas[-1]:+.1f}%", xy=(sizes[-1], rand_deltas[-1]),
               xytext=(8, 10), textcoords="offset points", fontsize=9, color=ORANGE)

    ax.axhline(y=0, color=TEXT, linewidth=0.5, alpha=0.3, linestyle="--")
    ax.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper right")

    # Subtitle annotation
    ax.text(0.02, 0.02,
            "13% of heads outperform the full model on structured data at every scale.\n"
            "Gap narrows with scale as holistic improvement matters more than specialization.",
            transform=ax.transAxes, fontsize=9, color=GRAY, va="bottom")

    save(fig, "sufficiency-scaling.png")


# ── Chart 2: Adversarial Robustness (#21) ──

def chart_adversarial_robustness():
    adv = data["adversarial_robustness"]
    a_det = adv["model_a_detection_pct"]
    a_abl = adv["model_a_ablated_detection_pct"]

    corruptions = ["wrong_delimiters", "missing_fields", "wrong_header", "swapped_values"]
    labels = ["Wrong\ndelimiters", "Missing\nfields", "Wrong\nheader", "Swapped\nvalues"]

    a_vals = [a_det[c] for c in corruptions]
    a_abl_vals = [a_abl[c] for c in corruptions]

    fig, ax = plt.subplots(figsize=(10, 6))
    setup_ax(ax, "Adversarial Robustness: Corruption Detection Under Ablation",
             "Corruption type", "PPL spike vs clean (%)")

    x = np.arange(len(corruptions))
    width = 0.35

    bars1 = ax.bar(x - width/2, a_vals, width, label="Model A (baseline)",
                   color=CYAN, edgecolor=BG, linewidth=0.5)
    bars2 = ax.bar(x + width/2, a_abl_vals, width, label="Model A (ablated)",
                   color=RED, edgecolor=BG, linewidth=0.5, alpha=0.8)

    # Value labels
    for bar, val in zip(bars1, a_vals):
        ax.annotate(f"{val:+.0f}%", xy=(bar.get_x() + bar.get_width()/2, max(val, 0)),
                   xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                   fontsize=9, fontweight="bold", color=CYAN)
    for bar, val in zip(bars2, a_abl_vals):
        y = max(val, 0) if val >= 0 else val
        offset = 5 if val >= 0 else -15
        ax.annotate(f"{val:+.0f}%", xy=(bar.get_x() + bar.get_width()/2, y),
                   xytext=(0, offset), textcoords="offset points", ha="center", va="bottom",
                   fontsize=9, fontweight="bold", color=RED)

    # Mark "lost" corruptions
    for i, (a, ab) in enumerate(zip(a_vals, a_abl_vals)):
        if abs(ab) < abs(a) * 0.5 and corruptions[i] != "clean":
            ax.annotate("LOST", xy=(x[i] + width/2, ab),
                       xytext=(0, -18), textcoords="offset points",
                       ha="center", fontsize=8, fontweight="bold", color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=10)
    ax.axhline(y=0, color=TEXT, linewidth=0.5, alpha=0.3, linestyle="--")
    ax.legend(facecolor="#1a1a1a", edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper left")

    ax.text(0.02, 0.02,
            "Ablation reduces structural corruption detection by ~56%.\n"
            "Wrong-delimiter detection retained (heads tuned for pipe, not comma).",
            transform=ax.transAxes, fontsize=9, color=GRAY, va="bottom")

    save(fig, "adversarial-robustness-ablation.png")


# ── Chart 3: Embedding Cohesion (#19) ──

def chart_embedding_cohesion():
    emb = data["embedding_space"]
    baseline_ratio = emb["baseline"]["ratio"]
    ablated_ratio = emb["ablated"]["ratio"]
    rand_ratios = emb["random_control"]["ratios"]
    rand_mean = emb["random_control"]["mean_ratio"]

    fig, ax = plt.subplots(figsize=(8, 6))
    setup_ax(ax, "Embedding Space Cohesion Under Ablation (#19)",
             "", "Delimiter/content cohesion ratio")

    categories = ["Baseline\n(all heads)", "Delimiter\nablated", "Random\nablated (mean)"]
    values = [baseline_ratio, ablated_ratio, rand_mean]
    colors = [CYAN, RED, ORANGE]

    bars = ax.bar(categories, values, width=0.5, color=colors, edgecolor=BG, linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.annotate(f"{val:.2f}x", xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                   fontsize=11, fontweight="bold", color=TEXT)

    # Show individual random seeds
    for i, r in enumerate(rand_ratios):
        ax.plot(2, r, "o", color=ORANGE, markersize=6, alpha=0.5, zorder=5)

    ax.set_ylim(0, 1.8)
    ax.axhline(y=1.0, color=TEXT, linewidth=0.5, alpha=0.3, linestyle="--")
    ax.annotate("1.0x = no cohesion advantage", xy=(2.3, 1.02), fontsize=9, color=GRAY)

    change = emb["changes"]["ratio_change_pct"]
    ax.text(0.02, 0.02,
            f"NULL RESULT: ratio changes by only {change:+.1f}% under ablation.\n"
            "Embedding structure is a whole-model property, not head-controlled.",
            transform=ax.transAxes, fontsize=9, color=GRAY, va="bottom")

    save(fig, "embedding-cohesion-ablation.png")


# ── Generate all ──

if __name__ == "__main__":
    print("Generating remaining ablation charts...")
    chart_sufficiency_scaling()
    chart_adversarial_robustness()
    chart_embedding_cohesion()
    print("Done.")
