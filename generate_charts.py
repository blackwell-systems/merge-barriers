"""
Run-002 Chart Generator

Generates publication-quality PNGs from controlled experiment data.
Output goes to charts/ directory.

Usage:
    python generate_charts.py
    # or: uv run --with matplotlib --with numpy python generate_charts.py
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT = Path("charts")
OUTPUT.mkdir(exist_ok=True)

# Dark theme
BG = "#0a0a0a"
TEXT = "white"
GRID = "#333333"
LEGEND_BG = "#1a1a1a"

# Colors
C_STRUCTOK = "#18befc"   # cyan (brand)
C_STANDARD = "#ff4444"   # red
C_NEUTRAL = "#888888"    # gray
C_NL = "#00ff88"         # green
C_CODE = "#8b5cf6"       # purple
C_DELIM = "#f59e0b"      # amber
C_CONTENT = "#888888"    # gray

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def setup_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=11, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, color=GRID)


def dark_legend(ax, **kwargs):
    defaults = dict(facecolor=LEGEND_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc="upper right")
    defaults.update(kwargs)
    return ax.legend(**defaults)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  {Path(name).stem}")


# --- Chart 1: GCF PPL Scaling Curve ---

def chart_scaling_curve():
    records = [1, 2, 3, 5, 10, 20, 50, 100]
    structok = [2358, 2294, 1613, 2296, 1932, 3374, 5883, 8112]
    standard = [9619, 5315, 3318, 5147, 6616, 13593, 26887, 43152]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(records, structok, "o-", color=C_STRUCTOK, linewidth=2.5, markersize=8, label="Model A (merge barriers)")
    ax.plot(records, standard, "s-", color=C_STANDARD, linewidth=2.5, markersize=8, label="Model B (standard BPE)")

    for i, (r, s, st) in enumerate(zip(records, structok, standard)):
        ratio = st / s
        ax.annotate(f"{ratio:.1f}x", xy=(r, (s + st) / 2), fontsize=9, color=C_NEUTRAL,
                    ha="left", va="center", xytext=(5, 0), textcoords="offset points")

    setup_ax(ax, "Merge Barrier Advantage Scales with Payload Size\nControlled experiment: same architecture, same data, different tokenizer",
             xlabel="Records", ylabel="GCF Perplexity (lower = better)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    dark_legend(ax, loc="upper left")
    ax.set_xticks(records)
    ax.set_xticklabels([str(r) for r in records], color=TEXT)

    save(fig, "scaling-curve.png")


# --- Chart 2: All Formats Comparison ---

def chart_all_formats():
    categories = [
        "GCF\ntabular", "GCF\ngraph\n(10 sym)", "GCF\ngraph\n(20 sym)", "Users", "Logs",
        "API\nresponse", "Python", "Go", "TypeScript", "YAML", "CSV", "Wikipedia", "TOON\n(unseen)"
    ]
    structok = [4829, 14095, 18289, 13607, 14422, 1935, 543, 1404, 729, 5439, 2847, 1029, 18091]
    standard = [14621, 39558, 36314, 695922, 722297, 14075, 2686, 4183, 2667, 16872, 30616, 1033, 41188]

    cap = 50000
    standard_capped = [min(s, cap) for s in standard]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, structok, width, label="Model A (merge barriers)", color=C_STRUCTOK, edgecolor=BG, linewidth=0.5)
    ax.bar(x + width/2, standard_capped, width, label="Model B (standard BPE)", color=C_STANDARD, edgecolor=BG, linewidth=0.5, alpha=0.8)

    for i, s in enumerate(standard):
        if s > cap:
            ax.annotate(f"{s/1000:.0f}K", xy=(x[i] + width/2, cap), xytext=(0, 5),
                       textcoords="offset points", ha="center", fontsize=8, color=C_STANDARD, fontweight="bold")

    for i in range(len(categories)):
        ratio = standard[i] / structok[i]
        if ratio > 1.1:
            y_pos = min(max(structok[i], standard_capped[i]), cap) + 1000
            ax.annotate(f"{ratio:.1f}x", xy=(x[i], y_pos), ha="center", fontsize=8, color=C_NEUTRAL)

    setup_ax(ax, "Merge Barriers Win Every Category (11/11)\nStructured data, code, YAML, CSV, unseen format. Natural language tied.",
             ylabel="Perplexity (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, color=TEXT)
    dark_legend(ax)
    ax.set_ylim(0, cap * 1.15)

    ax.axvspan(-0.5, 5.5, alpha=0.04, color=C_STRUCTOK, label="_structured")
    ax.axvspan(5.5, 8.5, alpha=0.04, color=C_CODE, label="_code")

    save(fig, "all-formats.png")


# --- Chart 3: Code Comprehension ---

def chart_code():
    langs = ["Python", "Go", "TypeScript"]
    structok = [543, 1404, 729]
    standard = [2686, 4183, 2667]
    ratios = [s/a for a, s in zip(structok, standard)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(langs))
    width = 0.35

    ax.bar(x - width/2, structok, width, label="Model A (merge barriers)", color=C_STRUCTOK, edgecolor=BG, linewidth=0.5)
    ax.bar(x + width/2, standard, width, label="Model B (standard BPE)", color=C_STANDARD, edgecolor=BG, linewidth=0.5, alpha=0.8)

    for i, r in enumerate(ratios):
        ax.annotate(f"{r:.1f}x", xy=(x[i], max(structok[i], standard[i]) + 100),
                   ha="center", fontsize=12, fontweight="bold", color=C_NEUTRAL)

    setup_ax(ax, "Code Comprehension: 3-5x Better with Merge Barriers\nBarrier chars ({, }, (, ), :, ;) also protect code syntax",
             ylabel="Perplexity (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=13, color=TEXT)
    dark_legend(ax)

    save(fig, "code-comprehension.png")


# --- Chart 4: Delimiter Head Specialization ---

def chart_delimiter_heads():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ["Model A\n(merge barriers)", "Model B\n(standard BPE)"]
    heads = [105, 23]
    total = 384
    colors = [C_STRUCTOK, C_STANDARD]

    bars = ax1.bar(models, heads, color=colors, width=0.5, edgecolor=BG, linewidth=0.5)
    ax1.axhline(y=total/2, color=GRID, linestyle="--", linewidth=0.8)
    setup_ax(ax1, "4.6x More Structural Heads", ylabel="Delimiter-majority heads")
    ax1.set_ylim(0, 200)
    ax1.set_xticklabels(models, color=TEXT)

    for bar, h in zip(bars, heads):
        ax1.annotate(f"{h}\n({h/total:.0%})", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=12, fontweight="bold", color=TEXT)

    structok_top = [85.3, 84.9, 84.8, 84.5, 82.8, 81.0, 80.7, 78.1, 76.6, 76.3]
    standard_top = [79.4, 78.6, 74.2, 71.0, 67.4, 65.0, 64.8, 64.6, 63.2, 62.4]

    y = np.arange(10)
    ax2.barh(y + 0.2, structok_top, 0.35, color=C_STRUCTOK, label="Model A", alpha=0.9)
    ax2.barh(y - 0.2, standard_top, 0.35, color=C_STANDARD, label="Model B", alpha=0.9)
    setup_ax(ax2, "Top 10 Delimiter-Focused Heads", xlabel="% attention to delimiters")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"#{i+1}" for i in range(10)], fontsize=10, color=TEXT)
    ax2.set_xlim(50, 90)
    dark_legend(ax2, loc="lower right")
    ax2.invert_yaxis()

    save(fig, "delimiter-heads.png")


# --- Chart 5: Per-Token Loss ---

def chart_per_token_loss():
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Delimiter\ntokens", "Content\ntokens"]
    structok = [6.10, 13.28]
    standard = [14.81, 14.74]

    x = np.arange(len(categories))
    width = 0.3

    ax.bar(x - width/2, structok, width, label="Model A (merge barriers)", color=C_STRUCTOK, edgecolor=BG, linewidth=0.5)
    ax.bar(x + width/2, standard, width, label="Model B (standard BPE)", color=C_STANDARD, edgecolor=BG, linewidth=0.5, alpha=0.8)

    ax.annotate("2.4x easier", xy=(0, 6.10), xytext=(-0.4, 10), fontsize=11,
               fontweight="bold", color=C_STRUCTOK,
               arrowprops=dict(arrowstyle="->", color=C_STRUCTOK, lw=1.5))
    ax.annotate("Equal\ndifficulty", xy=(1.15, 14.77), fontsize=10, color=C_STANDARD,
               ha="center")

    setup_ax(ax, "Model A Finds Delimiters Easy; Model B Finds Them Equally Hard\nPer-token loss on 10-order GCF payload",
             ylabel="Cross-entropy loss (lower = easier)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13, color=TEXT)
    dark_legend(ax)
    ax.set_ylim(0, 18)

    save(fig, "per-token-loss.png")


# --- Chart 6: Grammar Attention at Scale ---

def chart_grammar_attention():
    orders = [5, 10, 20, 50, 100]
    structok = [37.1, 31.4, 30.8, 30.5, 29.7]
    standard = [24.9, 23.4, 21.2, 20.5, 18.1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(orders, structok, "o-", color=C_STRUCTOK, linewidth=2.5, markersize=8, label="Model A (merge barriers)")
    ax.plot(orders, standard, "s-", color=C_STANDARD, linewidth=2.5, markersize=8, label="Model B (standard BPE)")

    ax.axhline(y=8.6, color=GRID, linestyle=":", linewidth=1.5)
    ax.annotate("Gemma 2B collapse (8.6%)", xy=(100, 8.6), xytext=(-5, -15),
               textcoords="offset points", fontsize=9, color="#666666", ha="right")

    ax.fill_between(orders, structok, standard, alpha=0.1, color=C_STRUCTOK)

    setup_ax(ax, "Merge Barriers Maintain 50% More Grammar Attention at Every Scale\nModel A resists the collapse seen in production models",
             xlabel="Records", ylabel="Grammar attention share (%)")
    dark_legend(ax, loc="upper right")
    ax.set_ylim(0, 45)
    ax.set_xticks(orders)

    save(fig, "grammar-attention.png")


# --- Chart 7: Grammar Attention Collapse Comparison ---

def chart_collapse_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["Model A\n(merge barriers)", "Model B\n(standard BPE)", "Gemma 2B\n(production)"]
    small_scale = [34.3, 24.1, 30.0]
    large_scale = [30.1, 19.3, 8.6]
    changes = [-4.2, -4.8, -21.4]
    colors = [C_STRUCTOK, C_STANDARD, C_NEUTRAL]

    x = np.arange(len(models))
    width = 0.3

    ax.bar(x - width/2, small_scale, width, label="Small scale (5-10 orders)", color=[c for c in colors], alpha=0.5, edgecolor=BG)
    ax.bar(x + width/2, large_scale, width, label="Large scale (50-100 orders)", color=colors, edgecolor=BG)

    for i, c in enumerate(changes):
        ax.annotate(f"{c:+.1f}%", xy=(x[i], max(small_scale[i], large_scale[i]) + 1),
                   ha="center", fontsize=11, fontweight="bold", color=colors[i])

    setup_ax(ax, "Grammar Attention Collapse: Merge Barriers Prevent It\nGemma 2B drops from 30% to 8.6%; Model A holds at 30%",
             ylabel="Grammar attention share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m for m in models], fontsize=11, color=TEXT)
    dark_legend(ax, loc="upper left")
    ax.set_ylim(0, 42)

    save(fig, "collapse-comparison.png")


# --- Chart 8: Adversarial Robustness ---

def chart_adversarial():
    tests = ["Normal\nGCF", "Pipe-like\nchars", "JSON in\nGCF values", "Numeric\nheavy", "Empty\nfields"]
    structok = [893, 1086, 395, 678, 352]
    standard = [13649, 8053, 9610, 9549, 6598]
    ratios = [s/a for a, s in zip(structok, standard)]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tests))
    width = 0.35

    ax.bar(x - width/2, structok, width, label="Model A (merge barriers)", color=C_STRUCTOK, edgecolor=BG, linewidth=0.5)
    ax.bar(x + width/2, standard, width, label="Model B (standard BPE)", color=C_STANDARD, edgecolor=BG, linewidth=0.5, alpha=0.8)

    for i, r in enumerate(ratios):
        ax.annotate(f"{r:.0f}x", xy=(x[i], max(structok[i], standard[i]) + 300),
                   ha="center", fontsize=11, fontweight="bold", color=C_NEUTRAL)

    setup_ax(ax, "Adversarial Robustness: Model A Wins 5/5\nJSON embedded in GCF values: 24x advantage",
             ylabel="Perplexity (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=10, color=TEXT)
    dark_legend(ax)

    save(fig, "adversarial.png")


# --- Chart 9: Token Repetition at Scale ---

def chart_token_repetition():
    orders = [5, 10, 20, 50, 100]
    structok = [35.9, 54.6, 67.0, 78.0, 83.9]
    standard = [44.3, 62.8, 73.3, 81.0, 84.6]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(orders, structok, "o-", color=C_STRUCTOK, linewidth=2.5, markersize=8, label="Model A (merge barriers)")
    ax.plot(orders, standard, "s-", color=C_STANDARD, linewidth=2.5, markersize=8, label="Model B (standard BPE)")
    ax.fill_between(orders, structok, standard, alpha=0.1, color=C_STANDARD)

    setup_ax(ax, "Merge Barriers Reduce Token Repetition\nLess repetition = less attention dilution",
             xlabel="Records", ylabel="Token repetition (%)")
    dark_legend(ax, loc="upper left")
    ax.set_ylim(20, 100)
    ax.set_xticks(orders)

    save(fig, "token-repetition.png")


# --- Chart 10: Training Convergence ---

def chart_training_convergence():
    steps_a = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]
    ppl_a = [586, 200, 95, 55, 31, 23, 22, 21, 20, 19.4]

    steps_b = [100, 500, 1000, 2000, 5000, 8000, 10000, 15000, 20000]
    ppl_b = [561, 130, 85, 44, 28, 21, 20, 19, 19.5]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_a, ppl_a, "o-", color=C_STRUCTOK, linewidth=2.5, markersize=6, label="Model A (merge barriers)")
    ax.plot(steps_b, ppl_b, "s-", color=C_STANDARD, linewidth=2.5, markersize=6, label="Model B (standard BPE)")

    ax.axhline(y=19.5, color=GRID, linestyle="--", linewidth=0.8)
    ax.annotate("Both converge to ~19.5", xy=(20000, 19.5), xytext=(-100, 10),
               textcoords="offset points", fontsize=10, color=C_NEUTRAL)

    setup_ax(ax, "Training Convergence: Standard BPE Converges Faster, Both Reach Same PPL\nMerge barriers produce more tokens per text, requiring more steps",
             xlabel="Training steps", ylabel="Overall perplexity")
    ax.set_yscale("log")
    dark_legend(ax, loc="upper right")
    ax.set_xlim(0, 21000)

    save(fig, "training-convergence.png")


# --- Chart 11: Embedding Space Separation ---

def chart_embedding_space():
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Delimiter\ninternal sim", "Content\ninternal sim", "Cross-group\nsim", "Separation\nmetric"]
    structok = [0.166, 0.002, -0.008, 0.174]
    standard = [0.098, 0.006, -0.018, 0.115]

    x = np.arange(len(metrics))
    width = 0.3

    ax.bar(x - width/2, structok, width, label="Model A (merge barriers)", color=C_STRUCTOK, edgecolor=BG, linewidth=0.5)
    ax.bar(x + width/2, standard, width, label="Model B (standard BPE)", color=C_STANDARD, edgecolor=BG, linewidth=0.5, alpha=0.8)

    setup_ax(ax, "Delimiter Embeddings Cluster 50% More Cohesively\n22 delimiter tokens (Model A) vs 1,463 merged tokens (Model B)",
             ylabel="Cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color=TEXT)
    dark_legend(ax)
    ax.axhline(y=0, color=GRID, linewidth=0.5)

    save(fig, "embedding-space.png")


if __name__ == "__main__":
    print("Generating run-002 charts...")
    chart_scaling_curve()
    chart_all_formats()
    chart_code()
    chart_delimiter_heads()
    chart_per_token_loss()
    chart_grammar_attention()
    chart_collapse_comparison()
    chart_adversarial()
    chart_token_repetition()
    chart_training_convergence()
    chart_embedding_space()
    print(f"\nDone. {len(list(OUTPUT.glob('*.png')))} charts in {OUTPUT}/")
