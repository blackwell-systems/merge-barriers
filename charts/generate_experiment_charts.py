#!/usr/bin/env python3
"""Generate 5 experiment charts for structok research."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTDIR = '/Users/dayna.blackwell/code/structok/charts'

# Theme
BG = '#0a0a0a'
TEXT = 'white'
GRID = '#333333'
CYAN = '#18befc'
GREEN = '#00ff88'
RED = '#ff4444'
GRAY = '#888888'
FIGSIZE = (10, 6)
DPI = 150

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT)
    ax.grid(True, color=GRID, alpha=0.5, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)


# ── Chart 1: Emergence Timeline ──

def chart1():
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax2 = ax1.twinx()

    # Run 1
    steps_r1 = [1000, 1500, 2000, 2500]
    heads_r1 = [107, 96, 110, 105]
    conc_r1 = [37.2, 39.5, 37.8, 37.3]

    # Run 2
    steps_r2 = [3500, 4000, 4500, 5000]
    heads_r2 = [70, 60, 66, 61]
    conc_r2 = [50.2, 54.4, 53.4, 54.1]

    ax1.plot(steps_r1, heads_r1, color=CYAN, marker='o', linewidth=2, markersize=7, label='Head count')
    ax1.plot(steps_r2, heads_r2, color=CYAN, marker='o', linewidth=2, markersize=7)

    ax2.plot(steps_r1, conc_r1, color=GREEN, linestyle='--', linewidth=2, marker='s', markersize=5, label='Concentration')
    ax2.plot(steps_r2, conc_r2, color=GREEN, linestyle='--', linewidth=2, marker='s', markersize=5)

    # Gap indicator
    ax1.axvspan(2500, 3500, color=GRAY, alpha=0.1)
    ax1.axvline(3000, color=GRAY, linestyle=':', alpha=0.4)

    # Annotations
    ax1.annotate('Run 1', xy=(1750, 112), color=CYAN, fontsize=11, fontweight='bold', ha='center')
    ax1.annotate('Run 2', xy=(4250, 72), color=CYAN, fontsize=11, fontweight='bold', ha='center')

    style_ax(ax1, 'Delimiter Head Emergence During Training', 'Training Step', 'Delimiter Head Count')
    ax2.set_ylabel('Concentration Ratio (%)', color=TEXT, fontsize=11)
    ax2.tick_params(colors=TEXT)
    ax2.spines['right'].set_color(GRID)

    ax1.set_xlim(500, 5500)
    ax1.set_ylim(40, 125)
    ax2.set_ylim(30, 60)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               facecolor='#1a1a1a', edgecolor=GRID, labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/emergence-timeline.png', dpi=DPI, facecolor=BG)
    plt.close()
    print('  emergence-timeline.png')


# ── Chart 2: Structural Pattern Test ──

def chart2():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    labels = ['A: tab + GCF\nlayout', 'B: tab + TSV\nlayout', 'C: tab +\nwrapping',
              'D: pipe +\nwrapping', 'E: GCF\ncontrol']
    values = [51.2, 32.1, 123.4, -54.0, 59.9]
    colors = [CYAN, CYAN, CYAN, RED, CYAN]

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='none')

    ax.axhline(0, color=TEXT, linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, values):
        y = val + (3 if val > 0 else -6)
        ax.text(bar.get_x() + bar.get_width()/2, y, f'{val:+.1f}%',
                ha='center', va='bottom' if val > 0 else 'top', color=TEXT, fontsize=10, fontweight='bold')

    # Annotation on bar D
    ax.annotate('pipe adversarial\nin wrapping', xy=(3, -54), xytext=(3.6, -35),
                color=RED, fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                ha='center')

    style_ax(ax, 'Structural Pattern Test: Character vs Layout', '', 'Ablation Delta (%)')
    ax.set_ylim(-70, 140)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/structural-pattern-test.png', dpi=DPI, facecolor=BG)
    plt.close()
    print('  structural-pattern-test.png')


# ── Chart 3: Production Probing ──

def chart3():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Our models
    our = [
        ('Model A\n(merge barriers)', 54.3, 0.349, 384),
        ('Model B\n(standard BPE)', 63.6, 0.282, 384),
    ]
    # Production models
    prod = [
        ('Gemma 2 2B', 18.0, 0.662, 208),
        ('Llama 3.1 8B', 14.9, 0.755, 1024),
        ('Mistral 7B', 14.5, 0.836, 1024),
        ('Qwen 2.5 7B', 72.6, 0.247, 784),
    ]

    scale = 0.15  # size scaling

    for name, conc, top10, heads in our:
        ax.scatter(conc, top10, s=heads*scale, color=CYAN, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)
        ax.annotate(name, (conc, top10), textcoords='offset points', xytext=(10, -10),
                    color=CYAN, fontsize=8.5, fontweight='bold')

    prod_offsets = {
        'Gemma 2 2B': (12, -15),
        'Llama 3.1 8B': (12, 5),
        'Mistral 7B': (-10, 10),
        'Qwen 2.5 7B': (10, -15),
    }
    for name, conc, top10, heads in prod:
        ax.scatter(conc, top10, s=heads*scale, color=GREEN, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=5)
        offset = prod_offsets.get(name, (10, 8))
        ha = 'right' if name == 'Mistral 7B' else 'left'
        ax.annotate(name, (conc, top10), textcoords='offset points', xytext=offset,
                    color=GREEN, fontsize=8.5, fontweight='bold', ha=ha)

    # Cluster annotations
    ax.annotate('Concentrated\nspecialists', xy=(42, 0.32), fontsize=10, color=GRAY,
                fontstyle='italic', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=GRAY, alpha=0.7))
    ax.annotate('Diffuse\ngeneralists', xy=(30, 0.88), fontsize=10, color=GRAY,
                fontstyle='italic', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a', edgecolor=GRAY, alpha=0.7))

    # Legend
    ax.scatter([], [], s=50, color=CYAN, label='Our models')
    ax.scatter([], [], s=50, color=GREEN, label='Production models')
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor=GRID, labelcolor=TEXT)

    style_ax(ax, 'Production Model Probing: Concentration vs Depth',
             'Concentration Ratio (%)', 'Top-10 Excess Delimiter Attention')
    ax.set_xlim(5, 85)
    ax.set_ylim(0.15, 0.95)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/production-probing.png', dpi=DPI, facecolor=BG)
    plt.close()
    print('  production-probing.png')


# ── Chart 4: Per-Token Loss Ablation ──

def chart4():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    groups = ['Model A\nbaseline', 'Model A\nablated', 'Model B']
    delim = [6.1, 5.7, 14.8]
    content = [13.3, 11.4, 14.7]

    x = np.arange(len(groups))
    w = 0.3

    ax.bar(x - w/2, delim, w, color=CYAN, label='Delimiter loss')
    ax.bar(x + w/2, content, w, color=GRAY, label='Content loss')

    # Value labels
    for i in range(len(groups)):
        ax.text(x[i] - w/2, delim[i] + 0.3, f'{delim[i]:.1f}', ha='center', color=TEXT, fontsize=9)
        ax.text(x[i] + w/2, content[i] + 0.3, f'{content[i]:.1f}', ha='center', color=TEXT, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)

    # Annotation
    ax.text(0.5, 0.95, 'Ablating heads does NOT spike delimiter loss.\nThe 2.4x advantage is a whole-model property.',
            transform=ax.transAxes, ha='center', va='top', color=TEXT, fontsize=10,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a', edgecolor=GRAY, alpha=0.8))

    ax.legend(loc='upper left', facecolor='#1a1a1a', edgecolor=GRID, labelcolor=TEXT)
    style_ax(ax, 'Per-Token Loss Under Ablation (Null Result)', '', 'Loss')
    ax.set_ylim(0, 18)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/per-token-loss-ablation.png', dpi=DPI, facecolor=BG)
    plt.close()
    print('  per-token-loss-ablation.png')


# ── Chart 5: Transplant Controls ──

def chart5():
    fig, ax = plt.subplots(figsize=FIGSIZE)

    formats = ['gcf_generic', 'json', 'toon', 'csv', 'nl']
    delim_vals = [-81.2, -86.4, -32.5, -59.3, 11.8]
    rand_vals = [-69.6, -99.2, -86.6, -95.3, 1.4]

    x = np.arange(len(formats))
    w = 0.3

    ax.bar(x - w/2, delim_vals, w, color=CYAN, label='Delimiter heads A->B')
    ax.bar(x + w/2, rand_vals, w, color=GRAY, label='Random heads A->B')

    ax.axhline(0, color=TEXT, linewidth=0.8)

    # Value labels
    for i in range(len(formats)):
        yd = delim_vals[i] - 4 if delim_vals[i] < 0 else delim_vals[i] + 2
        yr = rand_vals[i] - 4 if rand_vals[i] < 0 else rand_vals[i] + 2
        ax.text(x[i] - w/2, yd, f'{delim_vals[i]:+.1f}%', ha='center', color=TEXT, fontsize=7.5, rotation=90)
        ax.text(x[i] + w/2, yr, f'{rand_vals[i]:+.1f}%', ha='center', color=TEXT, fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(formats)

    # Annotation
    ax.text(0.5, 0.48, 'Random heads also help. The improvement is holistic,\nnot delimiter-specific.',
            transform=ax.transAxes, ha='center', va='center', color=TEXT, fontsize=10,
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a', edgecolor=GRAY, alpha=0.9))

    ax.legend(loc='lower right', facecolor='#1a1a1a', edgecolor=GRID, labelcolor=TEXT)
    style_ax(ax, 'Head Transplant Controls (20 heads)', '', 'PPL Change from Model B Baseline (%)')
    ax.set_ylim(-115, 30)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/transplant-controls.png', dpi=DPI, facecolor=BG)
    plt.close()
    print('  transplant-controls.png')


if __name__ == '__main__':
    print('Generating charts...')
    chart1()
    chart2()
    chart3()
    chart4()
    chart5()
    print('Done.')
