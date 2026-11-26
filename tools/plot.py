"""Reproduce the two figures from the byte-level C run."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


INK, GRAY, RULE = "#171717", "#555555", "#222222"
BLUE, VERMILION = "#2449a0", "#aa3426"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, path):
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None}, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def curve_panel(ax, run, name):
    history = rows(run / "history.csv")
    summary, = rows(run / "summary.csv")
    x = [int(r["step"]) for r in history]
    train = [float(r["train_nll"]) for r in history]
    val = [float(r["val_nll"]) for r in history]
    best = int(summary["best_step"])
    baseline = float(summary["bigram_val_nll"])
    ax.axhline(baseline, color=GRAY, lw=1.15, ls=(0, (4, 3)), label="Bigram, validation")
    ax.plot(x, train, color=BLUE, lw=1.25, label="GPT, training sample")
    ax.plot(x, val, color=VERMILION, lw=1.1, marker="o", ms=2.4, mfc="white",
            markevery=max(1, len(x)//8), label="GPT, validation")
    ax.scatter(best, val[x.index(best)], s=27, marker="s", ec=INK, fc="white", zorder=4, lw=1,
               label="Selected checkpoint")
    ax.set_title(name, loc="left", pad=5)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlim(min(x), max(x))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel("Optimizer step")


def curve(run, output):
    other = run / "shakespeare"
    n = 2 if (other / "history.csv").exists() else 1
    fig, axes = plt.subplots(1, n, figsize=(3.75*n, 3.55), squeeze=False, sharey=True)
    ax = axes[0, 0]
    curve_panel(ax, run, "(a)  Lab records")
    ax.set_ylabel(r"Cross-entropy $\mathcal{L}$ (nats per target token)")
    if n == 2:
        curve_panel(axes[0, 1], other, "(b)  Shakespeare")
    handles, labels = ax.get_legend_handles_labels()
    order = (1, 2, 0, 3)
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, loc="lower center", ncol=2, columnspacing=1.6, fontsize=8)
    fig.subplots_adjust(left=.11 if n==2 else .2, right=.98, top=.92, bottom=.28, wspace=.1)
    save(fig, output / "learning-curves")


def token_label(token):
    if token == 256:
        return "BOS"
    if token == 32:
        return "␣"
    if 33 <= token < 127:
        return chr(token)
    return f"{token:02X}"


def attention(run, output, layer, head):
    data = rows(run / "attention.csv")
    n = math.isqrt(len(data))
    if n < 1 or n*n != len(data):
        raise ValueError("Incomplete attention matrix")
    probs = np.full((n, n), np.nan)
    tokens = [0] * n
    for r in data:
        q, k = int(r["query_pos"]), int(r["key_pos"])
        probs[q, k] = float(r["weight"])
        tokens[q] = int(r["query_token"])
    mask = np.triu(np.ones_like(probs, dtype=bool), k=1)
    if (not np.isfinite(probs).all() or (probs < 0).any() or
            np.any(probs[mask] != 0) or not np.allclose(probs.sum(axis=1), 1, atol=2e-6)):
        raise ValueError("Invalid causal attention weights")
    labels = [token_label(tok) for tok in tokens]
    cmap = colors.LinearSegmentedColormap.from_list("weights", ["#ffffff", "#bbdfdc", "#559d9b", "#074760"])
    cmap.set_bad((1, 1, 1, 0))
    fig, ax = plt.subplots(figsize=(5.55, 4.5))
    fig.subplots_adjust(left=.18, right=.95, top=.89, bottom=.15)
    # In the animation available positions grow, but the color scale and axes
    # stay fixed. Causal prefix invariance is checked by nablaflow check.
    boundaries = np.arange(n+1)-.5
    ax.pcolormesh(boundaries, boundaries, np.where(mask, .90, .98), vmin=0, vmax=1, cmap="gray")
    im = ax.pcolormesh(boundaries, boundaries, np.ma.array(probs, mask=mask),
                       vmin=0, vmax=1, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlim(-.5, n-.5)
    ax.set_ylim(n-.5, -.5)
    ax.set_axisbelow(False)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xticks(np.arange(n+1)-.5, minor=True)
    ax.set_yticks(np.arange(n+1)-.5, minor=True)
    ax.grid(which="minor", color="#b8b8b8", lw=.35)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=.73, pad=.05, ticks=[0, .25, .5, .75, 1])
    cbar.solids.set_rasterized(False)
    cbar.set_label("Attention weight", labelpad=10)
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.set(xlabel="Key position", ylabel="Query position")
    title = ax.set_title(f"Block {layer+1}, head {head+1}", loc="left", pad=7)
    # Save the static map before turning the same axes into an animated prefix.
    fig.savefig(output / "attention.svg", metadata={"Date": None}, bbox_inches="tight", pad_inches=.12)
    fig.savefig(output / "attention.png", dpi=180, bbox_inches="tight", pad_inches=.12)
    def show_prefix(last):
        rows_hidden = np.arange(n)[:, None] > last
        im.set_array(np.ma.array(probs, mask=mask | rows_hidden).ravel())
        title.set_text(f"Block {layer+1}, head {head+1}     Prefix length {last+1:2d} / {n}")
        return im, title
    frames = [*range(n), *([n-1]*4)]
    anim = FuncAnimation(fig, show_prefix, frames=frames, interval=450, repeat=True)
    anim.save(output / "attention.gif", writer=PillowWriter(fps=2.2), dpi=135)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--layer", type=int, default=1, help="Zero-based layer used for attention.csv")
    parser.add_argument("--head", type=int, default=0, help="Zero-based head used for attention.csv")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "stix", "font.size": 9,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": RULE,
                         "axes.linewidth": .75, "axes.titlesize": 10,
                         "xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.axisbelow": True, "svg.hashsalt": "nabla-flow"})
    curve(args.run, args.out)
    attention(args.run, args.out, args.layer, args.head)


if __name__ == "__main__":
    main()
