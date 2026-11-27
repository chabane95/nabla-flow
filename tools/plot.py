"""Reproduce the two figures from the byte-level C run."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Polygon
import numpy as np


INK, GRAY = "#17212b", "#656b70"
BLUE, VERMILION = "#245787", "#aa4430"
MASK, HATCH = "#eeeeeb", "#a7a9a6"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, path):
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=320)


def curve_panel(ax, run, name):
    history = rows(run / "history.csv")
    summary, = rows(run / "summary.csv")
    x = [int(r["step"]) for r in history]
    train = [float(r["train_nll"]) for r in history]
    val = [float(r["val_nll"]) for r in history]
    best = int(summary["best_step"])
    baseline = float(summary["bigram_val_nll"])
    ax.plot(x, train, color=BLUE, lw=1.3, label="Training sample")
    ax.plot(x, val, color=VERMILION, lw=.88, marker="o", ms=2.65, mew=.68, mfc="white",
            label="Validation")
    ax.axhline(baseline, color=GRAY, lw=.87, ls=(0, (4, 2.5)), label="Bigram validation")
    minimum = val[x.index(best)]
    ax.plot(best, minimum, marker="s", ms=5, mew=.95, color=INK, mfc="white", ls="none",
            zorder=4, label="Selected checkpoint")
    ax.annotate(f"{minimum:.3f}", (best, minimum), xytext=(-8, -14 if best > .85*max(x) else 12),
                textcoords="offset points", ha="right", va="center", fontsize=8.5,
                arrowprops={"arrowstyle": "-", "lw": .55, "color": GRAY, "shrinkB": 3})
    ax.set_title(name, loc="left", pad=7, fontsize=10.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.margins(x=.015, y=.08)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4, steps=(1, 2, 3, 6, 10), integer=True))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(6))
    ax.grid(axis="y", which="major", color="#d9dad8", lw=.42, ls=(0, (1.5, 2.4)))
    ax.set_xlabel("Optimizer step", labelpad=3)


def curve(run, output):
    other = run / "shakespeare"
    n = 2 if (other / "history.csv").exists() else 1
    fig, axes = plt.subplots(1, n, figsize=(7.1 if n==2 else 3.75, 3.55), squeeze=False, sharey=True)
    ax = axes[0, 0]
    curve_panel(ax, run, r"$\mathbf{(a)}$   Lab records" if n==2 else "Lab records")
    ax.set_ylabel(r"Cross-entropy $\mathcal{L}$  (nats per target)")
    if n == 2:
        curve_panel(axes[0, 1], other, r"$\mathbf{(b)}$   Shakespeare")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(.5, .015),
               ncol=4 if n==2 else 2, columnspacing=1.6, handlelength=2.1, fontsize=8.7)
    fig.subplots_adjust(left=.09 if n==2 else .17, right=.98, top=.91,
                        bottom=.24 if n==2 else .30, wspace=.14)
    save(fig, output / "learning-curves")
    plt.close(fig)


def token_label(token):
    if token == 256:
        return "BOS"
    if token == 32:
        return "space"
    if token == 257:
        return "EOS"
    if 33 <= token < 127:
        return chr(token)
    return f"{token:02X}"


def attention(run, output, layer, head):
    data = rows(run / "attention.csv")
    n = math.isqrt(len(data))
    if n < 1 or n*n != len(data):
        raise ValueError("Incomplete attention matrix")
    probs = np.full((n, n), np.nan)
    tokens = [None] * n
    for r in data:
        q, k = int(r["query_pos"]), int(r["key_pos"])
        if not (0 <= q < n and 0 <= k < n):
            raise ValueError("Attention position outside the matrix")
        probs[q, k] = float(r["weight"])
        for pos, field in ((q, "query_token"), (k, "key_token")):
            token = int(r[field])
            if tokens[pos] is not None and tokens[pos] != token:
                raise ValueError("Inconsistent token IDs at one attention position")
            tokens[pos] = token
    mask = np.triu(np.ones_like(probs, dtype=bool), k=1)
    if (not np.isfinite(probs).all() or (probs < 0).any() or
            np.any(probs[mask] != 0) or not np.allclose(probs.sum(axis=1), 1, atol=2e-6)):
        raise ValueError("Invalid causal attention weights")
    labels = [token_label(tok) for tok in tokens]
    fig, ax = plt.subplots(figsize=(5.3, 4.55))
    fig.subplots_adjust(left=.155, right=.785, top=.905, bottom=.24)
    boundaries = np.arange(n+1)-.5
    im = ax.pcolormesh(boundaries, boundaries, np.ma.array(probs, mask=mask),
                       vmin=0, vmax=1, cmap="Blues")
    # A single hatched staircase identifies inaccessible keys, even in grayscale.
    # Its edge follows cell boundaries, keeping the diagonal itself measurable.
    if n > 1:
        outline = [(.5, -.5), (n-.5, -.5), (n-.5, n-1.5)]
        for i in range(n-2, -1, -1):
            outline.extend([(i+.5, i+.5), (i+.5, i-.5)])
        ax.add_patch(Polygon(outline, facecolor=MASK, edgecolor=HATCH, linewidth=0,
                             hatch="///", zorder=2))
    ax.set_aspect("equal")
    ax.set_xlim(-.5, n-.5)
    ax.set_ylim(n-.5, -.5)
    ax.set_axisbelow(False)
    positions = sorted(set(range(0, n, max(1, math.ceil(n/14)))) | {n-1})
    ax.set_xticks(positions, [f"{i}\n{labels[i]}" for i in positions])
    ax.set_yticks(positions, [f"{labels[i]}   {i}" for i in positions])
    ax.set_xticks(np.arange(n+1)-.5, minor=True)
    ax.set_yticks(np.arange(n+1)-.5, minor=True)
    if n <= 32:
        ax.grid(which="minor", color="#adbac5", alpha=.6, lw=.27)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0, pad=3, labelsize=8.1)
    cbax = fig.add_axes([.83, .295, .022, .55])
    cbar = fig.colorbar(im, cax=cbax, ticks=[0, .25, .5, .75, 1])
    cbar.solids.set_rasterized(False)
    cbar.set_label(r"Attention weight $a_{ij}$", labelpad=7)
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    ax.set(xlabel=r"Key position $j$ and input token", ylabel=r"Query token and position $i$")
    ax.set_title(f"Block {layer+1}, head {head+1}", loc="left", pad=8)
    prefix = ax.set_title("", loc="right", fontsize=9, pad=8)
    masked = Patch(fc=MASK, ec=HATCH, lw=.5, hatch="///", label=r"Future key ($j>i$)")
    legend = fig.legend(handles=[masked], loc="lower center", bbox_to_anchor=(.5,.016),
                        frameon=False, handlelength=1.65)
    save(fig, output / "attention")
    legend.remove()
    unseen = "#eee2c6"
    ax.set_facecolor(unseen)
    fig.legend(handles=[masked, Patch(fc=unseen, ec=HATCH, lw=.5, label="Unrevealed query")],
               loc="lower center", bbox_to_anchor=(.5,.016), frameon=False, ncol=2,
               handlelength=1.65, columnspacing=2.2)
    def show_prefix(last):
        rows_hidden = np.arange(n)[:, None] > last
        im.set_array(np.ma.array(probs, mask=mask | rows_hidden).ravel())
        prefix.set_text(f"Prefix length  {last+1} / {n}")
        return im, prefix
    frames = [*range(n), *([n-1]*4)]
    anim = FuncAnimation(fig, show_prefix, frames=frames, interval=450, repeat=True)
    anim.save(output / "attention.gif", writer=PillowWriter(fps=2.2), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--layer", type=int, default=1, help="Zero-based layer used for attention.csv")
    parser.add_argument("--head", type=int, default=0, help="Zero-based head used for attention.csv")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9.2,
                         "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
                         "axes.linewidth": .7, "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "xtick.top": True, "ytick.right": True, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2,
                         "ytick.minor.size": 2, "xtick.color": INK, "ytick.color": INK,
                         "hatch.linewidth": .42, "axes.axisbelow": True, "svg.hashsalt": "nabla-flow",
                         "pdf.fonttype": 42})
    curve(args.run, args.out)
    attention(args.run, args.out, args.layer, args.head)


if __name__ == "__main__":
    main()
