"""Plot two parameter matrices directly from the trained byte decoder."""

import argparse
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BYTE_IDS = [65, 66, 71, 72, 97, 108, 114, 122]


def read_matrices(path):
    data = path.read_bytes()
    header = struct.Struct("<5IQ256x")
    start = 8 + header.size
    if len(data) < start + 8 or data[:8] != b"NFGPT01\n":
        raise ValueError(f"{path} has no complete NFGPT01 header")
    vocab, context, width, heads, layers, count = header.unpack_from(data, 8)
    if (vocab != 258 or min(context, width, heads, layers) < 1 or width % heads
            or count != (vocab + context)*width + layers*(12*width**2 + 13*width) + 2*width
            or len(data) != start + 4*count + 8):
        raise ValueError(f"{path} has incompatible byte decoder dimensions")
    digest = 14695981039346656037
    for byte in memoryview(data)[8:-8]:
        digest = ((digest ^ byte) * 1099511628211) & 0xffffffffffffffff
    if digest != int.from_bytes(data[-8:], "little"):
        raise ValueError(f"{path} failed its FNV-1a checksum")
    weights = np.frombuffer(data, dtype="<f4", count=count, offset=start)
    embedding = weights[:vocab*width].reshape(vocab, width)
    # Layer 0 starts after the two embedding tables. Its first matrix stacks Q, K, V.
    q_start = (vocab + context)*width
    query = weights[q_start:q_start + width**2].reshape(width, width)
    return embedding[BYTE_IDS], query[:width//heads]


def draw(model, output):
    embedding, query = read_matrices(model)
    width = embedding.shape[1]
    limit = max(np.abs(embedding).max(), np.abs(query).max())
    limit = max(.05, np.ceil(float(limit)*20)/20)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 9, "axes.linewidth": .65, "pdf.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "nabla-flow-byte-weights",
    })
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.95), sharex=True)
    fig.subplots_adjust(left=.15, right=.865, bottom=.12, top=.93, hspace=.30)
    names = [r"$\mathbf{(a)}$   Tokentabelle $E$", r"$\mathbf{(b)}$   Query-Matrix $W_Q$, erster Block und erster Kopf"]
    for ax, matrix, name in zip(axes, (embedding, query), names):
        rows = matrix.shape[0]
        colored = ax.pcolormesh(np.arange(width+1)-.5, np.arange(rows+1)-.5, matrix,
                                cmap="RdBu_r", vmin=-limit, vmax=limit, shading="flat",
                                edgecolors="face", linewidth=0, antialiased=False)
        ax.set_ylim(rows-.5, -.5)
        ax.set_yticks(range(rows))
        ax.set_title(name, loc="left", pad=5, fontsize=10)
        ax.tick_params(direction="in", top=True, right=True, length=2.6, width=.6)
    axes[0].set_yticklabels([f"{chr(i)}  ({i})" for i in BYTE_IDS])
    axes[0].set_ylabel("Byte (ID)")
    axes[1].set_ylabel(r"Query-Komponente $i$")
    axes[1].set_xticks(sorted(set([*range(0, width, 4), width-1])))
    axes[1].set_xlabel(r"Zustandskoordinate $j$")
    bar = fig.colorbar(colored, cax=fig.add_axes([.895, .23, .022, .59]))
    bar.ax.set_title(r"$w$", fontsize=10)
    bar.ax.tick_params(direction="in", length=2.6, width=.6)
    bar.formatter = FuncFormatter(lambda x, _: f"{x:g}".replace("-", "−").replace(".", ","))
    bar.update_ticks()
    output.mkdir(parents=True, exist_ok=True)
    base = output / "byte-weights"
    fig.savefig(base.with_suffix(".png"), dpi=400)
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "results/model.nfg")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    options = parser.parse_args()
    draw(options.model, options.out)
