"""Draw the Medium run's board-memory samples and explicit CUDA allocations."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DEFAULT = Path(__file__).resolve().parents[1] / "results" / "gpt2-medium-355m"
GIB = 2**30
INK, BLUE, GRAY = "#152d41", "#176391", "#536777"


def number(value):
    return f"{value:.2f}".replace(".", ",")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def plot(board, cuda, output):
    rows = read_csv(board)
    if not rows or rows[0]["phase"] != "before" or rows[-1]["phase"] != "after":
        raise ValueError("A complete board trace needs before and after measurements")
    time = [float(row["elapsed_s"]) for row in rows]
    used = [int(row["used_mib"]) / 1024 for row in rows]
    active = [i for i, row in enumerate(rows) if row["phase"] == "run"]
    if not active or any(b < a for a, b in zip(time, time[1:])):
        raise ValueError("Expected chronological samples during the run")
    peak = max(active, key=used.__getitem__)
    capacity = int(rows[0]["total_mib"]) / 1024

    ready = next((row for row in read_csv(cuda) if row["phase"] == "buffers_ready"), None)
    if ready is None:
        raise ValueError("Missing buffers_ready in the CUDA memory trace")
    training = int(ready["training_buffer_bytes"]) / GIB
    arena = int(ready["arena_bytes"]) / GIB
    if min(training, arena) <= 0:
        raise ValueError("Expected both FP32 training state and a working arena")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY})
    fig, (trace, buffers) = plt.subplots(2, 1, figsize=(10.8, 6.8),
                                        gridspec_kw={"height_ratios": (4, 1)})
    fig.subplots_adjust(left=.105, right=.97, bottom=.12, top=.80, hspace=.61)
    fig.text(.105, .95, "GPU-Speicher beim Training von GPT-2 Medium",
             fontsize=23, weight="bold", va="top")
    fig.text(.105, .885, "RTX 4070 SUPER  ·  4 Fenster je Batch  ·  128 Tokens je Fenster",
             color=GRAY, fontsize=12)
    trace.set_title("Belegung der gesamten Grafikkarte", loc="left", weight="bold", fontsize=13)
    trace.plot(time, used, color=BLUE, linewidth=1.6, marker=".", markersize=2, zorder=3)
    trace.axhline(capacity, lw=1, ls=":", color=GRAY)
    trace.annotate(f"Kapazität  {number(capacity)} GiB", (time[-1], capacity),
                   xytext=(0, -5), textcoords="offset points", ha="right", va="top", color=GRAY)
    trace.axhline(used[0], lw=.9, ls="--", color=GRAY)
    trace.annotate(f"Vor dem Start  {number(used[0])} GiB", (time[0], used[0]),
                   xytext=(8, 8), textcoords="offset points", color=GRAY)
    trace.plot(time[peak], used[peak], marker="o", ms=6, color=BLUE, zorder=4)
    trace.annotate(f"Beobachtetes Maximum  {number(used[peak])} GiB",
                   (time[peak], used[peak]), xytext=(0, 15), textcoords="offset points",
                   ha="center", weight="bold", color=BLUE)
    trace.set(xlim=(0, time[-1] * 1.025), ylim=(0, math.ceil(capacity + .4)),
              xlabel="Zeit seit Beginn der Aufzeichnung (s)", ylabel="Belegter Speicher (GiB)")
    trace.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[2, 5, 10], nbins=8))
    trace.grid(axis="y", color="#dfe5e9", linewidth=.65)
    trace.set_axisbelow(True)
    trace.spines[["top", "right"]].set_visible(False)
    trace.spines[["bottom", "left"]].set_color("#b0bec7")

    buffers.set_title("Vom Trainer reservierte Puffer", loc="left", weight="bold", fontsize=13)
    buffers.set_title(f"{number(training+arena)} GiB zusammen", loc="right", color=GRAY, fontsize=12)
    sizes = [training/4] * 4 + [arena]
    labels = ["Gewichte", "Gradienten", "Adam-Moment 1", "Adam-Moment 2", "Arbeitsbereich"]
    left = 0.0
    for i, (width, label) in enumerate(zip(sizes, labels)):
        buffers.barh(0, width, left=left, height=.9, color=BLUE if i < 4 else "#246655",
                     edgecolor="white", linewidth=1.5)
        buffers.text(left+width/2, 0, f"{label}\n{number(width)} GiB", ha="center",
                     va="center", fontsize=10.5, color="white", linespacing=1.4)
        left += width
    buffers.set(xlim=(0, left), ylim=(-.52, .52))
    buffers.set_axis_off()
    fig.text(.105, .052, f"Die Kurve zeigt {len(rows)} Messwerte von nvidia-smi einschließlich Desktop und anderer Prozesse.",
             fontsize=10, color=GRAY)
    fig.text(.105, .025, "Die Puffergrößen stammen aus den beiden CUDA-Allokationen in memory.csv. Ein GiB entspricht 2³⁰ Byte.",
             fontsize=10, color=GRAY)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT/"board-memory.csv")
    parser.add_argument("--cuda", type=Path, default=DEFAULT/"memory.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT/"memory.png")
    args = parser.parse_args()
    plot(args.board, args.cuda, args.out)
