"""Plot the recorded Medium run. Default output is a 178 mm PNG, SVG and PDF.

    python tools/plot_medium_memory.py
    python tools/plot_medium_memory.py --board BOARD.csv --cuda CUDA.csv --out figure.pdf
    python tools/plot_medium_memory.py --out results/gpt2-medium-355m/memory.gif
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator


DEFAULT = Path(__file__).resolve().parents[1] / "results" / "gpt2-medium-355m"
GIB = 2**30


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def decimal(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def board_readings(path):
    rows = read_rows(path)
    if not rows or rows[0]["phase"] != "before" or rows[-1]["phase"] != "after":
        raise ValueError("A complete board trace needs before and after measurements")
    time = [float(row["elapsed_s"]) for row in rows]
    used = [int(row["used_mib"])/1024 for row in rows]
    active = [i for i, row in enumerate(rows) if row["phase"] == "run"]
    if not active or any(b < a for a, b in zip(time, time[1:])):
        raise ValueError("Expected chronological samples during the run")
    return time, used, max(active, key=used.__getitem__)


def cuda_readings(path):
    rows = read_rows(path)
    ready = next((row for row in rows if row["phase"] == "buffers_ready"), None)
    if ready is None or rows[0]["phase"] != "context_baseline":
        raise ValueError("Expected the context baseline and buffers_ready in the CUDA trace")
    observed = max(int(row["delta_from_context_bytes"]) for row in rows)/GIB
    reserved = (int(ready["training_buffer_bytes"])+int(ready["arena_bytes"]))/GIB
    if min(observed, reserved) <= 0:
        raise ValueError("Expected positive memory readings")
    return observed, reserved


def paper_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": .65,
                         "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "nabla-flow-medium-memory"})


def style_axis(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True, width=.6)
    ax.tick_params(which="major", length=3.3, pad=3)
    ax.tick_params(which="minor", length=1.8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))


def set_trace_axes(trace, time, used):
    trace.set(xlabel="Zeit seit Messbeginn (s)", ylabel="Belegter Speicher (GiB)",
              xlim=(-.025*time[-1], time[-1]*1.028), ylim=(0, max(used)*1.20))
    trace.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    trace.xaxis.set_minor_locator(AutoMinorLocator())
    trace.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))


def animate_board(board, output):
    """One frame per actual sample, paced at double speed with a 2 s final hold."""
    from PIL import Image

    time, used, peak = board_readings(board)
    fig, ax = plt.subplots(figsize=(6.1, 2.9), dpi=150)
    fig.subplots_adjust(left=.105, right=.977, top=.86, bottom=.19)
    style_axis(ax)
    set_trace_axes(ax, time, used)
    ax.set_title("GPT-2 Medium · gesamte Grafikkarte", loc="left", pad=6)
    stamp = ax.set_title("", loc="right", pad=6, fontsize=8)
    ax.axhline(used[0], color=".50", linewidth=.65, linestyle=(0, (4, 3)))
    ax.annotate(f"vor dem Lauf  {decimal(used[0])} GiB", ((time[0]+time[-1])/2, used[0]),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    curve, = ax.plot([], [], color=".15", linewidth=.9)
    sample, = ax.plot([], [], "o", color=".15", markersize=3.5)
    high, = ax.plot(time[peak], used[peak], "^", color=".15", markerfacecolor="white",
                    markersize=5, markeredgewidth=.8)
    peak_label = ax.annotate(f"Maximum  {decimal(used[peak])} GiB", (time[peak], used[peak]),
                             xytext=(4, 7), textcoords="offset points", fontsize=8)
    frames = []
    try:
        for i, (t, value) in enumerate(zip(time, used)):
            curve.set_data(time[:i+1], used[:i+1])
            sample.set_data([t], [value])
            high.set_visible(i >= peak)
            peak_label.set_visible(i >= peak)
            stamp.set_text(f"{decimal(t)} s   ·   {decimal(value)} GiB")
            fig.canvas.draw()
            frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                     fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
            frames.append(frame.convert("L"))
        # GIF delay units are centiseconds. Round cumulative timestamps to avoid drift.
        centiseconds = [round((t-time[0])*50) for t in time]
        delays = [10*max(1, b-a) for a, b in zip(centiseconds, centiseconds[1:])]
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=delays+[2000], loop=0, optimize=False)
    finally:
        plt.close(fig)


def plot(board, cuda, output):
    paper_style()
    if output is not None and output.suffix.lower() == ".gif":
        animate_board(board, output)
        return
    time, used, peak = board_readings(board)
    cuda_delta, reserved = cuda_readings(cuda)
    quantities = (used[peak]-used[0], cuda_delta, reserved)
    fig, (trace, comparison) = plt.subplots(1, 2, figsize=(178/25.4, 66/25.4),
                                           gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.072, right=.982, top=.891, bottom=.20, wspace=.30)
    for ax in (trace, comparison):
        style_axis(ax)

    trace.set_title("(a) Belegung der Grafikkarte", loc="left", pad=4)
    trace.axhline(used[0], color=".52", linewidth=.65, linestyle=(0, (4, 3)), zorder=1)
    trace.plot(time, used, color="#233c57", linewidth=.75, zorder=2)
    trace.plot(time[0], used[0], marker="s", markersize=3.5, markerfacecolor="white",
               markeredgecolor=".2", markeredgewidth=.7, zorder=3)
    trace.plot(time[peak], used[peak], marker="^", markersize=4.6, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=.8, zorder=3)
    trace.annotate(f"vor dem Lauf  {decimal(used[0])}", ((time[0]+time[-1])/2, used[0]),
                   xytext=(0, 5), textcoords="offset points", fontsize=8, ha="center")
    trace.annotate(f"Maximum  {decimal(used[peak])}", (time[peak], used[peak]),
                   xytext=(4, 6), textcoords="offset points", fontsize=8)
    set_trace_axes(trace, time, used)

    comparison.set_title("(b) Speicherzuwachs und Puffer", loc="left", pad=4)
    for i, (value, symbol) in enumerate(zip(quantities, ("^", "o", "D"))):
        comparison.plot(i, value, marker=symbol, markersize=4.6, markeredgewidth=.8,
                        markerfacecolor="white" if i < 2 else ".25", markeredgecolor=".15")
        comparison.annotate(decimal(value), (i, value), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=8)
    span = max(max(quantities)-min(quantities), .1)
    comparison.set(xlim=(-.44, 2.44), ylim=(min(quantities)-.48*span, max(quantities)+.60*span),
                   ylabel="Speicher (GiB)")
    comparison.set_xticks(range(3), [r"$\Delta$ Karte", r"$\Delta$ CUDA", "Puffersumme"])
    comparison.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    comparison.grid(axis="y", linewidth=.35, color=".84", linestyle=(0, (1, 3)))
    comparison.set_axisbelow(True)

    paths = [output] if output is not None else [DEFAULT/f"memory.{fmt}" for fmt in ("png", "svg", "pdf")]
    for path in paths:
        metadata = {"Date": None} if path.suffix.lower() == ".svg" else None
        if path.suffix.lower() == ".pdf":
            metadata = {"CreationDate": None, "ModDate": None}
        fig.savefig(path, dpi=400, facecolor="white", metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=DEFAULT/"board-memory.csv")
    parser.add_argument("--cuda", type=Path, default=DEFAULT/"memory.csv")
    parser.add_argument("--out", type=Path, help="write only the specified output file")
    args = parser.parse_args()
    plot(args.board, args.cuda, args.out)
