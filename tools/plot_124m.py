"""Draw GPT-2 Small training curves from the C trainer's history.csv."""

import argparse
import csv
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from PIL import Image


BLUE, RED = "#2449a0", "#aa3426"
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "results" / "gpt2-124m"


def loss_field(raw, line, name):
    if raw is None:
        raise ValueError(f"Incomplete CSV row {line}")
    if raw.strip().lower() in ("", "nan", "na"):
        return math.nan
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name} on row {line}")
    return value


def load_history(path):
    """Join disjoint reports for a step, leaving unmeasured losses missing."""
    records = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "tokens", "train_nll", "val_nll"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Expected columns step,tokens,train_nll,val_nll")
        for line, row in enumerate(reader, 2):
            step, tokens = int(row["step"]), int(row["tokens"])
            if min(step, tokens) < 0 or records and (step < records[-1][0] or tokens < records[-1][1]):
                raise ValueError(f"Step or token count is negative or decreases on row {line}")
            loss = [loss_field(row[k], line, k) for k in ("train_nll", "val_nll")]
            if records and step == records[-1][0]:
                previous = records[-1]
                if tokens != previous[1]:
                    raise ValueError(f"Conflicting token counts at step {step}")
                for index, value in enumerate(loss, 2):
                    if math.isfinite(value):
                        if math.isfinite(previous[index]) and previous[index] != value:
                            raise ValueError(f"Conflicting loss measurements at step {step}")
                        previous[index] = value
            else:
                records.append([step, tokens, *loss])
    if not records:
        raise ValueError(f"No records in {path}")
    steps, tokens, train, val = zip(*records)
    train, val = np.array(train), np.array(val)
    if not (np.isfinite(train).any() or np.isfinite(val).any()):
        raise ValueError("No finite training or validation losses")
    return np.array(steps), np.array(tokens), train, val


def bounded(values, *, positive_log=False):
    lo, hi = np.min(values), np.max(values)
    if positive_log:
        factor = max(1.08, (hi/lo)**0.08)
        return lo/factor, hi*factor
    pad = max(0.06*(hi-lo), 0.015*hi, 0.015)
    return max(0, lo-pad), hi+pad


def token_axis(last):
    if last >= 1_000_000:
        return 1_000_000, "Training target tokens (millions)"
    if last >= 10_000:
        return 1_000, "Training target tokens (thousands)"
    return 1, "Training target tokens"


def frame_indices(length):
    # Keep the animation bounded for long logs, retaining all observed points
    # up to the indicated frame. The first and last record are always shown.
    return np.unique(np.rint(np.linspace(0, length-1, min(45, length))).astype(int))


def render(run, out):
    steps, tokens, train, val = load_history(run / "history.csv")
    measured = np.isfinite(val)
    with np.errstate(over="raise"):
        perplexity = np.exp(val[measured])
    unit, token_label = token_axis(tokens[-1])
    x_tokens = tokens[measured] / unit
    fig, (first, second) = plt.subplots(1, 2, figsize=(7.45, 3.58))
    fig.subplots_adjust(left=.092, right=.971, top=.89, bottom=.235, wspace=.36)
    for ax, name in ((first, "(a)  Cross-entropy"), (second, "(b)  Validation perplexity")):
        ax.set_title(name, loc="left", fontsize=9.5, pad=5)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,g}"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
    first.set(xlabel="Optimizer step", ylabel="NLL (nats per target token)")
    first.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    first.yaxis.set_major_locator(ticker.MaxNLocator(5))
    first.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    first.set_ylim(*bounded(np.concatenate((train[np.isfinite(train)], val[measured]))))
    first.set_xlim(steps[0]-.035*max(steps[-1]-steps[0], 1),
                   steps[-1]+.035*max(steps[-1]-steps[0], 1))
    tr_line, = first.plot([], [], color=BLUE, lw=.95, marker=".", ms=2.7, label="Training sample")
    va_line, = first.plot([], [], ls="none", color=RED, marker="o", mfc="white", ms=3.8,
                          mew=.8, label="Validation")
    fig.legend(handles=(tr_line, va_line), loc="lower left", bbox_to_anchor=(.069,.001),
               ncol=2, frameon=False, columnspacing=1.25, handletextpad=.4)
    second.set(xlabel=token_label, ylabel=r"Perplexity $\exp(\mathcal{L}_{\mathrm{val}})$")
    second.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=(unit == 1)))
    extent = max((tokens[-1]-tokens[0])/unit, 1/unit)
    second.set_xlim(tokens[0]/unit-.035*extent, tokens[-1]/unit+.035*extent)
    if measured.any():
        use_log = max(perplexity)/min(perplexity) >= 4
        if use_log:
            second.set_yscale("log")
            if max(perplexity)/min(perplexity) >= 100:
                second.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1,)))
                second.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
            else:
                second.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
                second.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
            second.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            second.yaxis.set_major_locator(ticker.MaxNLocator(5))
            second.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        second.set_ylim(*bounded(perplexity, positive_log=use_log))
    else:
        second.set_yticks([])
    ppl_line, = second.plot([], [], ls="none", marker="o", color=RED, mfc="white", ms=4, mew=.85)
    waiting = second.text(.50, .50, "Awaiting validation", ha="center", va="center",
                          color="0.4", fontsize=8, transform=second.transAxes)
    stamp = fig.text(.97, .038, "", ha="right", va="center", fontsize=8)

    def show(index):
        take = slice(0, index+1)
        tr_line.set_data(steps[take], train[take])
        nval = np.count_nonzero(measured[take])
        va_line.set_data(steps[take][measured[take]], val[take][measured[take]])
        ppl_line.set_data(x_tokens[:nval], perplexity[:nval])
        waiting.set_visible(nval == 0)
        stamp.set_text(f"Step {steps[index]:,}    ·    {tokens[index]:,} target tokens")

    out.mkdir(parents=True, exist_ok=True)
    show(len(steps)-1)
    fig.savefig(out / "training.png", dpi=300)
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    with (out / "training.svg").open("w", encoding="utf-8", newline="\n") as dest:
        dest.writelines(line.rstrip()+"\n" for line in svg.getvalue().splitlines())
    frames = []
    fig.set_dpi(145)
    for index in frame_indices(len(steps)):
        show(index)
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(pixels).convert("P", palette=Image.Palette.ADAPTIVE))
    plt.close(fig)
    durations = [310]*len(frames)
    durations[0], durations[-1] = 700, 1600
    frames[0].save(out / "training.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"Plotted {len(steps)} steps and {measured.sum()} validation losses in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="Directory containing history.csv")
    parser.add_argument("--out", type=Path, help="Output directory (defaults to the run directory)")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.linewidth": .75, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "svg.hashsalt": "nabla-flow-124m"})
    try:
        render(args.run, args.out or args.run)
    except (OSError, ValueError, TypeError, FloatingPointError) as exc:
        parser.exit(1, f"Could not plot training history: {exc}\n")


if __name__ == "__main__":
    main()
