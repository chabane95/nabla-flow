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
    # A two-column journal figure at 178 mm, with common baselines and type.
    fig, (first, second) = plt.subplots(1, 2, figsize=(178/25.4, 72/25.4))
    fig.subplots_adjust(left=.087, right=.98, top=.904, bottom=.167, wspace=.31)
    for ax, name in ((first, "(a)   Cross-entropy"), (second, "(b)   Validation perplexity")):
        ax.set_title(name, loc="left", fontsize=10, pad=4)
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
    train_style = dict(color=BLUE, lw=.85, marker=".", ms=3.4)
    val_style = dict(color=RED, ls="none", marker="o", mfc="white", ms=3.8, mew=.75)
    tr_line, = first.plot([], [], **train_style, label="Training sample")
    va_line, = first.plot([], [], **val_style, label="Validation")
    first.legend(loc="upper right", ncol=2, frameon=False, columnspacing=.9,
                 borderaxespad=.55, handlelength=1.5, handletextpad=.35)

    # Magnify the late gap between the losses without changing the main scale.
    # Keep short or sparse histories on the two main axes.
    late = steps >= .5*(float(steps[0])+steps[-1])
    inset = None
    if np.count_nonzero(late & np.isfinite(train) & measured) >= 6:
        inset = first.inset_axes([.47, .46, .47, .31])
        late_train, = inset.plot([], [], **{**train_style, "ms": 2.6})
        late_val, = inset.plot([], [], **{**val_style, "ms": 2.8})
        inset.set_title("Second half", fontsize=7.5, pad=2)
        late_loss = np.concatenate((train[late & np.isfinite(train)], val[late & measured]))
        inset.set_ylim(*bounded(late_loss))
        late_steps = steps[late]
        xpad = .04*np.ptp(late_steps)
        inset.set_xlim(late_steps[0]-xpad, late_steps[-1]+xpad)
        inset.set_xticks(late_steps[[0, len(late_steps)//2, -1]])
        inset.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,g}"))
        inset.yaxis.set_major_locator(ticker.MaxNLocator(3))
        inset.tick_params(labelsize=6.7, length=2.2, pad=1.8)
        for spine in inset.spines.values():
            spine.set_linewidth(.5)
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
                second.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
            else:
                second.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
                second.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
                second.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=(3, 4, 6, 7, 8, 9)))
            second.yaxis.set_minor_formatter(ticker.NullFormatter())
        else:
            second.yaxis.set_major_locator(ticker.MaxNLocator(5))
            second.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        second.set_ylim(*bounded(perplexity, positive_log=use_log))
    else:
        second.set_yticks([])
    ppl_line, = second.plot([], [], **val_style)
    waiting = second.text(.50, .50, "Awaiting validation", ha="center", va="center",
                          color="0.4", fontsize=8, transform=second.transAxes)
    stamp = second.text(.96, .95, "", transform=second.transAxes, ha="right", va="top",
                        fontsize=8, linespacing=1.3)

    def show(index):
        take = slice(0, index+1)
        tr_line.set_data(steps[take], train[take])
        nval = np.count_nonzero(measured[take])
        va_line.set_data(steps[take][measured[take]], val[take][measured[take]])
        ppl_line.set_data(x_tokens[:nval], perplexity[:nval])
        waiting.set_visible(nval == 0)
        stamp.set_text(f"Step {steps[index]:,}\n{tokens[index]:,} target tokens")
        if inset is not None:
            visible = late[take]
            late_train.set_data(steps[take][visible], train[take][visible])
            late_val.set_data(steps[take][visible], val[take][visible])
            inset.set_visible(visible.any())

    out.mkdir(parents=True, exist_ok=True)
    show(len(steps)-1)
    fig.savefig(out / "training.png", dpi=400)
    fig.savefig(out / "training.pdf", metadata={"CreationDate": None, "ModDate": None})
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    with (out / "training.svg").open("w", encoding="utf-8", newline="\n") as dest:
        dest.writelines(line.rstrip()+"\n" for line in svg.getvalue().splitlines())
    frames = []
    fig.set_dpi(160)
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
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
                         "pdf.fonttype": 42, "svg.hashsalt": "nabla-flow-124m",
                         "axes.linewidth": .65, "axes.labelpad": 3, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "xtick.major.pad": 3, "ytick.major.pad": 3,
                         "xtick.major.size": 3.2, "ytick.major.size": 3.2,
                         "xtick.minor.size": 1.7, "ytick.minor.size": 1.7})
    try:
        render(args.run, args.out or args.run)
    except (OSError, ValueError, TypeError, FloatingPointError) as exc:
        parser.exit(1, f"Could not plot training history: {exc}\n")


if __name__ == "__main__":
    main()
