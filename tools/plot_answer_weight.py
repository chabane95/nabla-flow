"""Plot fixed-budget paired-question recall from the two answer-weight runs."""
import argparse
from pathlib import Path

from plot import BLUE, INK, VERMILION, rows, save
import matplotlib.pyplot as plt
from matplotlib import ticker


def read_scores(run):
    """Return three (hits, total) pairs per split and factor; cross-check the total."""
    totals = {}
    steps = set()
    for row in rows(run / "comparison.csv"):
        if row["checkpoint"] != "final":
            continue
        key = (row["split"], int(row["answer_weight"]))
        if key in totals:
            raise ValueError(f"Duplicate final measurement: {key}")
        totals[key] = (int(row["both"]), int(row["allocations"]))
        steps.add(int(row["step"]))
    if len(steps) != 1:
        raise ValueError("The final checkpoints must use the same training budget.")

    pairs = {"val": {}, "test": {}}
    for row in rows(run / "pairs.csv"):
        split, assignment = row["split"], row["assignment"]
        if assignment.startswith("Links: "):
            task = "Farben"
        elif assignment.startswith("Fach A: "):
            task = "Materialien"
        else:
            raise ValueError(f"Unknown assignment: {assignment}")
        scores = (int(row["1x_final"]), int(row["4x_final"]))
        if assignment in pairs[split] or any(score not in (0, 1, 2) for score in scores):
            raise ValueError(f"Invalid or duplicate assignment: {assignment}")
        pairs[split][assignment] = (task, scores)

    result = {}
    for split, assignments in pairs.items():
        for i, factor in enumerate((1, 4)):
            values = []
            for group in ("Gesamt", "Farben", "Materialien"):
                relevant = [scores[i] for task, scores in assignments.values()
                            if group == "Gesamt" or task == group]
                if not relevant:
                    raise ValueError(f"No {group} assignments in {split}")
                values.append((relevant.count(2), len(relevant)))
            if values[0] != totals[split, factor]:
                raise ValueError(f"Paired counts differ from comparison.csv: {split}, {factor}")
            result[split, factor] = values
    return result, steps.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/answer-weight"))
    parser.add_argument("--out", type=Path, help="output directory; defaults to the run directory")
    args = parser.parse_args()
    output = args.out or args.run
    output.mkdir(parents=True, exist_ok=True)
    scores, steps = read_scores(args.run)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "axes.linewidth": .7, "axes.axisbelow": True, "axes.labelsize": 10,
        "xtick.color": INK, "ytick.color": INK, "ytick.direction": "in", "ytick.right": True,
        "pdf.fonttype": 42, "svg.hashsalt": "nabla-flow",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), sharey=True)
    fig.subplots_adjust(left=.1, right=.983, top=.8, bottom=.22, wspace=.13)
    fig.text(.1, .948, f"Beide Gegenfragen richtig nach {steps:,} Updates".replace(",", "."),
             size=13, va="top")
    for ax, split, heading in zip(axes, ("val", "test"),
                                  (r"$\mathbf{(a)}$  Validierung", r"$\mathbf{(b)}$  Test")):
        ax.set_title(heading, loc="left", fontsize=10, pad=5)
        for factor, offset, edge, fill in ((1, -.18, BLUE, "white"), (4, .18, VERMILION, VERMILION)):
            x = [j + offset for j in range(3)]
            counts = scores[split, factor]
            rates = [100*hits/total for hits, total in counts]
            ax.bar(x, rates, width=.31, facecolor=fill, edgecolor=edge, linewidth=1,
                   label=f"Antwortgewicht {factor}×")
            for pos, rate, (hits, total) in zip(x, rates, counts):
                ax.annotate(f"{hits}/{total}", (pos, rate), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8.25)
        ax.set_xlim(-.58, 2.58)
        ax.set_xticks((0, 1, 2), ("Gesamt", "Farben", "Materialien"))
        ax.tick_params(axis="x", length=0, pad=5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(axis="y", color="#d9dad8", linewidth=.45, ls=(0, (1.5, 2.4)))
    axes[0].set_ylabel("Vollständig gelöste Belegungen (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .03), ncol=2,
               frameon=False, handlelength=1.8, columnspacing=2.3)
    save(fig, output / "paired-recall")
    plt.close(fig)


if __name__ == "__main__":
    main()
