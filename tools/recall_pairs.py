"""Score both questions about each lab assignment with nablaflow recall.

    python tools/recall_pairs.py --exe ./nablaflow --model results/model.nfg --data data/val.txt

Accepts full lab records or the prompt/answer TSV emitted by tools/corpus.c.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory


WORD = r"[^\s.]+"
FORMATS = (
    (re.compile(rf"(Fach A: ({WORD})\. Fach B: ({WORD})\.) In (A|B) liegt \t?({WORD}\.)"), ("A", "B")),
    (re.compile(rf"(Links: ({WORD})\. Rechts: ({WORD})\.) (Links|Rechts) ist \t?({WORD}\.)"), ("Links", "Rechts")),
)


def load_pairs(path):
    pairs = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line:
            continue
        for pattern, labels in FORMATS:
            match = pattern.fullmatch(line)
            if match:
                break
        else:
            raise ValueError(f"{path}:{line_no}: expected a lab record")
        assignment, first, second, question, answer = match.groups()
        if first == second or answer != (first, second)[labels.index(question)] + ".":
            raise ValueError(f"{path}:{line_no}: ambiguous assignment or inconsistent answer")
        pair = pairs.setdefault(assignment, {})
        if question in pair:
            raise ValueError(f"{path}:{line_no}: duplicate question for {assignment}")
        prompt = line[:match.start(5)].rstrip("\t")
        pair[question] = f"{prompt}\t{answer}\n"
    if not pairs:
        raise ValueError(f"{path}: empty dataset")
    for assignment, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"{path}: missing second question for {assignment}")
    return pairs


def score_pairs(exe, model, pairs):
    results = []
    with TemporaryDirectory() as folder:
        tsv = Path(folder) / "pair.tsv"
        for assignment, pair in sorted(pairs.items()):
            tsv.write_bytes("".join(pair.values()).encode("utf-8"))
            run = subprocess.run([exe, "recall", model, tsv], check=True, stdout=subprocess.PIPE)
            rows = list(csv.DictReader(run.stdout.decode("ascii").splitlines()))
            if len(rows) != 1 or rows[0].get("records") != "2" or rows[0].get("exact") not in {"0", "1", "2"}:
                raise ValueError(f"Unexpected recall output for {assignment}: {run.stdout!r}")
            results.append((assignment, int(rows[0]["exact"])))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path, help="path to the nablaflow executable")
    parser.add_argument("--model", required=True, type=Path, help="byte-model checkpoint")
    parser.add_argument("--data", required=True, type=Path, help="lab records or recall TSV")
    parser.add_argument("--details", type=Path, help="new CSV with exact count per assignment")
    args = parser.parse_args()
    results = score_pairs(args.exe.resolve(), args.model.resolve(), load_pairs(args.data))
    if args.details:
        with args.details.open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(("assignment", "exact"))
            writer.writerows(results)
    counts = [exact for _, exact in results]
    print(json.dumps(dict(allocations=len(counts), zero=counts.count(0), one=counts.count(1),
                          both=counts.count(2), questions=2 * len(counts), exact=sum(counts))))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
