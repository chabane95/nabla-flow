"""Decode every recorded continuation and find its longest exact training match.

    python tools/report_124m.py --run RUN --data build-gpu/data --steps 1200 --out samples-report.json

Requires tiktoken, with the same GPT-2 encoding cache used by prepare_124m.py.
Input samples.csv holds NEW token IDs only. The prompt is never stripped by
string matching: a model can genuinely generate words that repeat its prompt.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import tiktoken


def csv_sha256(path):
    # C text streams use CRLF on Windows. Git stores the published CSVs with LF.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_steps(path, field):
    data = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"step", field}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected step and {field} in {path.name}")
        for row in reader:
            step = int(row["step"])
            if step < 0 or step in data or row[field] is None:
                raise ValueError(f"Duplicate step or malformed row at {path.name}:{reader.line_num}")
            data[step] = row[field]
    if 0 not in data:
        raise ValueError(f"Missing step zero in {path.name}")
    return data


class TrainingText:
    def __init__(self, raw, stats, eot):
        if len(raw) != 2 * stats["tokens_including_eot"] or hashlib.sha256(raw).hexdigest() != stats["sha256"]:
            raise ValueError("train.u16 does not match the recorded training split")
        # One Unicode code point per token, independent of characters in the
        # original text. Native str.find thus finds exact *token* substrings.
        ids = [token for token, in struct.iter_unpack("<H", raw)]
        if any(token > eot for token in ids) or not ids or ids[-1] != eot:
            raise ValueError("Invalid training tokens or missing final EOT")
        self.text = "".join(map(chr, ids))
        self.ends = [p for p, token in enumerate(ids) if token == eot]
        if len(self.ends) != stats["speeches"]:
            raise ValueError("Training document count disagrees with meta.json")

    def longest_match(self, generated):
        sequence = "".join(map(chr, generated))

        def locate(length):
            for start in range(len(sequence) - length + 1):
                pos = self.text.find(sequence[start:start+length])
                if pos >= 0:
                    return start, pos
            return None

        # Existence of a matching length k implies existence for every length
        # below k. EOT is absent from sequence and separates all training docs.
        low, high = 0, len(sequence)
        while low < high:
            mid = (low + high + 1) // 2
            if locate(mid) is None:
                high = mid - 1
            else:
                low = mid
        result = {"tokens": low, "generated_start": None, "train_document": None,
                  "train_token_start": None}
        if low:
            start, position = locate(low)
            doc = bisect.bisect_left(self.ends, position)
            doc_start = self.ends[doc-1] + 1 if doc else 0
            result.update(generated_start=start, train_document=doc,
                          train_token_start=position-doc_start)
        return result


def sample_details(codec, corpus, step, cell, validation):
    ids = [int(s) for s in cell.split()]
    eot = codec.eot_token
    if any(i < 0 or i >= codec.n_vocab for i in ids):
        raise ValueError(f"Out-of-range token ID at step {step}")
    if eot in ids[:-1]:
        raise ValueError(f"Found tokens after EOT at step {step}")
    ended = bool(ids and ids[-1] == eot)
    words = ids[:-1] if ended else ids
    raw = codec.decode_bytes(words)
    overlap = corpus.longest_match(words)
    start, count = overlap["generated_start"], overlap["tokens"]
    overlap["text"] = codec.decode(words[start:start+count], errors="replace") if count else ""
    record = {"step": step, "val_nll": validation, "generated_ids": ids,
              "length_before_eot": len(words), "ended_with_eot": ended,
              "text": raw.decode("utf-8", errors="replace"), "utf8_valid": True,
              "longest_training_match": overlap}
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        record.update(utf8_valid=False, original_bytes_hex=raw.hex())
    return record


def report(run, data, budget):
    history = {step: float(value) for step, value in read_steps(run/"history.csv", "val_nll").items()}
    samples = read_steps(run/"samples.csv", "generated_ids")
    if set(history) != set(samples):
        raise ValueError("History and samples must contain exactly the same evaluation steps")
    if any(not math.isfinite(v) or v < 0 for v in history.values()):
        raise ValueError("Invalid validation NLL")
    steps = sorted(history)
    if budget != steps[-1] or budget < 1:
        raise ValueError(f"Expected final step {budget}, last recorded step is {steps[-1]}")
    meta = json.loads((data/"meta.json").read_text(encoding="utf-8"))
    codec = tiktoken.get_encoding("gpt2")
    if meta["encoding"] != codec.name or meta["eot_id"] != codec.eot_token:
        raise ValueError("Expected GPT-2 BPE metadata")
    prompt_ids = meta["sample_prompt_ids"]
    if codec.encode_ordinary(meta["sample_prompt"]) != prompt_ids:
        raise ValueError("Prompt text and prompt IDs disagree")
    if (data/"prompt.u16").read_bytes() != struct.pack(f"<{len(prompt_ids)}H", *prompt_ids):
        raise ValueError("Prompt file and metadata disagree")
    corpus = TrainingText((data/"train.u16").read_bytes(), meta["splits"]["train"], codec.eot_token)
    # Integer distance avoids rounding the target quarter/half for odd budgets.
    quarter = min(steps, key=lambda s: (abs(4*s-budget), s))
    half = min(steps, key=lambda s: (abs(2*s-budget), s))
    best = min(steps, key=lambda s: (history[s], s))
    return {
        "training_budget_steps": budget, "source_sha256": meta["source_sha256"],
        "train_u16_sha256": meta["splits"]["train"]["sha256"],
        "prompt": meta["sample_prompt"], "prompt_ids": prompt_ids,
        "sampling": {"top_k": 40, "temperature": 0.8, "seed_per_checkpoint": 42,
                     "max_new_tokens": 64, "generator": "nfgpu train"},
        "csv_hash_policy": "SHA-256 of the CSV bytes with CRLF normalized to LF; all other bytes are preserved.",
        "history_csv_sha256": csv_sha256(run/"history.csv"),
        "samples_csv_sha256": csv_sha256(run/"samples.csv"),
        "selection_policy": "All recorded samples are included. Milestones use nearest evaluated step, ties earlier. Best is the first minimum recorded validation NLL. Milestones may coincide.",
        "selected_steps": {"initial": 0, "quarter_budget": quarter, "half_budget": half,
                           "best_validation": best, "final": steps[-1]},
        "text_policy": "CSV IDs are generated-only. Raw IDs retain a final EOT (50256); displayed text and overlap exclude it. Decode GPT-2 token bytes as UTF-8, replacing invalid sequences with U+FFFD, with raw hex retained when needed.",
        "match_policy": "Longest exact contiguous sequence of generated token IDs inside any single training speech. Coordinates are zero-based. Ties use the earliest generated start, then the earliest training occurrence. Prompt and EOT are excluded.",
        "samples": [sample_details(codec, corpus, step, samples[step], history[step]) for step in steps],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="Original training budget")
    parser.add_argument("--out", type=Path, required=True, help="Write the full UTF-8 JSON report")
    args = parser.parse_args()
    try:
        result = report(args.run, args.data, args.steps)
        serialized = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        if args.out.exists() and args.out.read_bytes() != serialized:
            raise ValueError(f"Choose a new report path: {args.out}")
        if not args.out.exists():
            with args.out.open("xb") as file:
                file.write(serialized)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Report failed: {error}\n")
    print(f"{len(result['samples'])} samples; selected steps {result['selected_steps']}")


if __name__ == "__main__":
    main()
