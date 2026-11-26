"""Prepare the pinned Tiny Shakespeare source for GPT-2 training from scratch.

    python tools/prepare_124m.py --source build-qa/tinyshakespeare.txt --out build-gpu/data

Requires tiktoken. Set TIKTOKEN_CACHE_DIR to the existing GPT-2 encoding cache
for offline use. Output uses little-endian uint16 token IDs with EOT per speech.
"""

import argparse
from array import array
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

import tiktoken


REVISION = "370cbcd448eb7daf32f21a6be560b70e0b33c4e3"
SOURCE = f"https://raw.githubusercontent.com/karpathy/char-rnn/{REVISION}/data/tinyshakespeare/input.txt"
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
GIT_BLOB = "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(raw):
    if digest(raw) != SOURCE_SHA256:
        raise ValueError(f"Unexpected source SHA-256 {digest(raw)}; expected {SOURCE_SHA256}")
    text = raw.decode("utf-8")
    # Keep every LF, including the separator after each speech. There are two
    # triple-LF separators in the source, and its last speech ends in one LF.
    speeches = [m.group() for m in re.finditer(r".+?(?:\n{2,}|\Z)", text, re.DOTALL)]
    if "".join(speeches) != text or not all(s.splitlines()[0].endswith(":") for s in speeches):
        raise ValueError("Speech segmentation failed")
    n = len(speeches)
    bounds = (0, 90*n//100, 95*n//100, n)
    codec = tiktoken.get_encoding("gpt2")
    metadata = {
        "source": SOURCE, "source_commit": REVISION, "source_git_blob": GIT_BLOB,
        "source_bytes": len(raw), "source_sha256": digest(raw),
        "source_speeches": n,
        "split_rule": "Contiguous source order by speech count, boundaries floor(0.90*N) and floor(0.95*N).",
        "document_rule": "A blank-line-delimited speech; keep all physical LFs, including its separator, then append EOT.",
        "deduplication": "Compare whole speeches after stripping trailing LFs only. Keep within-split repetitions. Drop later-split copies of earlier-split speeches.",
        "speaker_overlap_rule": "For each retained speech compare its first line (speaker heading, including the colon, exact case) to the set of first lines in retained training speeches. Count each speech once, including repeated speakers.",
        "encoding": codec.name, "tokenizer_version": version("tiktoken"),
        "format": "little-endian uint16", "eot_id": codec.eot_token,
        "sample_prompt": "ROMEO:\n", "sample_prompt_ids": codec.encode_ordinary("ROMEO:\n"),
        "splits": {},
    }
    outputs, earlier, removed = {}, {}, []
    training_speakers = set()
    for i, name in enumerate(("train", "val", "test")):
        start, stop = bounds[i:i+2]
        ids = array("H")
        if ids.itemsize != 2:
            raise RuntimeError("16-bit unsigned short is required")
        local = set()
        count = byte_count = unseen_speaker_speeches = 0
        for index in range(start, stop):
            speech = speeches[index]
            key = speech.rstrip("\n")
            if key in earlier:
                removed.append({"source_speech": index, "split": name, "also_in": earlier[key],
                                "speech_sha256": digest(key.encode())})
                continue
            encoded = codec.encode_ordinary(speech)
            if codec.decode_bytes(encoded) != speech.encode("utf-8"):
                raise ValueError(f"Failed round trip for speech {index}")
            speaker = speech.partition("\n")[0]
            if name == "train":
                training_speakers.add(speaker)
            elif speaker not in training_speakers:
                unseen_speaker_speeches += 1
            ids.extend(encoded)
            ids.append(codec.eot_token)
            local.add(key)
            count += 1
            byte_count += len(speech.encode("utf-8"))
        earlier.update(dict.fromkeys(local, name))
        if sys.byteorder != "little":
            ids.byteswap()
        chunk = ids.tobytes()
        outputs[f"{name}.u16"] = chunk
        metadata["splits"][name] = {
            "source_range_zero_based_half_open": [start, stop],
            "source_speeches": stop-start, "speeches": count, "distinct_speeches": len(local),
            "speeches_with_speaker_unseen_in_training": unseen_speaker_speeches,
            "cross_split_copies_removed": stop-start-count, "utf8_bytes_kept": byte_count,
            "tokens_including_eot": len(ids), "size_bytes": len(chunk), "sha256": digest(chunk),
        }
    metadata["removed_speeches"] = removed
    prompt_ids = metadata["sample_prompt_ids"]
    outputs["prompt.u16"] = struct.pack(f"<{len(prompt_ids)}H", *prompt_ids)
    outputs["meta.json"] = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()
    return outputs, metadata


def save(outputs, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # A repeated preparation may reuse identical files. Reject a different split
    # before publishing any output, so training cannot silently change its data.
    pending = {}
    for name, data in outputs.items():
        path = destination/name
        if path.exists():
            existing = path.read_bytes()
            if existing == data:
                continue
            # Additional descriptions/statistics may refresh the derived manifest
            # when it describes exactly the same pinned source and token files.
            compatible = False
            if name == "meta.json":
                old, new = json.loads(existing), json.loads(data)
                compatible = (old.get("source_sha256") == new["source_sha256"]
                              and old.get("sample_prompt_ids") == new["sample_prompt_ids"]
                              and all(old.get("splits", {}).get(k, {}).get("sha256") == v["sha256"]
                                      for k, v in new["splits"].items()))
            if not compatible:
                raise ValueError(f"Different file already exists: {path}. Choose a new output directory.")
        pending[path] = data
    for path, data in pending.items():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as file:
                temporary = Path(file.name)
                file.write(data)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        outputs, meta = prepare(args.source.read_bytes())
        save(outputs, args.out)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    for name, stats in meta["splits"].items():
        print(f"{name}: {stats['speeches']} speeches, {stats['tokens_including_eot']} tokens, "
              f"{stats['cross_split_copies_removed']} cross-split copies removed")
    print(f"Metadata: {args.out/'meta.json'}")


if __name__ == "__main__":
    main()
