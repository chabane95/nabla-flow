"""Optional GPT-2 BPE file conversion. All model computation lives in C."""
import argparse
import array
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="One UTF-8 document per line, append EOT")
    encode.add_argument("text", type=Path)
    encode.add_argument("shard", type=Path)
    prompt = commands.add_parser("prompt", help="Print the token IDs of an exact prompt")
    prompt.add_argument("text")
    prompt.add_argument("--out", type=Path, help="Also write the exact IDs as a new uint16-LE file (no EOT)")
    commands.add_parser("decode", help="Decode space-separated decimal IDs from stdin")
    args = parser.parse_args()
    import tiktoken
    codec = tiktoken.get_encoding("gpt2")
    if args.command == "encode":
        if args.text.resolve() == args.shard.resolve():
            parser.error("Use distinct input and output paths")
        count = 0
        with args.text.open(encoding="utf-8") as source, args.shard.open("xb") as dest:
            for line in source:
                text = line.rstrip("\r\n")
                if not text:
                    continue
                ids = array.array("H", [*codec.encode_ordinary(text), codec.eot_token])
                if ids.itemsize != 2:
                    raise RuntimeError("16-bit unsigned shorts required")
                if sys.byteorder != "little":
                    ids.byteswap()
                dest.write(ids.tobytes())
                count += len(ids)
        print(f"{count} tokens")
    elif args.command == "prompt":
        ids = codec.encode_ordinary(args.text)
        if args.out:
            import struct
            with args.out.open("xb") as stream:
                stream.write(struct.pack(f"<{len(ids)}H", *ids))
        print(*ids)
    else:
        ids = [int(s) for s in sys.stdin.read().split()]
        if any(i < 0 or i >= codec.n_vocab for i in ids):
            parser.error("Token ID outside GPT-2 vocabulary")
        print(codec.decode(ids))


if __name__ == "__main__":
    main()
