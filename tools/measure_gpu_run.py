"""Sample board-wide NVIDIA framebuffer use around a training command."""

import argparse
import csv
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new output CSV")
    parser.add_argument("--device", default="0", help="nvidia-smi GPU index or UUID")
    parser.add_argument("--interval-ms", type=int, default=100, help="target interval between polls")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- TRAINER [ARG ...]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval_ms < 1:
        parser.error("supply a positive interval and a training command")
    query = ["nvidia-smi", "--id=" + args.device,
             "--query-gpu=timestamp,index,memory.total,memory.reserved,memory.used,memory.free",
             "--format=csv,noheader,nounits"]
    with args.out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["elapsed_s", "phase", "smi_timestamp", "gpu", "total_mib",
                         "reserved_mib", "used_mib", "free_mib"])
        start = time.monotonic()

        def sample(phase):
            text = subprocess.check_output(query, text=True)
            timestamp, device, total, reserved, used, free = next(csv.reader([text.strip()], skipinitialspace=True))
            writer.writerow([f"{time.monotonic()-start:.3f}", phase, timestamp, device,
                             int(total), int(reserved), int(used), int(free)])
            stream.flush()

        sample("before")
        with subprocess.Popen(command) as trainer:
            while trainer.poll() is None:
                next_poll = time.monotonic() + args.interval_ms / 1000
                sample("run")
                try:
                    trainer.wait(timeout=max(0, next_poll - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
            sample("after")
            return trainer.returncode


if __name__ == "__main__":
    raise SystemExit(main())
