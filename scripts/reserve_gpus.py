"""Hold a modest CUDA allocation on selected GPUs for a bounded duration.

This is a visible, low-compute reservation marker, not an exclusive scheduler
lock. CUDA_VISIBLE_DEVICES must be set before importing torch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-indices", default="4,5,6,7",
                        help="physical nvidia-smi indices, default: 4,5,6,7")
    parser.add_argument("--memory-mib-per-gpu", type=int, default=1024)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        indices = [int(part.strip()) for part in args.gpu_indices.split(",")]
    except ValueError:
        parser.error("--gpu-indices must be comma-separated integers")
    if (not indices or len(indices) != len(set(indices)) or min(indices) < 0
            or args.memory_mib_per_gpu < 256 or args.hours <= 0
            or args.heartbeat_seconds <= 0):
        parser.error("use unique nonnegative GPU indices, >=256 MiB, and positive durations")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, indices))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != len(indices):
        raise RuntimeError(
            f"requested {len(indices)} CUDA devices, but PyTorch sees "
            f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )

    allocations = []
    devices = []
    requested_bytes = args.memory_mib_per_gpu * 1024 * 1024
    for logical_index, physical_index in enumerate(indices):
        properties = torch.cuda.get_device_properties(logical_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_index)
        if requested_bytes + 512 * 1024 * 1024 > free_bytes:
            raise RuntimeError(
                f"GPU {physical_index} has only {free_bytes // (1024 * 1024)} MiB free"
            )
        with torch.cuda.device(logical_index):
            allocation = torch.zeros(args.memory_mib_per_gpu * 1024 * 1024,
                                     dtype=torch.uint8, device=f"cuda:{logical_index}")
            torch.cuda.synchronize(logical_index)
        allocations.append(allocation)
        devices.append({
            "physical_index": physical_index,
            "logical_index": logical_index,
            "name": properties.name,
            "memory_mib": args.memory_mib_per_gpu,
            "total_memory_mib": total_bytes // (1024 * 1024),
        })

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status_path = args.status_file.resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.hours * 3600
    state = {
        "pid": os.getpid(),
        "status": "running",
        "started_utc": utc_now(),
        "gpu_indices": indices,
        "devices": devices,
        "hours_requested": args.hours,
        "memory_mib_per_gpu": args.memory_mib_per_gpu,
        "compute_policy": "one-time zero initialization; idle thereafter",
        "exclusive_lock": False,
        "stop_command": f"kill {os.getpid()}",
    }
    write_status(status_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    try:
        while not stop_requested and time.monotonic() < deadline:
            time.sleep(min(args.heartbeat_seconds, max(0.1, deadline - time.monotonic())))
            state["elapsed_seconds"] = round(time.monotonic() - started, 1)
            state["heartbeat_utc"] = utc_now()
            write_status(status_path, state)
    finally:
        state["status"] = "stopped" if stop_requested else "expired"
        state["stopped_utc"] = utc_now()
        state["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_status(status_path, state)
        allocations.clear()
        for index in range(len(indices)):
            torch.cuda.empty_cache()
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
