"""Shared plumbing for benchmarks: result files, tables, percentiles, process fan-out."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from switchboard.telemetry import percentile

RESULTS = Path(__file__).resolve().parent / "results"
SIMULATED_NOTE = (
    "Simulated on a virtual clock. Latencies come from configured provider latency "
    "distributions, so they measure the architecture's scheduling, not real providers."
)


def default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 8))


def pmap[T, R](fn: Callable[[T], R], items: Sequence[T], workers: int | None = None) -> list[R]:
    """Map in worker processes, results in input order. One worker runs in-process."""
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunk))


def pct(values: Sequence[float], q: float) -> float | None:
    return percentile(values, q)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}g}" if abs(value) < 1e5 else f"{value:.0f}"
    return str(value)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [[fmt(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def envelope(name: str, params: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": name,
        "simulated": True,
        "note": SIMULATED_NOTE,
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "params": params,
        "results": results,
    }


def write_result(name: str, data: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def cli(
    name: str,
    run: Callable[..., dict[str, Any]],
    render: Callable[[dict[str, Any]], str],
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    parser = argparse.ArgumentParser(prog=f"python -m benchmarks.{name}")
    parser.add_argument("--n", type=int, default=None, help="override the sample size")
    parser.add_argument("--workers", type=int, default=None, help="worker processes")
    parser.add_argument("--out", type=Path, default=None, help="results directory")
    args = parser.parse_args(argv)
    data = run(n=args.n, workers=args.workers)
    path = write_result(name, data, args.out)
    print(render(data))
    print(f"\nwrote {path}")
    return data
