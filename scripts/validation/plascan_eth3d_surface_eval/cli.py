"""Command-line entry point for ETH3D scan-to-mesh evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from .evaluation import evaluate_surface


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PlaScan mesh against aligned ETH3D laser scans."
    )
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scan-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-voxel-size-m", type=float, default=0.02)
    parser.add_argument("--mesh-sample-count", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _write_new_json(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate_surface(
        args.mesh,
        args.scan_alignment,
        voxel_size_m=args.scan_voxel_size_m,
        mesh_sample_count=args.mesh_sample_count,
        seed=args.seed,
    )
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
