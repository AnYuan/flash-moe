#!/usr/bin/env python3
"""
reorder_packed_experts.py — Reorder packed expert layer files using layout manifest v2.

This tool rewrites existing `packed_experts/` or `packed_experts_2bit/` layer files so
physical slot order matches the manifest's physical order.
"""

from __future__ import annotations

import argparse
import mmap
import os
import time
from pathlib import Path

from layout_manifest import layer_orders_from_manifest, load_layout_manifest

NUM_LAYERS = 60
NUM_EXPERTS = 512
EXPERT_SIZE_4BIT = 7_077_888
EXPERT_SIZE_2BIT = 3_932_160


def expert_size_for_format(fmt: str) -> int:
    if fmt == "4bit":
        return EXPERT_SIZE_4BIT
    if fmt == "2bit":
        return EXPERT_SIZE_2BIT
    raise ValueError(f"Unsupported format: {fmt}")


def reorder_layer(src_path: Path, dst_path: Path, expert_size: int, physical_order: list[int]) -> None:
    expected_size = NUM_EXPERTS * expert_size
    actual_size = src_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{src_path}: expected {expected_size} bytes, got {actual_size}")

    tmp_path = dst_path.with_suffix(".bin.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    t0 = time.time()
    with src_path.open("rb") as src_f, tmp_path.open("wb") as dst_f:
        src_mm = mmap.mmap(src_f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for physical_idx, logical_idx in enumerate(physical_order):
                start = logical_idx * expert_size
                end = start + expert_size
                dst_f.write(src_mm[start:end])
                if (physical_idx + 1) % 64 == 0 or physical_idx == NUM_EXPERTS - 1:
                    pct = (physical_idx + 1) * 100.0 / NUM_EXPERTS
                    print(f"  slot {physical_idx + 1:3d}/{NUM_EXPERTS} ({pct:5.1f}%)")
        finally:
            src_mm.close()

    os.replace(tmp_path, dst_path)
    elapsed = time.time() - t0
    print(f"[done] {dst_path.name}: {expected_size / 1e9:.2f} GB in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder packed expert files using layout manifest v2")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Source directory, e.g. MODEL/packed_experts or MODEL/packed_experts_2bit")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination directory for reordered layer files")
    parser.add_argument("--layout-manifest", type=Path, required=True,
                        help="Layout manifest v2 JSON")
    parser.add_argument("--format", choices=["4bit", "2bit"], required=True,
                        help="Expert blob format")
    parser.add_argument("--layer", type=int, default=None,
                        help="Optional single layer to rewrite")
    args = parser.parse_args()

    manifest = load_layout_manifest(args.layout_manifest)
    layer_orders = layer_orders_from_manifest(manifest)
    expert_size = expert_size_for_format(args.format)

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.layer is not None:
        layers = [args.layer]
    else:
        layers = [i for i in range(NUM_LAYERS) if (input_dir / f"layer_{i:02d}.bin").exists()]
    if not layers:
        raise SystemExit(f"No layer_XX.bin files found in {input_dir}")

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Format:  {args.format} ({expert_size:,} bytes/expert)")
    print(f"Layout:  {args.layout_manifest}")
    print(f"Layers:  {layers}")

    t0 = time.time()
    for layer in layers:
        src_path = input_dir / f"layer_{layer:02d}.bin"
        dst_path = output_dir / f"layer_{layer:02d}.bin"
        print(f"=== Layer {layer:02d} ===")
        reorder_layer(src_path, dst_path, expert_size, layer_orders[layer])
    print(f"All requested layers reordered in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
