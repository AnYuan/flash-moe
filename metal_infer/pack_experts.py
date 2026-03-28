#!/usr/bin/env python3
"""
pack_experts.py — Convert HF safetensors expert tensors into Flash-MoE packed_experts files.

The Hugging Face snapshot stores routed expert tensors per layer as:
  [512, out_dim, packed_cols] U32 for weights
  [512, out_dim, num_groups] BF16 for scales/biases

This repo expects each layer file to store 512 experts back-to-back, and each
expert blob to be laid out as:
  gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b

Usage:
    python pack_experts.py --model <snapshot_dir>
    python pack_experts.py --model <snapshot_dir> --layer 0
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path


NUM_LAYERS = 60
NUM_EXPERTS = 512
DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-397B-A17B-4bit"
    / "snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3"
)

ORDER = [
    ("gate_proj", "weight", "U32", [NUM_EXPERTS, 1024, 512], 2_097_152),
    ("gate_proj", "scales", "BF16", [NUM_EXPERTS, 1024, 64], 131_072),
    ("gate_proj", "biases", "BF16", [NUM_EXPERTS, 1024, 64], 131_072),
    ("up_proj", "weight", "U32", [NUM_EXPERTS, 1024, 512], 2_097_152),
    ("up_proj", "scales", "BF16", [NUM_EXPERTS, 1024, 64], 131_072),
    ("up_proj", "biases", "BF16", [NUM_EXPERTS, 1024, 64], 131_072),
    ("down_proj", "weight", "U32", [NUM_EXPERTS, 4096, 128], 2_097_152),
    ("down_proj", "scales", "BF16", [NUM_EXPERTS, 4096, 16], 131_072),
    ("down_proj", "biases", "BF16", [NUM_EXPERTS, 4096, 16], 131_072),
]
EXPERT_SIZE = sum(item[4] for item in ORDER)
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE


@dataclass(frozen=True)
class TensorSlice:
    filename: str
    start: int
    end: int
    per_expert_bytes: int


def parse_safetensors_header(filepath: Path) -> tuple[dict, int]:
    with filepath.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def build_layer_slices(model_path: Path) -> dict[int, list[TensorSlice]]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    weight_map = json.loads(index_path.read_text())["weight_map"]
    header_cache: dict[str, tuple[dict, int]] = {}
    out: dict[int, list[TensorSlice]] = {}

    for layer in range(NUM_LAYERS):
        slices: list[TensorSlice] = []
        for proj, field, dtype, shape, per_expert_bytes in ORDER:
            tensor_name = (
                f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}.{field}"
            )
            filename = weight_map.get(tensor_name)
            if not filename:
                raise KeyError(f"Missing tensor in weight_map: {tensor_name}")

            if filename not in header_cache:
                header_cache[filename] = parse_safetensors_header(model_path / filename)

            header, data_start = header_cache[filename]
            meta = header.get(tensor_name)
            if not meta:
                raise KeyError(f"Missing tensor in header: {tensor_name}")
            if meta["dtype"] != dtype:
                raise ValueError(
                    f"{tensor_name}: expected dtype {dtype}, got {meta['dtype']}"
                )
            if meta["shape"] != shape:
                raise ValueError(
                    f"{tensor_name}: expected shape {shape}, got {meta['shape']}"
                )

            rel_start, rel_end = meta["data_offsets"]
            total_bytes = rel_end - rel_start
            expected_total = NUM_EXPERTS * per_expert_bytes
            if total_bytes != expected_total:
                raise ValueError(
                    f"{tensor_name}: expected {expected_total} bytes, got {total_bytes}"
                )

            slices.append(
                TensorSlice(
                    filename=filename,
                    start=data_start + rel_start,
                    end=data_start + rel_end,
                    per_expert_bytes=per_expert_bytes,
                )
            )
        out[layer] = slices

    return out


def pack_layer(
    model_path: Path,
    output_dir: Path,
    layer: int,
    layer_slices: list[TensorSlice],
    mmaps: dict[str, mmap.mmap],
) -> None:
    output_path = output_dir / f"layer_{layer:02d}.bin"
    if output_path.exists() and output_path.stat().st_size == LAYER_SIZE:
        print(f"[skip] layer {layer:02d} already packed")
        return

    tmp_path = output_path.with_suffix(".bin.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    t0 = time.time()
    bytes_written = 0

    with tmp_path.open("wb") as out_f:
        for expert_idx in range(NUM_EXPERTS):
            for tensor_slice in layer_slices:
                src = mmaps[tensor_slice.filename]
                start = tensor_slice.start + expert_idx * tensor_slice.per_expert_bytes
                end = start + tensor_slice.per_expert_bytes
                out_f.write(src[start:end])
                bytes_written += tensor_slice.per_expert_bytes

            if (expert_idx + 1) % 64 == 0 or expert_idx == NUM_EXPERTS - 1:
                pct = (expert_idx + 1) * 100.0 / NUM_EXPERTS
                print(
                    f"  layer {layer:02d}: expert {expert_idx + 1:3d}/{NUM_EXPERTS} "
                    f"({pct:5.1f}%)"
                )

    if bytes_written != LAYER_SIZE:
        raise ValueError(
            f"layer {layer:02d}: wrote {bytes_written} bytes, expected {LAYER_SIZE}"
        )

    os.replace(tmp_path, output_path)
    elapsed = time.time() - t0
    throughput = bytes_written / elapsed / 1e9
    print(
        f"[done] layer {layer:02d}: {bytes_written / 1e9:.2f} GB "
        f"in {elapsed:.1f}s ({throughput:.2f} GB/s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack routed expert tensors into Flash-MoE layer_XX.bin files"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="HF snapshot directory containing model.safetensors.index.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: MODEL/packed_experts)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Pack only one layer index",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    output_dir = (args.output or (model_path / "packed_experts")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:  {model_path}")
    print(f"Output: {output_dir}")
    print(f"Expert size: {EXPERT_SIZE:,} bytes")
    print(f"Layer size:  {LAYER_SIZE / 1e9:.2f} GB")

    t0 = time.time()
    layer_map = build_layer_slices(model_path)
    layers = [args.layer] if args.layer is not None else list(range(NUM_LAYERS))
    for layer in layers:
        if layer < 0 or layer >= NUM_LAYERS:
            raise ValueError(f"--layer must be in [0, {NUM_LAYERS - 1}]")

    open_files: dict[str, object] = {}
    mmaps: dict[str, mmap.mmap] = {}
    filenames = sorted({s.filename for layer in layers for s in layer_map[layer]})
    try:
        for filename in filenames:
            f = (model_path / filename).open("rb")
            open_files[filename] = f
            mmaps[filename] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        for layer in layers:
            pack_layer(model_path, output_dir, layer, layer_map[layer], mmaps)
    finally:
        for mm in mmaps.values():
            mm.close()
        for f in open_files.values():
            f.close()

    elapsed = time.time() - t0
    print(f"All requested layers packed in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
