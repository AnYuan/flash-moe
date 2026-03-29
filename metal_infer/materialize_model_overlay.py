#!/usr/bin/env python3
"""
materialize_model_overlay.py — Build a runnable model overlay from a base snapshot
and a partially or fully reordered packed expert directory.

The overlay uses symlinks for unchanged files so layout experiments do not need to
duplicate an entire 200+ GB snapshot.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NUM_LAYERS = 60


def reset_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_replace(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a runnable overlay model snapshot")
    parser.add_argument("--base-model", type=Path, required=True,
                        help="Base Hugging Face snapshot directory")
    parser.add_argument("--output-model", type=Path, required=True,
                        help="Overlay directory to create")
    parser.add_argument("--reordered-experts", type=Path, required=True,
                        help="Directory containing reordered layer_XX.bin files")
    parser.add_argument("--format", choices=["4bit", "2bit"], default="4bit",
                        help="Which packed expert subdir to overlay")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking them")
    args = parser.parse_args()

    base_model = args.base_model.expanduser().resolve()
    output_model = args.output_model.expanduser().resolve()
    reordered_dir = args.reordered_experts.expanduser().resolve()
    packed_dir_name = "packed_experts_2bit" if args.format == "2bit" else "packed_experts"
    base_packed_dir = base_model / packed_dir_name
    output_packed_dir = output_model / packed_dir_name

    if not base_model.is_dir():
        raise SystemExit(f"Base model does not exist: {base_model}")
    if not base_packed_dir.is_dir():
        raise SystemExit(f"Base packed expert dir does not exist: {base_packed_dir}")
    if not reordered_dir.is_dir():
        raise SystemExit(f"Reordered expert dir does not exist: {reordered_dir}")

    reset_dir(output_model)
    output_packed_dir.mkdir(parents=True, exist_ok=True)

    for child in base_model.iterdir():
        if child.name in {"packed_experts", "packed_experts_2bit"}:
            continue
        dst = output_model / child.name
        if args.copy:
            if child.is_dir():
                shutil.copytree(child, dst, symlinks=True)
            else:
                shutil.copy2(child, dst)
        else:
            symlink_replace(child, dst)

    for layer in range(NUM_LAYERS):
        name = f"layer_{layer:02d}.bin"
        src = reordered_dir / name
        if not src.exists():
            src = base_packed_dir / name
        if not src.exists():
            continue
        dst = output_packed_dir / name
        if args.copy and src.is_file() and src.parent == base_packed_dir:
            shutil.copy2(src, dst)
        elif args.copy and src.is_file() and src.parent == reordered_dir:
            shutil.copy2(src, dst)
        else:
            symlink_replace(src, dst)

    print(f"Base:      {base_model}")
    print(f"Overlay:   {output_model}")
    print(f"Experts:   {output_packed_dir}")
    print(f"Format:    {args.format}")
    print(f"Strategy:  {'copy' if args.copy else 'symlink'}")
    print(f"Completed overlay with {len(list(output_packed_dir.glob('layer_*.bin')))} layer files")


if __name__ == "__main__":
    main()
