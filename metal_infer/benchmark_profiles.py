#!/usr/bin/env python3
"""
benchmark_profiles.py — Run infer benchmark matrix for runtime profiles / KV modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
from pathlib import Path

FIELDNAMES = [
    "profile",
    "kv_mode",
    "quant",
    "k",
    "prompt_tokens",
    "generated_tokens",
    "ttft_ms",
    "tok_per_s",
    "kv_raw_bytes",
    "kv_live_bytes",
    "kv_cold_blocks",
    "kv_cold_tokens",
    "wall_time_s",
]


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def build_infer_command(args, infer_path: Path, model_path: Path, weights_path: Path,
                        manifest_path: Path, vocab_path: Path, layout_manifest_path: Path | None,
                        prompt_tokens_path: Path | None, prompt_text: str,
                        metrics_path: Path, profile: str, kv_mode: str) -> list[str]:
    cmd = [
        str(infer_path),
        "--model", str(model_path),
        "--weights", str(weights_path),
        "--manifest", str(manifest_path),
        "--vocab", str(vocab_path),
        "--tokens", str(args.tokens),
        "--profile", profile,
        "--kv-quant", kv_mode,
        "--kv-hot-window", str(args.hot_window),
        "--kv-block-size", str(args.block_size),
        "--metrics-json", str(metrics_path),
    ]
    if prompt_tokens_path:
        cmd.extend(["--prompt-tokens", str(prompt_tokens_path)])
    else:
        cmd.extend(["--prompt", prompt_text])
    if layout_manifest_path:
        cmd.extend(["--layout-manifest", str(layout_manifest_path)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark infer profile / KV combinations")
    parser.add_argument("--infer", type=Path, default=Path("./infer"), help="Path to infer binary")
    parser.add_argument("--model", type=Path, required=True, help="Model snapshot dir")
    parser.add_argument("--weights", type=Path, required=True, help="model_weights.bin")
    parser.add_argument("--manifest", type=Path, required=True, help="model_weights.json")
    parser.add_argument("--vocab", type=Path, required=True, help="vocab.bin")
    parser.add_argument("--layout-manifest", type=Path, default=None, help="Optional layout manifest v2")
    parser.add_argument("--profiles", nargs="+", default=["speed", "balanced", "quality"])
    parser.add_argument("--kv-modes", nargs="+", default=["off", "baseline"])
    parser.add_argument("--tokens", type=int, default=16)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None,
                              help="Text prompt passed to ./infer --prompt")
    prompt_group.add_argument("--prompt-tokens", type=Path, default=None,
                              help="Binary prompt_tokens file passed to ./infer --prompt-tokens")
    parser.add_argument("--output-dir", type=Path, default=Path("./bench_profiles"))
    parser.add_argument("--hot-window", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=0,
                        help="Optional timeout per infer run (0 disables timeout)")
    parser.add_argument("--emit-script", type=Path, default=None,
                        help="Write a zsh benchmark script instead of launching infer from Python")
    args = parser.parse_args()

    infer_path = args.infer.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    vocab_path = args.vocab.expanduser().resolve()
    layout_manifest_path = args.layout_manifest.expanduser().resolve() if args.layout_manifest else None
    prompt_tokens_path = args.prompt_tokens.expanduser().resolve() if args.prompt_tokens else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.tsv"
    rows: list[dict] = []
    prompt_text = args.prompt or "Explain why expert locality matters for MoE inference."
    script_lines = ["#!/bin/zsh", "set -euo pipefail", f"cd {shlex.quote(str(Path.cwd()))}"]
    emitted_runs: list[tuple[str, str, Path]] = []

    for profile in args.profiles:
        for kv_mode in args.kv_modes:
            run_id = f"{profile}_{kv_mode}"
            metrics_path = output_dir / f"{run_id}.json"
            cmd = build_infer_command(
                args, infer_path, model_path, weights_path, manifest_path, vocab_path,
                layout_manifest_path, prompt_tokens_path, prompt_text, metrics_path, profile, kv_mode
            )
            quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
            log_path = output_dir / f"{run_id}.log"
            script_lines.append(f"echo '[run] {quoted_cmd}'")
            script_lines.append(f"{quoted_cmd} |& tee {shlex.quote(str(log_path))}")
            emitted_runs.append((profile, kv_mode, metrics_path))

            if args.emit_script:
                continue

            print(f"[run] {quoted_cmd}")
            t0 = time.time()
            timeout = args.timeout_s if args.timeout_s > 0 else None
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, timeout=timeout)
                combined_out = proc.stdout
            except subprocess.TimeoutExpired as exc:
                combined_out = exc.stdout or ""
                if isinstance(combined_out, bytes):
                    combined_out = combined_out.decode("utf-8", "replace")
                log_path.write_text(combined_out)
                raise SystemExit(
                    f"{run_id} timed out after {args.timeout_s:.1f}s.\n"
                    "On this machine, direct Python child launches can fall back to CPU.\n"
                    f"Use --emit-script {output_dir / 'run_benchmarks.zsh'} for a shell-driven Metal sweep."
                )

            log_path.write_text(combined_out)
            elapsed = time.time() - t0
            if "ERROR: No Metal device" in combined_out:
                raise SystemExit(
                    f"{run_id} launched without Metal support.\n"
                    "On this machine, direct Python child launches can fall back to CPU.\n"
                    f"Use --emit-script {output_dir / 'run_benchmarks.zsh'} for a shell-driven Metal sweep."
                )
            if proc.returncode != 0:
                raise SystemExit(f"{run_id} failed with code {proc.returncode}\n{combined_out}")

            metrics = read_metrics(metrics_path)
            metrics["profile"] = profile
            metrics["kv_mode"] = kv_mode
            metrics["wall_time_s"] = elapsed
            rows.append(metrics)

    if args.emit_script:
        script_path = args.emit_script.expanduser().resolve()
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_lines.extend([
            "python3 - <<'PY'",
            "import csv, json",
            f"summary_path = {str(summary_path)!r}",
            "rows = []",
        ])
        for profile, kv_mode, metrics_path in emitted_runs:
            script_lines.extend([
                f"with open({str(metrics_path)!r}) as f:",
                "    data = json.load(f)",
                f"data['profile'] = {profile!r}",
                f"data['kv_mode'] = {kv_mode!r}",
                "rows.append(data)",
            ])
        script_lines.extend([
            f"fieldnames = {FIELDNAMES!r}",
            "with open(summary_path, 'w', newline='') as f:",
            "    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\\t')",
            "    writer.writeheader()",
            "    for row in rows:",
            "        writer.writerow({key: row.get(key, '') for key in fieldnames})",
            "print(f'Wrote benchmark summary to {summary_path}')",
            "PY",
        ])
        script_path.write_text("\n".join(script_lines) + "\n")
        script_path.chmod(0o755)
        print(f"Wrote benchmark script to {script_path}")
        return

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})

    print(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
