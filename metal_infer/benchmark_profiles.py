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


def summarize_metrics_dir(output_dir: Path) -> Path:
    rows: list[dict] = []
    for metrics_path in sorted(output_dir.glob("*.json")):
        data = read_metrics(metrics_path)
        if "ttft_ms" not in data:
            continue
        data["profile"] = data.get("profile", metrics_path.stem.split("_", 1)[0])
        data["kv_mode"] = data.get("kv_mode", data.get("kv_quant", ""))
        rows.append(data)

    summary_path = output_dir / "summary.tsv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    return summary_path


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
    parser.add_argument("--model", type=Path, help="Model snapshot dir")
    parser.add_argument("--weights", type=Path, help="model_weights.bin")
    parser.add_argument("--manifest", type=Path, help="model_weights.json")
    parser.add_argument("--vocab", type=Path, help="vocab.bin")
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
                        help="Write direct infer commands for manual terminal execution")
    parser.add_argument("--summarize-existing", type=Path, default=None,
                        help="Scan an output directory of metrics JSON files and write summary.tsv")
    args = parser.parse_args()

    if args.summarize_existing:
        summary_path = summarize_metrics_dir(args.summarize_existing.expanduser().resolve())
        print(f"Wrote benchmark summary to {summary_path}")
        return

    missing = [
        flag for flag, value in (
            ("--model", args.model),
            ("--weights", args.weights),
            ("--manifest", args.manifest),
            ("--vocab", args.vocab),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {' '.join(missing)}")

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
    command_lines = [
        "# Run these commands manually from your current shell.",
        "# Do not execute this file as a child script on this machine; Metal may fall back to CPU.",
        f"cd {shlex.quote(str(Path.cwd()))}",
    ]

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
            command_lines.append(f"echo '[run] {quoted_cmd}'")
            command_lines.append(f"{quoted_cmd} | tee {shlex.quote(str(log_path))}")

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
                    f"Use --emit-script {output_dir / 'run_benchmarks.txt'} and run the commands manually,\n"
                    f"then call --summarize-existing {output_dir}."
                )

            log_path.write_text(combined_out)
            elapsed = time.time() - t0
            if "ERROR: No Metal device" in combined_out:
                raise SystemExit(
                    f"{run_id} launched without Metal support.\n"
                    "On this machine, direct Python child launches can fall back to CPU.\n"
                    f"Use --emit-script {output_dir / 'run_benchmarks.txt'} and run the commands manually,\n"
                    f"then call --summarize-existing {output_dir}."
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
        command_lines.extend([
            "",
            f"# After all commands finish, summarize with:",
            f"# python3 benchmark_profiles.py --summarize-existing {shlex.quote(str(output_dir))}",
        ])
        script_path.write_text("\n".join(command_lines) + "\n")
        print(f"Wrote benchmark command list to {script_path}")
        return

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})

    print(f"Wrote benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
