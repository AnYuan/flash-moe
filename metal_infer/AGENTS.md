# Repository Guidelines

## Project Structure & Module Organization
This repository is intentionally flat. Core runtime code lives at the top level: `main.m` plus `shaders.metal` implement the Metal benchmark and single-layer/full-MoE paths, `infer.m` contains the end-to-end inference server and CLI, and `chat.m` is the terminal chat client built on `linenoise.c`/`linenoise.h`. Python helpers such as `extract_weights.py`, `export_tokenizer.py`, `repack_experts_2bit.py`, and `train_predictor.py` prepare model assets or analyze routing data. Experimental storage/compression probes live in `test_lzfse.c`, `repack_experts_lz4.c`, and `test_apfs_compress.sh`. Keep generated binaries and model artifacts out of Git.

## Build, Test, and Development Commands
Use `make` to build `metal_infer` and `infer`. Use `make chat` for the TUI client and `make clean` to remove local build outputs. Validation is command-driven:

- `make verify` checks Metal output against the CPU reference.
- `make bench` or `make fullbench` records performance for the benchmark engine.
- `make infer-run` runs a short text generation smoke test.
- `./infer --serve 8000` starts the OpenAI-compatible server; pair it with `./chat --port 8000`.
- `python3 extract_weights.py --model <snapshot> --output .` generates `model_weights.bin` and `model_weights.json`.

## Coding Style & Naming Conventions
Match the existing C/Objective-C style: 4-space indentation, K&R braces, `ALL_CAPS` macros for model constants, and `snake_case` for helper functions. Keep performance-sensitive comments factual and close to the code they justify. Python scripts should stay standard-library-first, use `argparse`, and prefer `Path`/`snake_case`. There is no formatter config in-tree, so keep `clang -Wall -Wextra` clean and avoid large stylistic rewrites in performance patches.

## Testing Guidelines
There is no formal unit-test suite yet; contributors are expected to run targeted checks for the path they changed. Kernel or math changes should pass `make verify` and at least one benchmark command. Server or chat changes should be smoke-tested with `./infer --serve <port>` and `./chat`. Data-layout or compression work should include the relevant standalone tool, for example `clang -O2 -o test_lzfse test_lzfse.c -lcompression`.

## Commit & Pull Request Guidelines
Recent history mixes concise conventional prefixes (`feat:`, `docs:`) with direct performance summaries. Keep subjects short, imperative, and scoped, for example `feat: add 2-bit expert repacker` or `bench: document fullbench regression`. PRs should state the Apple Silicon hardware used, the model snapshot or asset path assumptions, exact commands run, and before/after latency or throughput when behavior affects performance. Do not introduce new hard-coded local paths without also adding a flag or documented override.
