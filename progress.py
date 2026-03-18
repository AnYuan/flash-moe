"""
progress.py — Visualize Flash-MoE experiment progress.
Reads results.tsv, generates progress.png focused on the 397B model journey.

Usage:
    pip install pandas matplotlib
    python progress.py
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load both results files
    dfs = []
    for path in ["results.tsv", "metal_infer/results.tsv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, sep="\t")
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        print("No results.tsv found.")
        sys.exit(0)

    df = pd.concat(dfs, ignore_index=True)
    df["tok_sec"] = pd.to_numeric(df["tok_sec"], errors="coerce")
    df["params_B"] = pd.to_numeric(df["params_B"], errors="coerce")
    df["mem_gb"] = pd.to_numeric(df["mem_gb"], errors="coerce")
    df["status"] = df["status"].str.strip().str.lower()

    # Split into 397B (the main story) and other (context)
    is_397b = df["params_B"] >= 300  # 397B model
    df_397b = df[is_397b].copy()
    df_other = df[~is_397b].copy()

    n_total = len(df)
    n_397b = len(df_397b)
    kept_397b = df_397b[df_397b["status"] == "keep"]

    print(f"\n=== Flash-MoE: 397B Model Journey ===")
    print(f"Total experiments: {n_total} ({n_397b} on 397B, {len(df_other)} on smaller models)")

    if len(kept_397b) > 0:
        best = kept_397b.loc[kept_397b["tok_sec"].idxmax()]
        print(f"Best 397B result: {best['tok_sec']:.1f} tok/s — {best['description'][:80]}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot 397B experiments (the main story)
    colors_397b = {"keep": "#2ecc71", "discard": "#e74c3c"}
    for status in ["discard", "keep"]:
        mask = (df_397b["status"] == status) & (df_397b["tok_sec"] > 0)
        if mask.any():
            subset = df_397b[mask]
            ax.scatter(range(len(subset.index)), subset["tok_sec"],
                       c=colors_397b.get(status, "#999"),
                       s=80 if status == "keep" else 30,
                       label=f"397B {status}" if status == "keep" else "397B discarded",
                       zorder=5 if status == "keep" else 3,
                       edgecolors="black" if status == "keep" else "none",
                       linewidths=0.5, alpha=0.9 if status == "keep" else 0.4)

    # Running best line for 397B kept experiments
    kept_nonzero = kept_397b[kept_397b["tok_sec"] > 0].copy()
    if len(kept_nonzero) > 1:
        running_best = kept_nonzero["tok_sec"].cummax()
        x_kept = []
        idx = 0
        for i, row in df_397b.iterrows():
            if row["status"] == "keep" and row["tok_sec"] > 0:
                x_kept.append(idx)
            if row["tok_sec"] > 0:
                idx += 1
        if len(x_kept) == len(running_best):
            ax.step(x_kept, running_best.values,
                    where="post", color="#27ae60", linewidth=2.5, alpha=0.8,
                    label="Running best (397B)")

    # Annotate key milestones
    milestones = [
        ("CPU-only\n0.28 tok/s", 0.28),
        ("GPU matmuls\n1.85 tok/s", 1.85),
        ("Fused pipeline\n5.29 tok/s", 5.29),
        ("2-bit experts\n5.55 tok/s", 5.55),
        ("Trust OS cache\n5.74 tok/s", 5.74),
    ]

    ax.set_ylabel("Tokens/second", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experiment # (397B model only)", fontsize=12)
    ax.set_title("Flash-MoE: Running a 397B Model on a Laptop\n"
                 "From 0.28 tok/s to 5.7 tok/s through 90+ experiments",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=-0.5)

    # Add phase annotations
    ax.axhline(y=5.74, color="#27ae60", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(0.98, 5.74, "  5.74 tok/s (current best)", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=9, color="#27ae60", alpha=0.7)

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved progress.png")


if __name__ == "__main__":
    main()
