#!/usr/bin/env python3
"""
build_layout_manifest.py — Build expert layout manifest v2 from routing logs.

The manifest keeps model semantics unchanged. It only remaps logical expert ids
to physical on-disk slots to improve short-range locality for MoE expert reads.

Input routing log format matches `./infer --collect-routing`:
  int32 layer_idx
  int32 K
  float32[4096] hidden_state
  int32[K] expert_indices

Usage:
    python build_layout_manifest.py --output layout_manifest_v2.json
    python build_layout_manifest.py --routing-log routing.bin --output layout_manifest_v2.json
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

NUM_LAYERS = 60
NUM_EXPERTS = 512
HIDDEN_DIM = 4096
RECENT_WINDOW = 8


def load_routing_log(path: Path):
    data = path.read_bytes()
    offset = 0
    samples = []
    while offset + 8 <= len(data):
        layer_idx, k = struct.unpack_from("<ii", data, offset)
        offset += 8
        offset += HIDDEN_DIM * 4  # skip hidden state payload
        if offset + k * 4 > len(data):
            break
        experts = struct.unpack_from(f"<{k}i", data, offset)
        offset += k * 4
        samples.append((layer_idx, tuple(sorted(set(experts)))))
    return samples


def build_layer_orders(samples, strategy: str):
    freq = [Counter() for _ in range(NUM_LAYERS)]
    pair_counts = [defaultdict(int) for _ in range(NUM_LAYERS)]

    for layer_idx, experts in samples:
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            continue
        freq[layer_idx].update(experts)
        if strategy == "coaccess":
            for i in range(len(experts)):
                for j in range(i + 1, len(experts)):
                    pair_counts[layer_idx][(experts[i], experts[j])] += 1

    orders = []
    for layer_idx in range(NUM_LAYERS):
        remaining = set(range(NUM_EXPERTS))
        placed = []
        if not remaining:
            orders.append([])
            continue

        # Start with the hottest expert to keep physical slot 0 very hot.
        start = max(remaining, key=lambda e: (freq[layer_idx][e], -e))
        placed.append(start)
        remaining.remove(start)

        while remaining:
            recent = placed[-RECENT_WINDOW:]

            def score(expert: int):
                co = 0
                if strategy == "coaccess":
                    for prev in recent:
                        a, b = sorted((expert, prev))
                        co += pair_counts[layer_idx][(a, b)]
                return (co, freq[layer_idx][expert], -expert)

            nxt = max(remaining, key=score)
            placed.append(nxt)
            remaining.remove(nxt)
        orders.append(placed)
    return orders, freq


def invert_orders_to_mapping(orders):
    layers = []
    for layer_idx, order in enumerate(orders):
        logical_to_physical = [0] * NUM_EXPERTS
        for physical, logical in enumerate(order):
            logical_to_physical[logical] = physical
        hot = order[:16]
        layers.append(
            {
                "layer": layer_idx,
                "logical_to_physical": logical_to_physical,
                "physical_order": order,
                "hot_experts": hot,
            }
        )
    return layers


def main():
    parser = argparse.ArgumentParser(description="Build expert layout manifest v2")
    parser.add_argument("--routing-log", type=Path, help="Routing data collected by ./infer --collect-routing")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest path")
    parser.add_argument("--strategy", choices=["identity", "coaccess"], default="coaccess",
                        help="Layout strategy (default: coaccess)")
    parser.add_argument("--model-revision", default="", help="Optional model revision string")
    args = parser.parse_args()

    samples = []
    if args.routing_log:
        samples = load_routing_log(args.routing_log)
        if not samples:
            raise SystemExit(f"Routing log {args.routing_log} is empty or invalid")

    if args.strategy == "identity" or not samples:
        orders = [list(range(NUM_EXPERTS)) for _ in range(NUM_LAYERS)]
        freq = [Counter() for _ in range(NUM_LAYERS)]
        strategy = "identity"
    else:
        orders, freq = build_layer_orders(samples, args.strategy)
        strategy = args.strategy

    layers = invert_orders_to_mapping(orders)
    manifest = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layout_strategy": strategy,
        "routing_log": str(args.routing_log) if args.routing_log else None,
        "model_revision": args.model_revision or "",
        "layers": layers,
        "summary": {
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "samples": len(samples),
            "top_hot_experts_per_layer": [
                [expert for expert, _count in freq[layer_idx].most_common(8)]
                for layer_idx in range(NUM_LAYERS)
            ],
        },
    }

    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote layout manifest v2 to {args.output}")


if __name__ == "__main__":
    main()
