#!/usr/bin/env python3
"""
Helpers for Flash-MoE expert layout manifest v2.
"""

from __future__ import annotations

import json
from pathlib import Path

NUM_LAYERS = 60
NUM_EXPERTS = 512


def identity_orders() -> list[list[int]]:
    return [list(range(NUM_EXPERTS)) for _ in range(NUM_LAYERS)]


def load_layout_manifest(path: Path | str | None) -> dict | None:
    if not path:
        return None
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("version") != 2:
        raise ValueError(f"Unsupported layout manifest version: {data.get('version')}")
    layers = data.get("layers")
    if not isinstance(layers, list) or len(layers) != NUM_LAYERS:
        raise ValueError(f"Layout manifest must contain {NUM_LAYERS} layers")
    return data


def layer_orders_from_manifest(data: dict | None) -> list[list[int]]:
    if not data:
        return identity_orders()
    orders = []
    for layer in data["layers"]:
        if "physical_order" in layer:
            order = list(layer["physical_order"])
        else:
            mapping = list(layer["logical_to_physical"])
            order = [0] * NUM_EXPERTS
            for logical, physical in enumerate(mapping):
                order[physical] = logical
        if len(order) != NUM_EXPERTS:
            raise ValueError("Each manifest layer must map 512 experts")
        if sorted(order) != list(range(NUM_EXPERTS)):
            raise ValueError("Manifest physical order must be a permutation of logical experts")
        orders.append(order)
    return orders


def logical_to_physical_from_orders(orders: list[list[int]]) -> list[list[int]]:
    out = []
    for order in orders:
        mapping = [0] * NUM_EXPERTS
        for physical, logical in enumerate(order):
            mapping[logical] = physical
        out.append(mapping)
    return out
