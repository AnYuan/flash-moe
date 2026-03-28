#!/usr/bin/env python3
"""
export_vocab.py — Export HuggingFace tokenizer.json to the vocab.bin format used by infer.m.

Usage:
    python export_vocab.py [tokenizer.json] [vocab.bin]

Binary format expected by infer.m:
  Header:
    num_entries: uint32  (must cover every token id index)
    max_id: uint32
  Entries:
    For token_id in [0, max_id]:
      byte_len: uint16
      bytes[byte_len]: UTF-8 token string

Missing token ids are written as zero-length entries so token_id can be used as
an array index during decoding.
"""

import json
import os
import struct
import sys


def main():
    tok_path = sys.argv[1] if len(sys.argv) > 1 else (
        '/Users/danielwoods/.cache/huggingface/hub/'
        'models--mlx-community--Qwen3.5-397B-A17B-4bit/'
        'snapshots/39159bd8aa74f5c8446d2b2dc584f62bb51cb0d3/tokenizer.json'
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.bin'

    with open(tok_path, 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)

    vocab = tokenizer['model']['vocab']  # str -> int
    added_tokens = tokenizer.get('added_tokens', [])

    id_to_token = {token_id: token_str for token_str, token_id in vocab.items()}
    for added in added_tokens:
        id_to_token[added['id']] = added['content']

    max_id = max(id_to_token)
    tokens_by_id = [None] * (max_id + 1)

    for token_id, token_str in id_to_token.items():
        tokens_by_id[token_id] = token_str

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', len(tokens_by_id)))
        f.write(struct.pack('<I', max_id))
        for token_str in tokens_by_id:
            if token_str is None:
                f.write(struct.pack('<H', 0))
                continue
            token_bytes = token_str.encode('utf-8')
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)

    present = len(id_to_token)
    missing = len(tokens_by_id) - present
    size = os.path.getsize(out_path)

    print(f"Exported to {out_path}:")
    print(f"  Entries: {len(tokens_by_id)}")
    print(f"  Max token id: {max_id}")
    print(f"  Present tokens: {present}")
    print(f"  Missing token ids: {missing}")
    print(f"  File size: {size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
