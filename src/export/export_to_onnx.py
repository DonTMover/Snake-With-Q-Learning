#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Candle safetensors weights (DQN) to ONNX for ONNX Runtime / DirectML (NPU).

Assumptions:
- Network: Embedding(vocab, hidden) + Linear(hidden->hidden) + Linear(hidden->hidden) + Linear(hidden->actions)
- Input: state index tensor of shape [B, 1] (int64)
- Output: logits of shape [B, actions]

This script tries to map safetensors parameter keys produced by Candle VarMap with
prefixes: emb, mlp1, mlp2, out. If actual keys differ, the script will print all keys
and try a fuzzy match. You can also use --transpose-linear to handle different weight
layout conventions (Candle Linear may save weight as [in, out]).

Usage (PowerShell):
  # Install deps
  # pip install -r src/export/requirements.txt

  # Export
  # python src/export/export_to_onnx.py --safetensors dqn_agent.safetensors \
  #   --vocab 1024 --hidden 256 --actions 3 --out snake_dqn.onnx
"""

import argparse
import sys
from typing import Dict, List

try:
    import torch
    import torch.nn as nn
    from safetensors.torch import load_file as st_load
except Exception as e:
    print("[error] Missing dependencies. Run: pip install -r src/export/requirements.txt")
    raise


class DqnNet(nn.Module):
    def __init__(self, vocab: int, hidden: int, actions: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.mlp1 = nn.Linear(hidden, hidden)
        self.mlp2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, actions)

    def forward(self, state_idx: torch.Tensor) -> torch.Tensor:
        # state_idx: [B, 1] int64
        if state_idx.dim() == 2 and state_idx.size(1) == 1:
            idx = state_idx.squeeze(1)
        else:
            idx = state_idx
        x = self.emb(idx)
        x = torch.relu(x)
        x = torch.relu(self.mlp1(x))
        x = torch.relu(self.mlp2(x))
        logits = self.out(x)
        return logits


def _find_key(keys: List[str], *parts: str) -> str:
    # 1) exact join with '.'
    dot = ".".join(parts)
    if dot in keys:
        return dot
    # 2) exact join with '/'
    slash = "/".join(parts)
    if slash in keys:
        return slash
    # 3) fuzzy: contains all parts
    for k in keys:
        if all(p in k for p in parts):
            return k
    raise KeyError(f"Could not locate key for: {parts}")


def _assign_linear_weight_bias(layer: nn.Linear, w_t: torch.Tensor, b_t: torch.Tensor, transpose_ok: bool):
    # Torch expects weight [out, in]
    # If incoming is [in, out], transpose
    if w_t.shape != layer.weight.data.shape:
        if transpose_ok and w_t.shape[::-1] == tuple(layer.weight.data.shape):
            w_t = w_t.t().contiguous()
        else:
            raise ValueError(f"Shape mismatch for linear weight: got {tuple(w_t.shape)} vs expected {tuple(layer.weight.data.shape)}")
    layer.weight.data.copy_(w_t)
    if b_t is not None:
        if b_t.shape != layer.bias.data.shape:
            raise ValueError(f"Shape mismatch for linear bias: got {tuple(b_t.shape)} vs expected {tuple(layer.bias.data.shape)}")
        layer.bias.data.copy_(b_t)


def load_from_safetensors(model: DqnNet, st: Dict[str, torch.Tensor], transpose_linear: bool):
    keys = list(st.keys())
    print("[info] safetensors keys:", keys)

    # Embedding
    emb_w_k = _find_key(keys, "emb", "weight")
    model.emb.weight.data.copy_(st[emb_w_k])

    # mlp1
    mlp1_w_k = _find_key(keys, "mlp1", "weight")
    mlp1_b_k = _find_key(keys, "mlp1", "bias")
    _assign_linear_weight_bias(model.mlp1, st[mlp1_w_k], st[mlp1_b_k], transpose_ok=transpose_linear)

    # mlp2
    mlp2_w_k = _find_key(keys, "mlp2", "weight")
    mlp2_b_k = _find_key(keys, "mlp2", "bias")
    _assign_linear_weight_bias(model.mlp2, st[mlp2_w_k], st[mlp2_b_k], transpose_ok=transpose_linear)

    # out
    out_w_k = _find_key(keys, "out", "weight")
    out_b_k = _find_key(keys, "out", "bias")
    _assign_linear_weight_bias(model.out, st[out_w_k], st[out_b_k], transpose_ok=transpose_linear)


def main():
    ap = argparse.ArgumentParser(description="Export DQN safetensors to ONNX")
    ap.add_argument("--safetensors", required=True, help="Path to dqn_agent.safetensors")
    ap.add_argument("--out", required=True, help="Output ONNX path, e.g. snake_dqn.onnx")
    ap.add_argument("--vocab", type=int, default=1024, help="State vocabulary size")
    ap.add_argument("--hidden", type=int, default=256, help="Hidden size")
    ap.add_argument("--actions", type=int, default=3, help="Number of actions")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    ap.add_argument("--transpose-linear", action="store_true", help="Transpose linear weights if necessary")
    ap.add_argument("--no-verify", action="store_true", help="Skip onnxruntime verification")
    args = ap.parse_args()

    # Build model
    model = DqnNet(args.vocab, args.hidden, args.actions).eval()

    # Load safetensors
    st = st_load(args.safetensors)
    load_from_safetensors(model, st, transpose_linear=args.transpose_linear)

    # Dummy input: [1,1] int64
    x = torch.zeros(1, 1, dtype=torch.long)

    # Export with dynamic batch dimension
    dynamic_axes = {"state_idx": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model,
        x,
        args.out,
        input_names=["state_idx"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )
    print(f"[ok] Exported to ONNX: {args.out}")

    if args.no_verify:
        return

    # Optional verification with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])  # DirectML can be tested in-app
        test_idx = np.array([[0]], dtype=np.int64)
        res = sess.run(["logits"], {"state_idx": test_idx})
        print("[ok] ORT inference sample logits shape:", res[0].shape)
    except Exception as e:
        print("[warn] onnxruntime verification failed:", e)


if __name__ == "__main__":
    main()
