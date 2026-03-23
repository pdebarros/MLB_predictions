#!/usr/bin/env python3
"""
Unpickle `graph_dataset.pkl` and print the directory contents + basic object summary.

Run (inside your venv):
  python unpickle_graph_dataset.py
or:
  python unpickle_graph_dataset.py --path data/logs/graph_dataset.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Optional, Sequence


def summarize(obj: Any) -> None:
    print(f"Unpickled object type: {type(obj)}")

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"Dict keys: {len(keys)} total")
        print(f"First keys: {keys[:20]}")
        return

    if isinstance(obj, (list, tuple, set)):
        try:
            print(f"Iterable length: {len(obj)}")
        except Exception:
            pass
        return

    # Common “dataset/graph container” attributes (best-effort).
    for attr in ("data", "graphs", "dataset", "items"):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            print(f"Has attribute `{attr}` of type {type(v)}")
            try:
                print(f"`{attr}` length: {len(v)}")
            except Exception:
                pass
            break


def _preview_value(name: str, value: Any, max_items: int = 5) -> None:
    """Best-effort preview for tensors/arrays/scalars without hard deps."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        msg = f"    - {name}: shape={tuple(shape)}"
        if dtype is not None:
            msg += f", dtype={dtype}"
        print(msg)

        # Try to show a small preview for 1D-ish values.
        try:
            if hasattr(value, "reshape"):
                flat = value.reshape(-1)
                # torch.Tensor path
                if hasattr(flat, "detach"):
                    flat = flat.detach()
                if hasattr(flat, "cpu"):
                    flat = flat.cpu()
                preview = flat[:max_items].tolist()
                print(f"      preview: {preview}")
        except Exception:
            pass
        return

    if isinstance(value, (int, float, str, bool)):
        print(f"    - {name}: {value!r}")
        return

    # Fallback for objects/lists/etc.
    print(f"    - {name}: type={type(value)}")


def inspect_first_graph(graphs: Sequence[Any], sample_fields: int = 4) -> None:
    """Inspect the first graph payload in detail."""
    if len(graphs) == 0:
        print("\nFirst-graph inspection skipped: sequence is empty.")
        return

    g0 = graphs[0]
    print("\nDetailed inspection of first graph: graphs[0]")
    print(f"type(graphs[0]): {type(g0)}")

    if hasattr(g0, "node_types"):
        try:
            node_types = list(g0.node_types)
        except Exception:
            node_types = []
        print(f"node_types: {node_types}")
        for node_type in node_types[:3]:
            try:
                store = g0[node_type]
                keys = list(store.keys())
                print(f"  node store [{node_type}] keys: {keys}")
                for key in keys[:sample_fields]:
                    _preview_value(f"{node_type}.{key}", store[key])
            except Exception as e:
                print(f"  failed reading node store [{node_type}]: {e}")

    if hasattr(g0, "edge_types"):
        try:
            edge_types = list(g0.edge_types)
        except Exception:
            edge_types = []
        print(f"edge_types: {edge_types}")
        for edge_type in edge_types[:3]:
            try:
                store = g0[edge_type]
                keys = list(store.keys())
                print(f"  edge store [{edge_type}] keys: {keys}")
                for key in keys[:sample_fields]:
                    _preview_value(f"{edge_type}.{key}", store[key])
            except Exception as e:
                print(f"  failed reading edge store [{edge_type}]: {e}")

    if hasattr(g0, "timestamp"):
        try:
            print(f"timestamp: {getattr(g0, 'timestamp')}")
        except Exception:
            pass


def inspect_graph_key(obj: Any, sample_count: int = 3) -> Optional[Sequence[Any]]:
    """Print targeted details for dict['graphs'] (or fallback dict['graph'])."""
    if not isinstance(obj, dict):
        print("\nGraph inspection skipped: top-level object is not a dict.")
        return None

    selected_key = None
    if "graphs" in obj:
        selected_key = "graphs"
    elif "graph" in obj:
        selected_key = "graph"

    if selected_key is None:
        print("\nGraph inspection: neither `graphs` nor `graph` key found.")
        return None

    graph_value = obj[selected_key]
    print(f"\n`{selected_key}` key inspection")
    print(f"type(obj['{selected_key}']): {type(graph_value)}")

    if isinstance(graph_value, (list, tuple)):
        count = len(graph_value)
        print(f"obj['{selected_key}'] is a sequence with {count} items.")
        if count == 0:
            return graph_value

        n = min(sample_count, count)
        print(f"Sampling first {n} graph entries:")
        for i in range(n):
            item = graph_value[i]
            print(f" - {selected_key}[{i}] type: {type(item)}")
            # Best-effort metadata for PyG HeteroData-like objects.
            if hasattr(item, "node_types"):
                try:
                    print(f"   node_types: {list(item.node_types)}")
                except Exception:
                    pass
            if hasattr(item, "edge_types"):
                try:
                    print(f"   edge_types: {list(item.edge_types)}")
                except Exception:
                    pass
            # Common way to store timestamp metadata.
            if hasattr(item, "timestamp"):
                try:
                    print(f"   timestamp: {getattr(item, 'timestamp')}")
                except Exception:
                    pass
    else:
        print(
            f"obj['{selected_key}'] is not a list/tuple, so it is not a "
            "time-sequence container."
        )

    if "label_to_idx" in obj:
        lti = obj["label_to_idx"]
        print(f"\ntype(obj['label_to_idx']): {type(lti)}")
        if isinstance(lti, dict):
            sample_items = list(lti.items())[:10]
            print(f"label_to_idx sample (up to 10): {sample_items}")

    if isinstance(graph_value, (list, tuple)):
        return graph_value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="data/logs/graph_dataset_rich.pkl",
        help="Path to graph_dataset.pkl (default: data/logs/graph_dataset.pkl)",
    )
    args = parser.parse_args()

    pkl_path = args.path
    if not os.path.isabs(pkl_path):
        pkl_path = os.path.abspath(pkl_path)

    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}", file=sys.stderr)
        return 2

    parent_dir = os.path.dirname(pkl_path)
    print(f"graph_dataset.pkl: {pkl_path}")
    print(f"Listing directory: {parent_dir}")
    try:
        for name in sorted(os.listdir(parent_dir)):
            print(f" - {name}")
    except Exception as e:
        print(f"Failed to list directory: {e}", file=sys.stderr)

    print("\nUnpickling...")
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        print(
            "Unpickling failed due to a missing dependency module.\n"
            "Make sure you're running this script inside the same venv\n"
            "where the pickle's dependencies (e.g. torch_geometric) are installed.\n"
            f"Missing module: {e.name}",
            file=sys.stderr,
        )
        return 1

    summarize(obj)
    graphs = inspect_graph_key(obj)
    if graphs is not None:
        inspect_first_graph(graphs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

