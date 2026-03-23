"""
Train temporal GATv2 + Transformer + MLP from:
  - context-only graphs (no labels in graph edges)
  - CSV supervision rows (one row per PA)

Per epoch training:
  For each time step t in chronological order, train on all PAs at t.
"""
from __future__ import annotations

import argparse
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Deque

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv


REL = ("pitcher", "faces", "batter")
REV_REL = ("batter", "rev_faces", "pitcher")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TemporalGATTransformer(nn.Module):
    def __init__(
        self,
        pitcher_in_dim: int,
        batter_in_dim: int,
        edge_attr_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        model_dim: int = 128,
        gat_heads: int = 4,
        tf_layers: int = 2,
        tf_heads: int = 4,
        max_lookback: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_lookback = max_lookback
        self.edge_attr_dim = edge_attr_dim

        self.pitcher_proj = nn.Linear(pitcher_in_dim, hidden_dim)
        self.batter_proj = nn.Linear(batter_in_dim, hidden_dim)
        self.hetero_conv = HeteroConv(
            {
                REL: GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    edge_dim=edge_attr_dim,
                    add_self_loops=False,
                    dropout=dropout,
                ),
                REV_REL: GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    edge_dim=edge_attr_dim,
                    add_self_loops=False,
                    dropout=dropout,
                ),
            },
            aggr="sum",
        )

        self.graph_state_proj = nn.Linear(hidden_dim * 2 + edge_attr_dim, model_dim)
        self.edge_token_proj = nn.Linear(hidden_dim * 2 + edge_attr_dim, model_dim)
        self.pos_embed = nn.Embedding(max_lookback + 1, model_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=tf_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)

        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def encode_graph(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {
            "pitcher": F.gelu(self.pitcher_proj(torch.cat([data["pitcher"].x, data["pitcher"].x_season], dim=-1))),
            "batter": F.gelu(self.batter_proj(torch.cat([data["batter"].x, data["batter"].x_season], dim=-1))),
        }
        edge_index = data[REL].edge_index
        edge_attr = data[REL].edge_attr
        edge_index_dict = {REL: edge_index, REV_REL: edge_index.flip(0)}
        edge_attr_dict = {REL: edge_attr, REV_REL: edge_attr}
        h = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)
        return {k: F.gelu(v) for k, v in h.items()}

    def build_graph_state(self, h: dict[str, torch.Tensor], data: HeteroData) -> torch.Tensor:
        p_mean = h["pitcher"].mean(dim=0)
        b_mean = h["batter"].mean(dim=0)
        e_mean = data[REL].edge_attr.mean(dim=0) if data[REL].edge_attr.size(0) > 0 else torch.zeros(
            self.edge_attr_dim, device=p_mean.device
        )
        return self.graph_state_proj(torch.cat([p_mean, b_mean, e_mean], dim=-1))

    def temporal_context(self, history: Deque[torch.Tensor], state_t: torch.Tensor) -> torch.Tensor:
        seq = list(history) + [state_t]
        seq_t = torch.stack(seq, dim=0)  # [L,D]
        pos = torch.arange(seq_t.size(0), device=seq_t.device)
        seq_t = seq_t + self.pos_embed(pos)
        out = self.temporal_encoder(seq_t.unsqueeze(0))  # [1,L,D]
        return out[:, -1, :]  # [1,D]

    def build_sample_tokens(
        self,
        h: dict[str, torch.Tensor],
        data: HeteroData,
        pitcher_idx: torch.Tensor,
        batter_idx: torch.Tensor,
        edge_idx: torch.Tensor,
    ) -> torch.Tensor:
        p = h["pitcher"][pitcher_idx]
        b = h["batter"][batter_idx]
        e = data[REL].edge_attr[edge_idx]
        return self.edge_token_proj(torch.cat([p, b, e], dim=-1))

    def classify(self, sample_tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ctx_rep = ctx.expand(sample_tokens.size(0), -1)
        return self.classifier(torch.cat([sample_tokens, ctx_rep], dim=-1))


def split_time_indices(num_steps: int, ratio_train: float = 0.8) -> tuple[list[int], list[int]]:
    cut = max(1, int(num_steps * ratio_train))
    cut = min(cut, num_steps - 1)
    train_t = list(range(cut))
    test_t = list(range(cut, num_steps))
    return train_t, test_t


def evaluate(
    model: TemporalGATTransformer,
    graphs: list[HeteroData],
    samples_df: pd.DataFrame,
    time_steps: list[int],
    device: torch.device,
    max_lookback: int,
    class_weights: torch.Tensor | None = None,
) -> tuple[float, float]:
    model.eval()
    history: Deque[torch.Tensor] = deque(maxlen=max_lookback)
    loss_sum = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for t in time_steps:
            g = graphs[t].to(device)
            batch = samples_df[samples_df["time_idx"] == t]
            if batch.empty:
                continue

            h = model.encode_graph(g)
            state_t = model.build_graph_state(h, g)
            ctx = model.temporal_context(history, state_t)

            p_idx = torch.tensor(batch["pitcher_local_idx"].to_numpy(), dtype=torch.long, device=device)
            b_idx = torch.tensor(batch["batter_local_idx"].to_numpy(), dtype=torch.long, device=device)
            e_idx = torch.tensor(batch["edge_local_idx"].to_numpy(), dtype=torch.long, device=device)
            y = torch.tensor(batch["target_idx"].to_numpy(), dtype=torch.long, device=device)

            tok = model.build_sample_tokens(h, g, p_idx, b_idx, e_idx)
            logits = model.classify(tok, ctx)
            loss = F.cross_entropy(logits, y, weight=class_weights)

            n = y.numel()
            loss_sum += float(loss.item()) * n
            total += n
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            history.append(state_t.detach())

    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def export_test_predictions_csv(
    model: TemporalGATTransformer,
    graphs: list[HeteroData],
    samples_df: pd.DataFrame,
    time_steps: list[int],
    idx_to_label: dict[int, str],
    out_csv: Path,
    device: torch.device,
    max_lookback: int,
) -> None:
    """Run chronological inference on test steps and write per-sample predictions."""
    model.eval()
    history: Deque[torch.Tensor] = deque(maxlen=max_lookback)
    rows: list[dict] = []

    with torch.no_grad():
        for t in time_steps:
            g = graphs[t].to(device)
            batch = samples_df[samples_df["time_idx"] == t]
            if batch.empty:
                continue

            h = model.encode_graph(g)
            state_t = model.build_graph_state(h, g)
            ctx = model.temporal_context(history, state_t)

            p_idx = torch.tensor(batch["pitcher_local_idx"].to_numpy(), dtype=torch.long, device=device)
            b_idx = torch.tensor(batch["batter_local_idx"].to_numpy(), dtype=torch.long, device=device)
            e_idx = torch.tensor(batch["edge_local_idx"].to_numpy(), dtype=torch.long, device=device)
            y_true = batch["target_idx"].to_numpy(dtype=np.int64)

            tok = model.build_sample_tokens(h, g, p_idx, b_idx, e_idx)
            logits = model.classify(tok, ctx)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            y_pred = probs.argmax(axis=1).astype(np.int64)

            for i, (_, r) in enumerate(batch.reset_index(drop=True).iterrows()):
                out_row = {
                    "time_idx": int(r["time_idx"]),
                    "game_date": r["game_date"],
                    "game_pk": int(r["game_pk"]),
                    "inning": int(r["inning"]),
                    "at_bat_number": int(r["at_bat_number"]),
                    "pitcher": int(r["pitcher"]),
                    "batter": int(r["batter"]),
                    "target_idx": int(y_true[i]),
                    "target_class": idx_to_label[int(y_true[i])],
                    "pred_idx": int(y_pred[i]),
                    "pred_class": idx_to_label[int(y_pred[i])],
                    "correct": int(y_true[i] == y_pred[i]),
                }
                for cls_idx in sorted(idx_to_label.keys()):
                    out_row[f"prob_{idx_to_label[cls_idx]}"] = float(probs[i, cls_idx])
                rows.append(out_row)

            history.append(state_t.detach())

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)


def iter_pa_minibatches(batch_df: pd.DataFrame, batch_size: int, shuffle: bool) -> list[pd.DataFrame]:
    """Split one time-step dataframe into PA mini-batches."""
    if batch_df.empty:
        return []
    if batch_size <= 0 or batch_size >= len(batch_df):
        return [batch_df]
    if shuffle:
        batch_df = batch_df.sample(frac=1.0).reset_index(drop=True)
    chunks = []
    for i in range(0, len(batch_df), batch_size):
        chunks.append(batch_df.iloc[i : i + batch_size])
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-path", type=str, default="data/logs/graph_dataset_context_only.pkl")
    parser.add_argument("--samples-csv", type=str, default="data/logs/temporal_training_samples.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--tf-layers", type=int, default=2)
    parser.add_argument("--tf-heads", type=int, default=4)
    parser.add_argument("--max-lookback", type=int, default=5)
    parser.add_argument(
        "--pa-batch-size",
        type=int,
        default=0,
        help="PA mini-batch size within each time step; <=0 means all PAs at once per time step.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--class-weighting",
        type=str,
        default="inverse_freq",
        choices=["none", "inverse_freq", "manual"],
        help="Class weighting mode for cross entropy.",
    )
    parser.add_argument(
        "--manual-class-weights",
        type=str,
        default="",
        help="Comma-separated class weights in label-index order; used when --class-weighting=manual.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default="data/logs/test_predictions_temporal.csv",
        help="Where to write per-sample test predictions after final epoch.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    with open(args.graph_path, "rb") as f:
        payload = pickle.load(f)
    graphs: list[HeteroData] = payload["graphs"]
    label_to_idx = payload["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    samples_df = pd.read_csv(args.samples_csv)
    if len(graphs) < 2:
        raise ValueError("Need at least 2 time steps for chronological split.")

    train_steps, test_steps = split_time_indices(len(graphs), ratio_train=0.8)
    print(
        f"Loaded {len(graphs)} graphs and {len(samples_df):,} samples | "
        f"train_steps={len(train_steps)} test_steps={len(test_steps)}"
    )
    print(f"Classes: {label_to_idx}")
    if args.pa_batch_size <= 0:
        print("Training mode: one optimization step per time step using all PAs at that time.")
    else:
        print(f"Training mode: PA mini-batches within each time step (size={args.pa_batch_size}).")

    g0 = graphs[0]
    pitcher_in = g0["pitcher"].x.size(-1) + g0["pitcher"].x_season.size(-1)
    batter_in = g0["batter"].x.size(-1) + g0["batter"].x_season.size(-1)
    edge_dim = g0[REL].edge_attr.size(-1)

    model = TemporalGATTransformer(
        pitcher_in_dim=pitcher_in,
        batter_in_dim=batter_in,
        edge_attr_dim=edge_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        model_dim=args.model_dim,
        gat_heads=args.gat_heads,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        max_lookback=args.max_lookback,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build class weights from train split only.
    train_samples = samples_df[samples_df["time_idx"].isin(train_steps)].copy()
    class_weights: torch.Tensor | None = None
    if args.class_weighting == "inverse_freq":
        counts = train_samples["target_idx"].value_counts().to_dict()
        w = []
        for c in range(num_classes):
            cnt = float(counts.get(c, 0))
            w.append(1.0 / cnt if cnt > 0 else 0.0)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        if torch.any(class_weights > 0):
            class_weights = class_weights / class_weights.mean().clamp_min(1e-8)
        print(f"Using inverse-freq class weights: {class_weights.detach().cpu().tolist()}")
    elif args.class_weighting == "manual":
        parts = [p.strip() for p in args.manual_class_weights.split(",") if p.strip()]
        if len(parts) != num_classes:
            raise ValueError(
                f"--manual-class-weights must provide {num_classes} values; got {len(parts)}"
            )
        class_weights = torch.tensor([float(x) for x in parts], dtype=torch.float32, device=device)
        print(f"Using manual class weights: {class_weights.detach().cpu().tolist()}")
    else:
        print("Using unweighted cross entropy.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        history: Deque[torch.Tensor] = deque(maxlen=args.max_lookback)
        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0

        for t in train_steps:
            g = graphs[t].to(device)
            day_samples = samples_df[samples_df["time_idx"] == t]
            if day_samples.empty:
                continue

            # Keep the temporal context fixed for this time-step while consuming all PAs at t.
            with torch.no_grad():
                h_state = model.encode_graph(g)
                state_t = model.build_graph_state(h_state, g)
                ctx = model.temporal_context(history, state_t).detach()

            pa_batches = iter_pa_minibatches(day_samples, args.pa_batch_size, shuffle=True)
            for mb in pa_batches:
                p_idx = torch.tensor(mb["pitcher_local_idx"].to_numpy(), dtype=torch.long, device=device)
                b_idx = torch.tensor(mb["batter_local_idx"].to_numpy(), dtype=torch.long, device=device)
                e_idx = torch.tensor(mb["edge_local_idx"].to_numpy(), dtype=torch.long, device=device)
                y = torch.tensor(mb["target_idx"].to_numpy(), dtype=torch.long, device=device)

                # Fresh forward per mini-batch avoids retaining/rewinding the same graph.
                h = model.encode_graph(g)
                tok = model.build_sample_tokens(h, g, p_idx, b_idx, e_idx)
                logits = model.classify(tok, ctx)
                loss = F.cross_entropy(logits, y, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                n = y.numel()
                train_loss_sum += float(loss.item()) * n
                train_total += n
                train_correct += int((logits.argmax(dim=-1) == y).sum().item())
            history.append(state_t.detach())

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        test_loss, test_acc = evaluate(
            model,
            graphs,
            samples_df,
            test_steps,
            device,
            args.max_lookback,
            class_weights=class_weights,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    out_csv = Path(args.predictions_csv)
    if not out_csv.is_absolute():
        out_csv = Path.cwd() / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    export_test_predictions_csv(
        model=model,
        graphs=graphs,
        samples_df=samples_df,
        time_steps=test_steps,
        idx_to_label=idx_to_label,
        out_csv=out_csv,
        device=device,
        max_lookback=args.max_lookback,
    )
    print(f"Saved test predictions to {out_csv}")


if __name__ == "__main__":
    main()

