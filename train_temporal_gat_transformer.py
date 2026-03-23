"""
Train a temporal graph model on rich graph snapshots.

Pipeline:
1) Load graph sequence from data/logs/graph_dataset_rich.pkl
2) Chronological split: first 80% train, last 20% test
3) For each train time step t:
   - sample one random edge from graph[t]
   - run GATv2 on graph[t]
   - encode up to 5 previous hidden graph states + current state with transformer
   - classify sampled edge outcome with an MLP head
4) Optimize with cross entropy loss
"""

from __future__ import annotations

import argparse
import pickle
import random
from collections import deque
from pathlib import Path
from typing import Deque

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
        self.model_dim = model_dim
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=tf_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.classifier = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def encode_graph(self, data: HeteroData) -> dict[str, torch.Tensor]:
        pitcher_x = torch.cat([data["pitcher"].x, data["pitcher"].x_season], dim=-1)
        batter_x = torch.cat([data["batter"].x, data["batter"].x_season], dim=-1)
        x_dict = {
            "pitcher": F.gelu(self.pitcher_proj(pitcher_x)),
            "batter": F.gelu(self.batter_proj(batter_x)),
        }

        edge_index = data[REL].edge_index
        edge_attr = data[REL].edge_attr
        edge_index_dict = {
            REL: edge_index,
            REV_REL: edge_index.flip(0),
        }
        edge_attr_dict = {
            REL: edge_attr,
            REV_REL: edge_attr,
        }
        h_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)
        return {k: F.gelu(v) for k, v in h_dict.items()}

    def build_graph_state(self, h_dict: dict[str, torch.Tensor], data: HeteroData) -> torch.Tensor:
        pitcher_mean = h_dict["pitcher"].mean(dim=0)
        batter_mean = h_dict["batter"].mean(dim=0)
        if data[REL].edge_attr.size(0) > 0:
            edge_mean = data[REL].edge_attr.mean(dim=0)
        else:
            edge_mean = torch.zeros(self.edge_attr_dim, device=pitcher_mean.device)
        raw_state = torch.cat([pitcher_mean, batter_mean, edge_mean], dim=-1)
        return self.graph_state_proj(raw_state)

    def build_edge_token(
        self,
        h_dict: dict[str, torch.Tensor],
        data: HeteroData,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = data[REL].edge_index
        edge_attr = data[REL].edge_attr
        src_idx = edge_index[0, edge_ids]
        dst_idx = edge_index[1, edge_ids]
        src_h = h_dict["pitcher"][src_idx]
        dst_h = h_dict["batter"][dst_idx]
        edge_h = edge_attr[edge_ids]
        return self.edge_token_proj(torch.cat([src_h, dst_h, edge_h], dim=-1))

    def temporal_context(
        self,
        history_states: Deque[torch.Tensor],
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        seq = list(history_states) + [current_state]
        seq_t = torch.stack(seq, dim=0)  # [L, D]
        pos_ids = torch.arange(seq_t.size(0), device=seq_t.device)
        seq_t = seq_t + self.pos_embed(pos_ids)
        enc = self.temporal_encoder(seq_t.unsqueeze(0))  # [1, L, D]
        return enc[:, -1, :]  # [1, D]

    def classify(
        self,
        edge_token: torch.Tensor,
        temporal_ctx: torch.Tensor,
    ) -> torch.Tensor:
        # edge_token: [N, D], temporal_ctx: [1, D]
        ctx = temporal_ctx.expand(edge_token.size(0), -1)
        logits = self.classifier(torch.cat([edge_token, ctx], dim=-1))
        return logits


def load_graph_payload(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "graphs" not in payload:
        raise ValueError(f"Unexpected payload format in {path}")
    return payload


def split_chronological(graphs: list[HeteroData], ratio_train: float = 0.8) -> tuple[list[HeteroData], list[HeteroData]]:
    cut = max(1, int(len(graphs) * ratio_train))
    cut = min(cut, len(graphs) - 1)
    return graphs[:cut], graphs[cut:]


def evaluate(
    model: TemporalGATTransformer,
    graphs: list[HeteroData],
    device: torch.device,
    max_lookback: int,
) -> tuple[float, float]:
    model.eval()
    history_states: Deque[torch.Tensor] = deque(maxlen=max_lookback)
    loss_sum = 0.0
    count = 0
    correct = 0

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            if g[REL].edge_index.size(1) == 0:
                continue

            h_dict = model.encode_graph(g)
            state_t = model.build_graph_state(h_dict, g)
            ctx = model.temporal_context(history_states, state_t)

            edge_ids = torch.arange(g[REL].edge_index.size(1), device=device)
            edge_tok = model.build_edge_token(h_dict, g, edge_ids)
            logits = model.classify(edge_tok, ctx)
            y = g[REL].y

            loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.item()) * y.numel()
            count += y.numel()
            correct += int((logits.argmax(dim=-1) == y).sum().item())

            history_states.append(state_t.detach())

    if count == 0:
        return 0.0, 0.0
    return loss_sum / count, correct / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/logs/graph_dataset_rich.pkl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--tf-layers", type=int, default=2)
    parser.add_argument("--tf-heads", type=int, default=4)
    parser.add_argument("--max-lookback", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    payload = load_graph_payload(Path(args.data_path))
    graphs: list[HeteroData] = payload["graphs"]
    label_to_idx: dict[str, int] = payload["label_to_idx"]
    num_classes = len(label_to_idx)
    if len(graphs) < 2:
        raise ValueError("Need at least 2 graphs for chronological train/test split.")

    train_graphs, test_graphs = split_chronological(graphs, ratio_train=0.8)
    print(f"Loaded {len(graphs)} graphs | train={len(train_graphs)} | test={len(test_graphs)}")
    print(f"Classes: {label_to_idx}")

    sample_g = train_graphs[0]
    pitcher_in = sample_g["pitcher"].x.size(-1) + sample_g["pitcher"].x_season.size(-1)
    batter_in = sample_g["batter"].x.size(-1) + sample_g["batter"].x_season.size(-1)
    edge_attr_dim = sample_g[REL].edge_attr.size(-1)

    model = TemporalGATTransformer(
        pitcher_in_dim=pitcher_in,
        batter_in_dim=batter_in,
        edge_attr_dim=edge_attr_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        model_dim=args.model_dim,
        gat_heads=args.gat_heads,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        max_lookback=args.max_lookback,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        history_states: Deque[torch.Tensor] = deque(maxlen=args.max_lookback)
        train_loss = 0.0
        train_count = 0
        train_correct = 0

        for g in train_graphs:
            g = g.to(device)
            e_count = g[REL].edge_index.size(1)
            if e_count == 0:
                continue

            # Chronological step t: random sample from graph[t].
            edge_id = random.randrange(e_count)
            edge_ids = torch.tensor([edge_id], device=device)
            target = g[REL].y[edge_ids]

            h_dict = model.encode_graph(g)
            state_t = model.build_graph_state(h_dict, g)
            ctx = model.temporal_context(history_states, state_t)
            edge_tok = model.build_edge_token(h_dict, g, edge_ids)
            logits = model.classify(edge_tok, ctx)

            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_count += 1
            train_correct += int((logits.argmax(dim=-1) == target).sum().item())
            history_states.append(state_t.detach())

        avg_train_loss = train_loss / max(train_count, 1)
        train_acc = train_correct / max(train_count, 1)
        test_loss, test_acc = evaluate(model, test_graphs, device, args.max_lookback)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )


if __name__ == "__main__":
    main()

