"""
GCN / GAT Model — Graph-based anomaly detection (Core Contribution 1).

Implements:
    - GCNAnomaly:   GCN-based anomaly classifier
    - GATAnomaly:   GAT-based anomaly classifier (with attention)
    - TemporalGCN:  GCN + LSTM for spatio-temporal anomaly detection (Phase 3)

The graph captures spatial interactions between objects that CNN alone
cannot model (e.g., fighting = two persons close + fast motion towards
each other → anomalous edge pattern).

References:
    - STGCN for Video Anomaly Detection
    - Hierarchical ST-GCN for Video Anomaly Detection
    - WAGCN (Adaptive GCN for Video Anomaly Detection)
    - MissionGNN (GNN + Knowledge Graph)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================================
# Graph Convolution Layers (standalone — no torch_geometric dep)
# ============================================================

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution layer (Kipf & Welling, 2017).
    H' = σ(D^{-1/2} A D^{-1/2} H W)

    This standalone implementation avoids hard dependency on
    torch_geometric during prototyping.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, in_dim) node features
            edge_index: (2, E) edge indices
            num_nodes:  int
        Returns:
            (N, out_dim)
        """
        # Build adjacency matrix with self-loops
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + torch.eye(num_nodes, device=x.device)  # self-loop

        # Normalise: D^{-1/2} A D^{-1/2}
        degree = adj.sum(dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0.0
        D = torch.diag(degree_inv_sqrt)
        adj_norm = D @ adj @ D

        # Graph convolution
        h = x @ self.weight          # (N, out_dim)
        h = adj_norm @ h             # message passing
        if self.bias is not None:
            h = h + self.bias

        return h


class GATLayer(nn.Module):
    """
    Graph Attention Layer (Veličković et al., 2018).

    Computes attention coefficients for each edge and aggregates
    neighbour features using learned attention weights.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.FloatTensor(num_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            x:          (N, in_dim)
            edge_index: (2, E)
            num_nodes:  int
        Returns:
            (N, out_dim * num_heads) if concat else (N, out_dim)
        """
        N = num_nodes
        H = self.num_heads

        # Linear transform → multi-head
        h = self.W(x).view(N, H, self.out_dim)  # (N, H, out_dim)

        # Attention scores
        attn_src = (h * self.a_src).sum(dim=-1)  # (N, H)
        attn_dst = (h * self.a_dst).sum(dim=-1)  # (N, H)

        # Build attention matrix
        attn = torch.zeros(N, N, H, device=x.device)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            edge_attn = attn_src[src] + attn_dst[dst]  # (E, H)
            edge_attn = self.leaky_relu(edge_attn)

            # Scatter into attention matrix
            for k in range(edge_index.size(1)):
                attn[src[k], dst[k]] = edge_attn[k]

        # Self-loops
        self_attn = attn_src + attn_dst  # (N, H)
        for i in range(N):
            attn[i, i] = self_attn[i]

        # Softmax over neighbours
        attn = F.softmax(attn, dim=1)  # (N, N, H)
        attn = self.dropout(attn)

        # Aggregate
        # (N, N, H) x (N, H, out_dim) → (N, H, out_dim)
        out = torch.einsum("nmh,mhd->nhd", attn, h)

        if self.concat:
            return out.reshape(N, H * self.out_dim)
        else:
            return out.mean(dim=1)  # (N, out_dim)


# ============================================================
# Full Models
# ============================================================

class GCNAnomaly(nn.Module):
    """
    GCN-based anomaly detection model.

    Pipeline:
        Node features (N, F)
        → GCN layers
        → Graph-level readout (mean/max pooling)
        → Classifier → anomaly score
    """

    def __init__(
        self,
        node_feat_dim: int = config.GCN_NODE_FEAT_DIM,
        hidden_dim: int = config.GCN_HIDDEN_DIM,
        output_dim: int = config.GCN_OUTPUT_DIM,
        num_layers: int = config.GCN_NUM_LAYERS,
        dropout: float = config.GCN_DROPOUT,
    ):
        super().__init__()

        # Input projection (handle variable input dim)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else output_dim
            self.gcn_layers.append(GraphConvLayer(in_d, out_d))
            self.norms.append(nn.LayerNorm(out_d))

        self.dropout = nn.Dropout(dropout)

        # Classifier (graph-level)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (N, F)
            edge_index:    (2, E)
            num_nodes:     int
            batch:         (N,) batch assignment (for batched graphs)
        Returns:
            score: (B, 1) anomaly scores
        """
        x = self.input_proj(node_features)

        for gcn, norm in zip(self.gcn_layers, self.norms):
            x = gcn(x, edge_index, num_nodes)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Graph-level readout
        if batch is not None:
            # Batched graphs: pool per graph
            from torch_geometric.nn import global_mean_pool
            graph_feat = global_mean_pool(x, batch)
        else:
            # Single graph: mean pool all nodes
            graph_feat = x.mean(dim=0, keepdim=True)  # (1, output_dim)

        score = self.classifier(graph_feat)
        return score


class GATAnomaly(nn.Module):
    """
    GAT-based anomaly detection model.

    Uses attention mechanism to learn importance weights for
    different neighbours — crucial for anomaly detection where
    specific interaction patterns indicate abnormality.
    """

    def __init__(
        self,
        node_feat_dim: int = config.GCN_NODE_FEAT_DIM,
        hidden_dim: int = config.GCN_HIDDEN_DIM,
        output_dim: int = config.GCN_OUTPUT_DIM,
        num_layers: int = config.GCN_NUM_LAYERS,
        num_heads: int = config.GAT_NUM_HEADS,
        dropout: float = config.GCN_DROPOUT,
    ):
        super().__init__()

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_d = hidden_dim
            else:
                in_d = hidden_dim * num_heads if i < num_layers - 1 else hidden_dim * num_heads

            if i < num_layers - 1:
                self.gat_layers.append(
                    GATLayer(in_d, hidden_dim, num_heads=num_heads, concat=True, dropout=dropout)
                )
                self.norms.append(nn.LayerNorm(hidden_dim * num_heads))
            else:
                self.gat_layers.append(
                    GATLayer(in_d, output_dim, num_heads=1, concat=False, dropout=dropout)
                )
                self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(node_features)

        for gat, norm in zip(self.gat_layers, self.norms):
            x = gat(x, edge_index, num_nodes)
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)

        # Graph readout
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_feat = global_mean_pool(x, batch)
        else:
            graph_feat = x.mean(dim=0, keepdim=True)

        return self.classifier(graph_feat)


class TemporalGCN(nn.Module):
    """
    Spatio-Temporal GCN (Phase 3).

    Combines:
        - GCN/GAT per frame → spatial features
        - LSTM over frames  → temporal patterns

    This is the STATE-OF-THE-ART direction for video anomaly detection.

    Pipeline:
        Graph_t (per frame)
        → GCN → graph_embedding_t
        → [graph_emb_1, ..., graph_emb_T] → LSTM
        → anomaly score
    """

    def __init__(
        self,
        node_feat_dim: int = config.GCN_NODE_FEAT_DIM,
        gcn_hidden: int = config.GCN_HIDDEN_DIM,
        gcn_output: int = config.GCN_OUTPUT_DIM,
        lstm_hidden: int = config.LSTM_HIDDEN_DIM,
        lstm_layers: int = config.LSTM_NUM_LAYERS,
        model_type: str = config.GCN_MODEL_TYPE,
        dropout: float = config.GCN_DROPOUT,
    ):
        super().__init__()

        # Spatial model (per frame)
        if model_type == "GAT":
            self.spatial = GATAnomaly(
                node_feat_dim=node_feat_dim,
                hidden_dim=gcn_hidden,
                output_dim=gcn_output,
                dropout=dropout,
            )
        else:
            self.spatial = GCNAnomaly(
                node_feat_dim=node_feat_dim,
                hidden_dim=gcn_hidden,
                output_dim=gcn_output,
                dropout=dropout,
            )

        # Remove the spatial model's classifier (we use our own)
        self.spatial.classifier = nn.Identity()

        # Temporal model
        self.lstm = nn.LSTM(
            input_size=gcn_output,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward_single_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Process a single frame's graph → embedding."""
        graph_emb = self.spatial(node_features, edge_index, num_nodes)
        return graph_emb  # (1, gcn_output)

    def forward(
        self,
        graphs: list,
    ) -> torch.Tensor:
        """
        Args:
            graphs: list of T graph dicts, each with
                    {node_features, edge_index, num_nodes}
        Returns:
            score: (1, 1) anomaly score
        """
        embeddings = []
        for g in graphs:
            if g["num_nodes"] == 0:
                # Empty graph → zero embedding
                emb = torch.zeros(1, config.GCN_OUTPUT_DIM, device=next(self.parameters()).device)
            else:
                emb = self.forward_single_graph(
                    g["node_features"].to(next(self.parameters()).device),
                    g["edge_index"].to(next(self.parameters()).device),
                    g["num_nodes"],
                )
            embeddings.append(emb)

        # Stack temporal: (1, T, gcn_output)
        temporal = torch.stack(embeddings, dim=1)

        # LSTM
        lstm_out, _ = self.lstm(temporal)
        last_hidden = lstm_out[:, -1, :]  # (1, lstm_hidden)

        score = self.classifier(last_hidden)
        return score


def get_model(model_type: str = config.GCN_MODEL_TYPE, **kwargs) -> nn.Module:
    """Factory function to create model by type."""
    models_map = {
        "GCN": GCNAnomaly,
        "GAT": GATAnomaly,
        "TemporalGCN": TemporalGCN,
    }
    if model_type not in models_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models_map.keys())}")
    return models_map[model_type](**kwargs)


if __name__ == "__main__":
    # Quick test
    print("=== Testing GCN Model ===")
    gcn = GCNAnomaly(node_feat_dim=8, hidden_dim=64, output_dim=32)
    x = torch.randn(5, 8)  # 5 nodes, 8 features
    edge_idx = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    score = gcn(x, edge_idx, num_nodes=5)
    print(f"GCN output: {score.shape}, score={score.item():.4f}")

    print("\n=== Testing GAT Model ===")
    gat = GATAnomaly(node_feat_dim=8, hidden_dim=64, output_dim=32)
    score = gat(x, edge_idx, num_nodes=5)
    print(f"GAT output: {score.shape}, score={score.item():.4f}")

    print("\n=== Testing TemporalGCN ===")
    tgcn = TemporalGCN(node_feat_dim=8, gcn_hidden=64, gcn_output=32, lstm_hidden=64)
    graphs = [
        {"node_features": torch.randn(5, 8), "edge_index": edge_idx, "num_nodes": 5}
        for _ in range(16)
    ]
    score = tgcn(graphs)
    print(f"TemporalGCN output: {score.shape}, score={score.item():.4f}")

    total_params = sum(p.numel() for p in tgcn.parameters())
    print(f"\nTotal parameters: {total_params:,}")
