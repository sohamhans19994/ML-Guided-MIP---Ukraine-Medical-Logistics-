"""
ml/model.py
-----------
HubGNN: bipartite message-passing GNN for hub open/close prediction.

Architecture
------------
Stage 1 - Linear embedding
  Project hub features (6-dim) and scenario features (5-dim) into
  a shared hidden dimension h.

Stage 2 - Bipartite message passing (n_rounds rounds)
  Each round:
    (a) Hub -> Scenario half-step:
        Each scenario aggregates from its surviving hub neighbours.
        Message = MLP(hub_emb || edge_feat)  aggregated by mean.
    (b) Scenario -> Hub half-step:
        Each hub aggregates from the scenarios it survived.
        Message = MLP(scenario_emb || edge_feat)  aggregated by mean.

Stage 3 - Output head
  MLP(hub_emb) -> scalar logit -> sigmoid -> P(y_i = 1)

Input / output
--------------
  input : HeteroData from ml.dataset.HubLocationDataset
  output: FloatTensor [n_hubs]  — P(y_i=1) for each hub candidate
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


def _mlp(in_dim: int, out_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
    """Two-layer MLP with ReLU."""
    mid = hidden_dim or out_dim
    return nn.Sequential(
        nn.Linear(in_dim, mid),
        nn.ReLU(),
        nn.Linear(mid, out_dim),
    )


class HubGNN(nn.Module):
    """
    Bipartite GNN that predicts P(hub_i open) from a hub-scenario graph.

    Parameters
    ----------
    hub_feat_dim   : input feature dimension for hub nodes   (default 6)
    scen_feat_dim  : input feature dimension for scenario nodes (default 5)
    edge_feat_dim  : edge feature dimension                  (default 4)
    hidden_dim     : embedding size throughout the network   (default 64)
    n_rounds       : number of bipartite message-passing rounds (default 2)
    dropout        : dropout rate on hub embeddings before output head
    """

    def __init__(
        self,
        hub_feat_dim:  int = 6,
        scen_feat_dim: int = 5,
        edge_feat_dim: int = 4,
        hidden_dim:    int = 64,
        n_rounds:      int = 2,
        dropout:       float = 0.0,
    ) -> None:
        super().__init__()

        self.n_rounds = n_rounds
        h = hidden_dim

        # ---- Stage 1: initial embeddings ----
        self.hub_embed  = nn.Linear(hub_feat_dim,  h)
        self.scen_embed = nn.Linear(scen_feat_dim, h)

        # ---- Stage 2: message MLPs (one pair per round) ----
        # Hub -> Scenario: message = MLP(hub_emb || edge_feat)
        self.msg_h2s = nn.ModuleList([
            _mlp(h + edge_feat_dim, h) for _ in range(n_rounds)
        ])
        # Update scenario embedding after aggregation
        self.upd_scen = nn.ModuleList([
            _mlp(h + h, h) for _ in range(n_rounds)
        ])

        # Scenario -> Hub: message = MLP(scen_emb || edge_feat)
        self.msg_s2h = nn.ModuleList([
            _mlp(h + edge_feat_dim, h) for _ in range(n_rounds)
        ])
        # Update hub embedding after aggregation
        self.upd_hub = nn.ModuleList([
            _mlp(h + h, h) for _ in range(n_rounds)
        ])

        # ---- Stage 3: output head ----
        self.dropout = nn.Dropout(dropout)
        self.head = _mlp(h, 1, hidden_dim=h)

    # ------------------------------------------------------------------
    # message passing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        src_emb:    torch.Tensor,   # [n_src, h]
        edge_index: torch.Tensor,   # [2, n_edges]  row0=src, row1=dst
        edge_attr:  torch.Tensor,   # [n_edges, e]
        msg_mlp:    nn.Module,
        n_dst:      int,
    ) -> torch.Tensor:
        """
        For each destination node, aggregate messages from its source neighbours.

        message_k = msg_mlp( src_emb[src_k] || edge_attr[k] )
        aggregated_j = mean over all k where dst_k == j
        Returns zeros for dst nodes with no incoming edges.
        """
        src_idx = edge_index[0]   # source node indices
        dst_idx = edge_index[1]   # destination node indices

        # build messages: [n_edges, h]
        msgs = msg_mlp(torch.cat([src_emb[src_idx], edge_attr], dim=-1))

        # mean-aggregate into destination nodes
        agg = torch.zeros(n_dst, msgs.size(-1), device=msgs.device)
        count = torch.zeros(n_dst, 1, device=msgs.device)
        agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(msgs), msgs)
        count.scatter_add_(0, dst_idx.unsqueeze(-1), torch.ones_like(dst_idx.unsqueeze(-1), dtype=torch.float))
        # avoid division by zero for isolated nodes
        count = count.clamp(min=1.0)
        return agg / count

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Parameters
        ----------
        data : HeteroData with keys:
               data['hub'].x             [n_hubs, hub_feat_dim]
               data['scenario'].x        [n_scen, scen_feat_dim]
               data['hub','survives_in','scenario'].edge_index  [2, n_edges]
               data['hub','survives_in','scenario'].edge_attr   [n_edges, edge_feat_dim]
               data['scenario','rev_survives','hub'].edge_index [2, n_edges]
               data['scenario','rev_survives','hub'].edge_attr  [n_edges, edge_feat_dim]

        Returns
        -------
        probs : FloatTensor [n_hubs]  — P(y_i = 1) for each hub
        """
        # ---- Stage 1: initial embeddings ----
        h_emb = torch.relu(self.hub_embed(data["hub"].x))         # [n_hubs, h]
        s_emb = torch.relu(self.scen_embed(data["scenario"].x))   # [n_scen, h]

        ei_hs = data["hub", "survives_in",  "scenario"].edge_index  # hub->scen
        ea_hs = data["hub", "survives_in",  "scenario"].edge_attr
        ei_sh = data["scenario", "rev_survives", "hub"].edge_index  # scen->hub
        ea_sh = data["scenario", "rev_survives", "hub"].edge_attr

        n_hubs = h_emb.size(0)
        n_scen = s_emb.size(0)

        # ---- Stage 2: message passing rounds ----
        for r in range(self.n_rounds):

            # (a) Hub -> Scenario
            agg_s = self._aggregate(
                src_emb=h_emb,
                edge_index=ei_hs,
                edge_attr=ea_hs,
                msg_mlp=self.msg_h2s[r],
                n_dst=n_scen,
            )
            s_emb = torch.relu(self.upd_scen[r](torch.cat([s_emb, agg_s], dim=-1)))

            # (b) Scenario -> Hub
            agg_h = self._aggregate(
                src_emb=s_emb,
                edge_index=ei_sh,
                edge_attr=ea_sh,
                msg_mlp=self.msg_s2h[r],
                n_dst=n_hubs,
            )
            h_emb = torch.relu(self.upd_hub[r](torch.cat([h_emb, agg_h], dim=-1)))

        # ---- Stage 3: output head ----
        h_emb = self.dropout(h_emb)
        logits = self.head(h_emb).squeeze(-1)   # [n_hubs]
        return torch.sigmoid(logits)             # [n_hubs]  P(y_i=1)
