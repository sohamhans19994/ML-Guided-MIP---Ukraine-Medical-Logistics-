"""
ml/dataset.py
-------------
Converts TrainingRecord objects into PyTorch Geometric HeteroData graphs
for GNN training.

Graph structure
---------------
Two node types:
  'hub'      — 112 hub candidates (the y_i decision variables we predict)
  'scenario' — |S| attack scenarios

Edge type (defined in both directions for bidirectional message passing):
  ('hub', 'survives_in', 'scenario')  — hub j survived scenario s
  ('scenario', 'rev_survives', 'hub') — reverse for scenario → hub messages

An edge exists between hub j and scenario s iff j is in scenario.surviving_nodes
(i.e. the hub was not destroyed by the attack). If a hub is destroyed, no edge —
it contributes nothing to that scenario's routing.

Why not the raw A-matrix bipartite graph (HW1 style)?
  y_i only appears in constraint (2): u_i ≤ M·y_i. It does not appear in any
  scenario constraint. The raw A-matrix graph for y_i would be a diagonal
  one-to-one mapping with zero scenario information. The hub-scenario graph
  encodes the semantically meaningful structure instead: which hubs survive
  which attacks and at what routing cost.

Node features
-------------
Hub nodes  [n_hubs × 6]:
  a_i (norm), b_i (norm), frontline_cost_component,
  member_cost_component, edge_support_cost_component, degree (norm)

Scenario nodes  [n_scenarios × 5]:
  K / 5.0, T_s / 3.0, frac_nodes_surviving,
  frac_edges_removed, frac_edges_degraded

Edge features  [n_edges × 4]:
  mean_cost (norm), min_cost (norm), max_cost (norm), frac_demand_reachable

Labels
------
  data['hub'].y         — binary [n_hubs]         BCE label (optimal y_open)
  data['hub'].pool_y    — float  [n_pool, n_hubs] soft labels from pool (InfoNCE)
  data['hub'].pool_objs — float  [n_pool]         pool objective values (InfoNCE)
"""
from __future__ import annotations

import pickle
import io
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from mip.data import MIPInstance
from ml.training import TrainingRecord, PoolSolution


class _Unpickler(pickle.Unpickler):
    """Remaps __main__.TrainingRecord → ml.training.TrainingRecord.

    When ml/training.py is executed as __main__ (e.g. python -m ml.training),
    the dataclasses are stored in the pickle under the __main__ module path.
    This unpickler transparently redirects those lookups to the correct module.
    """
    _REMAP = {
        ("__main__", "TrainingRecord"): TrainingRecord,
        ("__main__", "PoolSolution"):   PoolSolution,
    }

    def find_class(self, module: str, name: str):
        remapped = self._REMAP.get((module, name))
        if remapped is not None:
            return remapped
        return super().find_class(module, name)


def _load_records(path: Path) -> list[TrainingRecord]:
    with open(path, "rb") as f:
        return _Unpickler(f).load()

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------
MAX_K        = 5.0    # largest attack budget in the oracle
MAX_T        = 3.0    # largest service threshold (K=5 scenarios)
MAX_COST_HR  = 24.0   # travel-time normalisation cap (hours)
N_EDGES_BASE = 148    # edges in the base coarse graph (from Table 1 in report)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class HubLocationDataset:
    """
    Wraps a list of TrainingRecord objects and exposes them as HeteroData
    graphs for PyG DataLoader consumption.

    All graphs are built eagerly in __init__ so that per-epoch access is O(1)
    and the DataLoader's worker processes share the same pre-built tensors.
    For 1000 records with 122-node graphs, the total memory footprint is small.
    """

    def __init__(
        self,
        records:  list[TrainingRecord],
        instance: MIPInstance,
    ) -> None:
        self.instance = instance
        self.hubs     = sorted(instance.N)          # canonical hub ordering
        self.n_hubs   = len(self.hubs)
        self.hub_idx  = {h: i for i, h in enumerate(self.hubs)}

        # hub features are the same for every record — build once
        self._hub_feats = self._build_hub_features()  # [n_hubs, 6]

        # build all graphs upfront
        self.graphs = [self._build_graph(r) for r in records]

    # ------------------------------------------------------------------
    # static hub features (identical across all records)
    # ------------------------------------------------------------------

    def _build_hub_features(self) -> torch.Tensor:
        CG = self.instance.CG

        a_vals   = np.array([CG.nodes[h]["a_i"]                              for h in self.hubs], dtype=np.float32)
        b_vals   = np.array([CG.nodes[h]["b_i"]                              for h in self.hubs], dtype=np.float32)
        fl_cost  = np.array([CG.nodes[h].get("frontline_cost_component", 0.5)   for h in self.hubs], dtype=np.float32)
        mem_cost = np.array([CG.nodes[h].get("member_cost_component",     0.5)  for h in self.hubs], dtype=np.float32)
        edg_cost = np.array([CG.nodes[h].get("edge_support_cost_component", 0.5) for h in self.hubs], dtype=np.float32)
        degree   = np.array([float(CG.degree(h))                              for h in self.hubs], dtype=np.float32)

        def _minmax(x: np.ndarray) -> np.ndarray:
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)

        feats = np.stack([
            _minmax(a_vals),   # opening cost (normalised)
            _minmax(b_vals),   # capacity cost (normalised)
            fl_cost,           # frontline danger (already 0-1)
            mem_cost,          # cluster-size cost component (already 0-1)
            edg_cost,          # edge-support cost component (already 0-1)
            _minmax(degree),   # road connectivity (normalised)
        ], axis=1)             # [n_hubs, 6]

        return torch.from_numpy(feats)

    # ------------------------------------------------------------------
    # per-record graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, record: TrainingRecord) -> HeteroData:
        scenarios = record.scenarios
        n_s       = len(scenarios)
        D         = self.instance.D
        n_d       = len(D)

        # ---- scenario node features ----------------------------------
        scen_rows = []
        for s in scenarios:
            ei           = s.summary.get("edge_impacts", {})
            n_removed_e  = float(ei.get("removed_edges",  0))
            n_degraded_e = float(ei.get("degraded_edges", 0))
            scen_rows.append([
                s.K / MAX_K,
                s.T / MAX_T,
                len(s.surviving_nodes) / max(self.n_hubs, 1),
                n_removed_e  / max(N_EDGES_BASE, 1),
                n_degraded_e / max(N_EDGES_BASE, 1),
            ])
        scen_x = torch.tensor(scen_rows, dtype=torch.float)   # [n_s, 5]

        # ---- edges: hub ↔ scenario -----------------------------------
        hub_idxs, scen_idxs, edge_rows = [], [], []

        for s_idx, s in enumerate(scenarios):
            surviving_set = set(s.surviving_nodes)
            for j in self.hubs:
                if j not in surviving_set:
                    continue   # hub destroyed in this scenario — no edge

                costs = [s.c[(i, j)] for i in D if (i, j) in s.c]
                if not costs:
                    continue   # hub unreachable from every demand node

                mean_c     = float(np.mean(costs)) / MAX_COST_HR
                min_c      = float(np.min(costs))  / MAX_COST_HR
                max_c      = float(np.max(costs))  / MAX_COST_HR
                frac_reach = len(costs) / n_d

                hub_idxs.append(self.hub_idx[j])
                scen_idxs.append(s_idx)
                edge_rows.append([mean_c, min_c, max_c, frac_reach])

        if edge_rows:
            ei_hs   = torch.tensor([hub_idxs, scen_idxs], dtype=torch.long)
            ea_hs   = torch.tensor(edge_rows, dtype=torch.float)
            ei_sh   = ei_hs.flip(0)   # reversed for scenario→hub direction
        else:
            ei_hs = torch.zeros((2, 0), dtype=torch.long)
            ea_hs = torch.zeros((0, 4), dtype=torch.float)
            ei_sh = ei_hs.clone()

        # ---- labels --------------------------------------------------
        opt   = record.pool_solutions[0]

        # hard binary labels for BCE — rounded optimal assignment
        y_bin = torch.tensor(
            [float(opt.y_open.get(h, 0)) for h in self.hubs],
            dtype=torch.float,
        )   # [n_hubs]

        # soft labels from all pool solutions for InfoNCE
        # y_raw are the raw Gurobi Xn values (continuous, not rounded)
        pool_y = torch.tensor(
            [[float(sol.y_raw.get(h, 0.0)) for h in self.hubs]
             for sol in record.pool_solutions],
            dtype=torch.float,
        )   # [n_pool, n_hubs]

        pool_objs = torch.tensor(
            [sol.obj_val for sol in record.pool_solutions],
            dtype=torch.float,
        )   # [n_pool]

        # ---- assemble ------------------------------------------------
        data = HeteroData()

        data["hub"].x         = self._hub_feats   # [n_hubs, 6]
        data["hub"].y         = y_bin             # [n_hubs]        BCE label
        data["hub"].pool_y    = pool_y            # [n_pool, n_hubs]
        data["hub"].pool_objs = pool_objs         # [n_pool]

        data["scenario"].x = scen_x              # [n_s, 5]

        # hub → scenario (hub survived in scenario)
        data["hub", "survives_in",  "scenario"].edge_index = ei_hs
        data["hub", "survives_in",  "scenario"].edge_attr  = ea_hs

        # scenario → hub (reverse for message passing)
        data["scenario", "rev_survives", "hub"].edge_index = ei_sh
        data["scenario", "rev_survives", "hub"].edge_attr  = ea_hs

        return data

    # ------------------------------------------------------------------
    # standard dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> HeteroData:
        return self.graphs[idx]


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_splits(
    records_path: str | Path,
    instance:     MIPInstance,
    val_frac:     float = 0.10,
    test_frac:    float = 0.10,
    seed:         int   = 42,
) -> tuple[HubLocationDataset, HubLocationDataset, HubLocationDataset]:
    """
    Load a records pickle and return (train, val, test) dataset splits.

    The split is done on the raw records list before graph construction
    so each split builds its own graphs independently.
    """
    records: list[TrainingRecord] = _load_records(Path(records_path))

    n       = len(records)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_records = [records[i] for i in perm[:n_train]]
    val_records   = [records[i] for i in perm[n_train:n_train + n_val]]
    test_records  = [records[i] for i in perm[n_train + n_val:]]

    train_ds = HubLocationDataset(train_records, instance)
    val_ds   = HubLocationDataset(val_records,   instance)
    test_ds  = HubLocationDataset(test_records,  instance)

    print(
        f"Loaded {n} records: "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds
