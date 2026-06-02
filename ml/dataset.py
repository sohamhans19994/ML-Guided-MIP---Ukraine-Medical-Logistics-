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
from mip.scenarios import ScenarioData
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
# Standalone graph-building utilities (used by both dataset and predict_search)
# ---------------------------------------------------------------------------

def build_hub_features(instance: MIPInstance) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    """
    Build static hub node features from the MIP instance.

    Returns
    -------
    hub_feats : FloatTensor [n_hubs, 6]
    hubs      : sorted list of hub node ids
    hub_idx   : dict mapping hub node id -> index in hubs
    """
    CG   = instance.CG
    hubs = sorted(instance.N)

    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    a_vals   = np.array([CG.nodes[h]["a_i"]                               for h in hubs], dtype=np.float32)
    b_vals   = np.array([CG.nodes[h]["b_i"]                               for h in hubs], dtype=np.float32)
    fl_cost  = np.array([CG.nodes[h].get("frontline_cost_component", 0.5) for h in hubs], dtype=np.float32)
    mem_cost = np.array([CG.nodes[h].get("member_cost_component",    0.5) for h in hubs], dtype=np.float32)
    edg_cost = np.array([CG.nodes[h].get("edge_support_cost_component", 0.5) for h in hubs], dtype=np.float32)
    degree   = np.array([float(CG.degree(h))                              for h in hubs], dtype=np.float32)

    feats = np.stack([
        _minmax(a_vals),
        _minmax(b_vals),
        fl_cost,
        mem_cost,
        edg_cost,
        _minmax(degree),
    ], axis=1)   # [n_hubs, 6]

    hub_idx = {h: i for i, h in enumerate(hubs)}
    return torch.from_numpy(feats), hubs, hub_idx


def build_instance_graph(
    instance:  MIPInstance,
    scenarios: list[ScenarioData],
    hub_feats: torch.Tensor,
    hubs:      list[int],
    hub_idx:   dict[int, int],
) -> HeteroData:
    """
    Build a HeteroData graph for one (instance, scenario-set) pair.

    Used at inference time by predict_search.py — no TrainingRecord needed.
    Labels (y, pool_y, pool_objs) are NOT attached; the graph is input-only.
    """
    D   = instance.D
    n_d = len(D)
    n_hubs = len(hubs)

    # ---- scenario node features --------------------------------------
    scen_rows = []
    for s in scenarios:
        ei           = s.summary.get("edge_impacts", {})
        n_removed_e  = float(ei.get("removed_edges",  0))
        n_degraded_e = float(ei.get("degraded_edges", 0))
        scen_rows.append([
            s.K / MAX_K,
            s.T / MAX_T,
            len(s.surviving_nodes) / max(n_hubs, 1),
            n_removed_e  / max(N_EDGES_BASE, 1),
            n_degraded_e / max(N_EDGES_BASE, 1),
        ])
    scen_x = torch.tensor(scen_rows, dtype=torch.float)   # [n_s, 5]

    # ---- edges: hub ↔ scenario ---------------------------------------
    hub_idxs, scen_idxs, edge_rows = [], [], []

    for s_idx, s in enumerate(scenarios):
        surviving_set = set(s.surviving_nodes)
        for j in hubs:
            if j not in surviving_set:
                continue

            costs = [s.c[(i, j)] for i in D if (i, j) in s.c]
            if not costs:
                continue

            hub_idxs.append(hub_idx[j])
            scen_idxs.append(s_idx)
            edge_rows.append([
                float(np.mean(costs)) / MAX_COST_HR,
                float(np.min(costs))  / MAX_COST_HR,
                float(np.max(costs))  / MAX_COST_HR,
                len(costs) / n_d,
            ])

    if edge_rows:
        ei_hs = torch.tensor([hub_idxs, scen_idxs], dtype=torch.long)
        ea_hs = torch.tensor(edge_rows, dtype=torch.float)
        ei_sh = ei_hs.flip(0)
    else:
        ei_hs = torch.zeros((2, 0), dtype=torch.long)
        ea_hs = torch.zeros((0, 4), dtype=torch.float)
        ei_sh = ei_hs.clone()

    data = HeteroData()
    data["hub"].x      = hub_feats
    data["scenario"].x = scen_x
    data["hub",      "survives_in",  "scenario"].edge_index = ei_hs
    data["hub",      "survives_in",  "scenario"].edge_attr  = ea_hs
    data["scenario", "rev_survives", "hub"      ].edge_index = ei_sh
    data["scenario", "rev_survives", "hub"      ].edge_attr  = ea_hs
    return data


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
        self._hub_feats, self.hubs, self.hub_idx = build_hub_features(instance)
        self.n_hubs = len(self.hubs)
        self.graphs = [self._build_graph(r) for r in records]

    def _build_graph(self, record: TrainingRecord) -> HeteroData:
        """Build graph + labels for one training record."""
        data = build_instance_graph(
            self.instance, record.scenarios,
            self._hub_feats, self.hubs, self.hub_idx,
        )

        opt = record.pool_solutions[0]
        data["hub"].y = torch.tensor(
            [float(opt.y_open.get(h, 0)) for h in self.hubs], dtype=torch.float,
        )
        data["hub"].pool_y = torch.tensor(
            [[float(sol.y_raw.get(h, 0.0)) for h in self.hubs]
             for sol in record.pool_solutions], dtype=torch.float,
        )
        data["hub"].pool_objs = torch.tensor(
            [sol.obj_val for sol in record.pool_solutions], dtype=torch.float,
        )
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
