from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from synthetic_data import load_saved_synthetic_bundle


@dataclass
class MIPInstance:
    """All inputs the MIP needs, derived from the synthetic data bundle."""

    CG: nx.Graph
    D: list[int]              # demand node ids on the coarse graph
    N: list[int]              # all candidate hub node ids (= all coarse nodes)
    demand: dict[int, float]  # daily demand at each demand node
    a: dict[int, float]       # fixed opening cost per node
    b: dict[int, float]       # per-unit capacity cost per node
    demand_df: pd.DataFrame   # demand rows with plot_lat / plot_lon for visualisation


def load_instance(bundle_path: str | Path | None = None) -> MIPInstance:
    """Load the synthetic data bundle and return a ready-to-use MIPInstance.

    Aggregates demand in case multiple snapped demand rows land on the same
    coarse node (can happen when k-medoids centroids snap to the same coarse
    cluster).
    """
    bundle = load_saved_synthetic_bundle(bundle_path)
    CG: nx.Graph = bundle["graphs"]["adaptive_graph"]
    raw_demand_nodes: pd.DataFrame = bundle["demand_nodes"].copy()

    demand_df = (
        raw_demand_nodes.groupby("coarse_node", as_index=False)
        .agg(
            demand_amount=("daily_demand", "sum"),
            total_fatalities=("total_fatalities", "sum"),
            plot_lat=("plot_lat", "first"),
            plot_lon=("plot_lon", "first"),
        )
        .sort_values("coarse_node")
        .reset_index(drop=True)
    )

    D: list[int] = demand_df["coarse_node"].tolist()
    N: list[int] = list(CG.nodes())
    demand: dict[int, float] = dict(zip(demand_df["coarse_node"], demand_df["demand_amount"]))
    a: dict[int, float] = {j: float(CG.nodes[j]["a_i"]) for j in N}
    b: dict[int, float] = {j: float(CG.nodes[j]["b_i"]) for j in N}

    return MIPInstance(CG=CG, D=D, N=N, demand=demand, a=a, b=b, demand_df=demand_df)
