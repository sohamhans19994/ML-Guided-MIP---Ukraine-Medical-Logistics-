from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .data import MIPInstance


@dataclass
class ScenarioData:
    """Fully resolved data for one attack scenario, ready to plug into the MIP."""

    id: str
    K: float                              # attack budget used to generate this scenario
    T: float                              # per-scenario service threshold (mapped from K)
    graph: nx.Graph                       # post-attack coarse graph
    surviving_nodes: list[int]            # nodes remaining after the attack
    proxy_demand_node: dict[int, int]     # demand node → nearest surviving node (identity if node survived)
    c: dict[tuple[int, int], float]       # c[(i,j)] travel time (hours) from proxied demand i to hub j
    summary: dict = field(default_factory=dict)


# Default mapping of attack budget K → service threshold T (hours).
# Larger attacks get a more lenient threshold because they are harder to survive.
DEFAULT_THRESHOLD_BY_K: dict[float, float] = {
    2.0: 2.25,
    3.0: 2.40,
    4.0: 2.50,
    5.0: 3.00,
}


def load_scenario_batch(
    scenario_root: str | Path,
    instance: MIPInstance,
    threshold_by_k: dict[float, float] | None = None,
) -> list[ScenarioData]:
    """Load all scenario sub-directories under *scenario_root* and return a
    list of :class:`ScenarioData` objects.

    Each sub-directory must contain:
    - ``scenario_summary.json`` — metadata including ``base_budget``
    - ``scenario_bundle.pkl``   — pickle with ``graphs.scenario_graph``

    If a demand node was removed by the attack, its demand is proxied to the
    geographically nearest surviving node so the MIP always has a feasible
    routing option.
    """
    if threshold_by_k is None:
        threshold_by_k = DEFAULT_THRESHOLD_BY_K

    scenario_root = Path(scenario_root)
    scenario_dirs = sorted(
        [p for p in scenario_root.iterdir() if p.is_dir() and p.name.startswith("K")]
    )
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario sub-directories found under {scenario_root}")

    node_pos: dict[int, tuple[float, float]] = {
        n: (float(instance.CG.nodes[n]["lon"]), float(instance.CG.nodes[n]["lat"]))
        for n in instance.N
    }

    scenarios: list[ScenarioData] = []
    for scenario_dir in scenario_dirs:
        summary = json.loads((scenario_dir / "scenario_summary.json").read_text())
        scenario_bundle = pd.read_pickle(scenario_dir / "scenario_bundle.pkl")
        Gs: nx.Graph = scenario_bundle["graphs"]["scenario_graph"]
        surviving_nodes = sorted(Gs.nodes())
        surviving_pos = {n: node_pos[n] for n in surviving_nodes}

        K = float(summary["base_budget"])
        T = float(threshold_by_k.get(K, max(threshold_by_k.values())))

        proxy_demand_node: dict[int, int] = {}
        c_s: dict[tuple[int, int], float] = {}

        for i in instance.D:
            if i in Gs.nodes:
                proxy = i
            else:
                lon_i, lat_i = node_pos[i]
                proxy = min(
                    surviving_nodes,
                    key=lambda n: (surviving_pos[n][0] - lon_i) ** 2 + (surviving_pos[n][1] - lat_i) ** 2,
                )
            proxy_demand_node[i] = proxy
            lengths = nx.single_source_dijkstra_path_length(Gs, proxy, weight="travel_time")
            for j in surviving_nodes:
                if j in lengths and np.isfinite(lengths[j]):
                    c_s[(i, j)] = float(lengths[j] / 3600.0)

        scenarios.append(
            ScenarioData(
                id=scenario_dir.name,
                K=K,
                T=T,
                graph=Gs,
                surviving_nodes=surviving_nodes,
                proxy_demand_node=proxy_demand_node,
                c=c_s,
                summary=summary,
            )
        )

    return scenarios


def scenario_summary_df(scenarios: list[ScenarioData]) -> pd.DataFrame:
    """Return a compact DataFrame summarising the loaded scenario batch."""
    rows = []
    for s in scenarios:
        ei = s.summary.get("edge_impacts", {})
        rows.append(
            {
                "scenario": s.id,
                "K": s.K,
                "T_s": s.T,
                "surviving_nodes": len(s.surviving_nodes),
                "removed_nodes": len(s.summary.get("removed_nodes", [])),
                "removed_edges": ei.get("removed_edges", 0),
                "degraded_edges": ei.get("degraded_edges", 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["K", "scenario"]).reset_index(drop=True)
