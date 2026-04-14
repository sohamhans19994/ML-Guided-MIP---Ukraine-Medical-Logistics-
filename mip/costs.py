from __future__ import annotations

import networkx as nx

from .data import MIPInstance


def compute_cost_matrix(instance: MIPInstance) -> dict[tuple[int, int], float]:
    """Compute c[(i,j)] = shortest-path travel time (hours) from demand node i
    to candidate hub j on the unattacked coarse graph.

    Raises ValueError if any demand node cannot reach any candidate hub — this
    would make the MIP trivially infeasible and indicates a graph connectivity
    problem upstream.
    """
    CG = instance.CG
    c: dict[tuple[int, int], float] = {}
    for i in instance.D:
        lengths = nx.single_source_dijkstra_path_length(CG, i, weight="travel_time")
        for j in instance.N:
            if j not in lengths:
                raise ValueError(
                    f"Demand node {i} cannot reach candidate node {j} on the adaptive graph. "
                    "Check graph connectivity."
                )
            c[(i, j)] = float(lengths[j] / 3600.0)
    return c
