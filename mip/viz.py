from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .data import MIPInstance
from .solution import DeterministicResult, RobustResult


def plot_solution(
    instance: MIPInstance,
    result: DeterministicResult | RobustResult,
    borders: dict,
    occupied: dict,
    title: str | None = None,
) -> plt.Figure:
    """Map the open hubs and demand nodes on the coarse graph.

    Parameters
    ----------
    instance : MIPInstance
    result   : DeterministicResult or RobustResult
    borders  : dict from load_ukraine_sovereign_geometry()
               keys: ukraine_shape, sovereign_shape, sovereign_border_gs
    occupied : dict from load_current_occupied_snapshot()
               keys: occupied_gs
    title    : optional override for the map title

    Returns
    -------
    matplotlib Figure
    """
    CG = instance.CG
    demand_df = instance.demand_df
    open_hubs = result.open_hubs
    hub_capacity = result.hub_capacity

    adaptive_pos = {n: (CG.nodes[n]["lon"], CG.nodes[n]["lat"]) for n in CG.nodes()}

    fig, ax = plt.subplots(figsize=(14, 9))

    borders["ukraine_shape"].plot(
        ax=ax, color="#f2ecdf", alpha=0.95, edgecolor="black", linewidth=1.0, zorder=0,
        label="Ukraine (pre-war border)",
    )
    occupied["occupied_gs"].plot(
        ax=ax, color="#f7c8c8", alpha=0.45, edgecolor="crimson", linewidth=1.2, zorder=0.4,
        label="Occupied area",
    )
    borders["sovereign_shape"].plot(
        ax=ax, color="#dfead9", alpha=0.55, edgecolor="darkgreen", linewidth=1.5, zorder=0.8,
        label="Sovereign-held area",
    )
    borders["sovereign_border_gs"].plot(
        ax=ax, color="darkgreen", linewidth=1.8, alpha=0.95, zorder=1.0,
    )
    occupied["occupied_gs"].boundary.plot(
        ax=ax, color="crimson", linewidth=1.3, alpha=0.95, zorder=1.1,
    )

    edge_list = list(CG.edges(data=True))
    path_counts = [d.get("abstracted_path_count", 1) for _, _, d in edge_list] or [1]
    edge_widths = [
        0.6 + 4.4 * np.sqrt(d.get("abstracted_path_count", 1)) / np.sqrt(max(path_counts))
        for _, _, d in edge_list
    ]
    ec = nx.draw_networkx_edges(
        CG, adaptive_pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        edge_color="gray", alpha=0.18, width=edge_widths,
    )
    ec.set_zorder(2)

    base_sizes = {n: 20 + 4 * np.sqrt(CG.nodes[n].get("member_count", 1)) for n in CG.nodes()}
    nc = nx.draw_networkx_nodes(
        CG, adaptive_pos, ax=ax,
        node_size=[base_sizes[n] for n in CG.nodes()],
        node_color="#bfb7a9", alpha=0.60,
    )
    nc.set_zorder(3)

    max_d = demand_df["demand_amount"].max() or 1.0
    demand_sizes = [25 + 175 * np.sqrt(max(v, 0.0) / max(max_d, 1e-9)) for v in demand_df["demand_amount"]]
    ax.scatter(
        demand_df["plot_lon"], demand_df["plot_lat"],
        c="#c62828", s=demand_sizes, alpha=0.9,
        edgecolors="white", linewidths=0.6, zorder=5, label="Demand node",
    )

    if open_hubs:
        max_cap = max(hub_capacity.values()) if hub_capacity else 1.0
        ax.scatter(
            [CG.nodes[j]["lon"] for j in open_hubs],
            [CG.nodes[j]["lat"] for j in open_hubs],
            s=[base_sizes[j] * (1.5 + 0.5 * np.sqrt(max(hub_capacity[j], 0.0) / max(max_cap, 1e-9))) for j in open_hubs],
            c="forestgreen", edgecolors="black", linewidths=1.0, zorder=6, label="Open hub",
        )

    xmin, ymin, xmax, ymax = borders["ukraine_shape"].total_bounds
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_title(title or f"Facility-location solution — {len(open_hubs)} hubs open")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig


def plot_service_levels(result: RobustResult) -> plt.Figure:
    """Bar chart of per-scenario service cost vs threshold.

    Violated scenarios (w_s=1) are shown in red; satisfied ones in steel blue.
    """
    df = result.scenario_service_df
    fig, ax = plt.subplots(figsize=(10, 5))

    bar_colors = ["crimson" if v else "steelblue" for v in df["violated"]]
    ax.bar(df["scenario"], df["lhs_service"], color=bar_colors, alpha=0.85, label="Avg routing cost")
    ax.plot(df["scenario"], df["T_s"], color="black", marker="o", linewidth=1.5, label="Threshold T_s")

    ax.set_title(
        f"Per-scenario service levels — {len(result.violated_scenarios)} violation(s) "
        f"(δ={result.delta:.0%}, budget={int(len(df) * result.delta)})"
    )
    ax.set_ylabel("Avg routing cost (hours)")
    ax.set_xlabel("Scenario")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def load_map_layers(config=None):
    """Convenience loader: returns (borders, occupied) dicts for plot_solution().

    This wraps the synthetic_data loaders so callers don't need to import them
    directly.
    """
    from synthetic_data.config import load_config
    from synthetic_data.occupied import load_current_occupied_snapshot, load_ukraine_sovereign_geometry

    cfg = config or load_config()
    occupied = load_current_occupied_snapshot(cfg)
    borders = load_ukraine_sovereign_geometry(cfg, occupied["occupied_geom"])
    return borders, occupied
