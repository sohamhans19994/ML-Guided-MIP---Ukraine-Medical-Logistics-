from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def plot_attack_scenario(bundle: dict, save_path: str | Path) -> None:
    base_graph = bundle["graphs"]["base_graph"]
    scenario_graph = bundle["graphs"]["scenario_graph"]
    edge_impacts = bundle["edge_impacts"]
    selected_locations = bundle["selected_locations"]
    strike_events = bundle["strike_events"]
    budget_summary = bundle["budget_summary"]
    geography = bundle.get("geography")

    base_positions = _project_positions(base_graph)
    scenario_positions = {node: base_positions[node] for node in scenario_graph.nodes()}

    edge_width_lookup = {
        row.edge_key: 0.8 + 3.2 * np.sqrt(max(1.0, row.abstracted_path_count)) / np.sqrt(max(1.0, edge_impacts["abstracted_path_count"].max()))
        for row in edge_impacts.itertuples(index=False)
    }
    location_points = _project_points(selected_locations)
    strike_points = _project_strike_points(strike_events)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    _draw_base_panel(
        axes[0],
        base_graph,
        base_positions,
        edge_impacts,
        edge_width_lookup,
        location_points,
        strike_points,
        budget_summary,
        geography,
    )
    _draw_scenario_panel(
        axes[1],
        scenario_graph,
        scenario_positions,
        edge_impacts,
        edge_width_lookup,
        location_points,
        strike_points,
        geography,
    )

    summary = bundle["summary"]
    fig.suptitle(
        f"Attack Scenario: {summary['scenario_id']} ({summary['attack_mode']}, K={summary['base_budget']:.2f})",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"locations={summary['location_count']}   "
            f"missile_strikes={summary['strike_counts']['missile']}   "
            f"bomb_strikes={summary['strike_counts']['bomb']}   "
            f"remaining_budget={summary['budget_summary']['remaining_budget']:.2f}   "
            f"removed_edges={summary['edge_impacts']['removed_edges']}   "
            f"degraded_edges={summary['edge_impacts']['degraded_edges']}   "
            f"infinite c_ij pairs={summary['c_ij']['infinite_pairs']}"
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _draw_base_panel(ax, graph, positions, edge_impacts, edge_width_lookup, location_points, strike_points, budget_summary, geography) -> None:
    _set_extent(ax, positions)
    _draw_geography(ax, geography)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edgelist=list(graph.edges()),
        edge_color="#c7c7c7",
        width=[edge_width_lookup.get(f"{min(u,v)}--{max(u,v)}", 1.0) for u, v in graph.edges()],
        alpha=0.45,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=[14 + 3.0 * np.sqrt(max(1.0, graph.nodes[n].get("member_count", 1.0))) for n in graph.nodes()],
        node_color="#546e7a",
        alpha=0.75,
        linewidths=0.0,
    )

    for status, color in [("struck_no_change", "#fdd835"), ("degraded", "#fb8c00"), ("removed", "#c62828")]:
        impacted = edge_impacts[edge_impacts["status"] == status]
        if impacted.empty:
            continue
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edgelist=[(int(row.u), int(row.v)) for row in impacted.itertuples(index=False)],
            edge_color=color,
            width=[edge_width_lookup[row.edge_key] + 1.4 for row in impacted.itertuples(index=False)],
            alpha=0.95,
        )

    for location in location_points:
        ax.add_patch(
            Circle(
                (location["x"], location["y"]),
                budget_summary["location_link_radius_km"] * 1000.0,
                facecolor="none",
                edgecolor="#546e7a",
                linewidth=1.0,
                linestyle=":",
                alpha=0.35,
                zorder=5,
            )
        )
        ax.scatter(location["x"], location["y"], marker="s", s=80, c="#37474f", edgecolors="white", zorder=7)

    for strike in strike_points:
        radius_color = "#8e24aa" if strike["attack_type"] == "missile" else "#00897b"
        ax.add_patch(
            Circle(
                (strike["x"], strike["y"]),
                strike["radius_km"] * 1000.0,
                facecolor="none",
                edgecolor=radius_color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                zorder=6,
            )
        )
        marker = "X" if strike["attack_type"] == "missile" else "o"
        size = 110 if strike["attack_type"] == "missile" else 70
        ax.scatter(strike["x"], strike["y"], marker=marker, s=size, c=radius_color, edgecolors="black", zorder=7)

    legend_items = [
        Line2D([0], [0], color="#c7c7c7", lw=2, label="Base graph"),
        Line2D([0], [0], color="#fdd835", lw=3, label="Struck, unchanged"),
        Line2D([0], [0], color="#fb8c00", lw=3, label="Degraded"),
        Line2D([0], [0], color="#c62828", lw=3, label="Removed"),
        Line2D([0], [0], color="#546e7a", lw=2, ls=":", label="Location reach (100 km)"),
        Line2D([0], [0], color="#8e24aa", lw=2, ls="--", label="Missile radius"),
        Line2D([0], [0], color="#00897b", lw=2, ls="--", label="Bomb radius"),
        Line2D([0], [0], marker="s", color="#37474f", lw=0, markersize=8, label="Chosen location"),
        Line2D([0], [0], marker="X", color="#8e24aa", lw=0, markersize=10, label="Missile strike point"),
        Line2D([0], [0], marker="o", color="#00897b", lw=0, markersize=8, label="Bomb strike point"),
    ]
    ax.legend(handles=legend_items, loc="lower left")
    ax.set_title("Attack Footprint on Base Graph")
    ax.set_axis_off()


def _draw_scenario_panel(ax, graph, positions, edge_impacts, edge_width_lookup, location_points, strike_points, geography) -> None:
    _set_extent(ax, positions)
    _draw_geography(ax, geography)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edgelist=list(graph.edges()),
        edge_color="#9e9e9e",
        width=[edge_width_lookup.get(f"{min(u,v)}--{max(u,v)}", 1.0) for u, v in graph.edges()],
        alpha=0.42,
    )
    degraded = edge_impacts[edge_impacts["status"] == "degraded"]
    if not degraded.empty:
        surviving_degraded = [(int(row.u), int(row.v)) for row in degraded.itertuples(index=False) if graph.has_edge(int(row.u), int(row.v))]
        if surviving_degraded:
            nx.draw_networkx_edges(
                graph,
                positions,
                ax=ax,
                edgelist=surviving_degraded,
                edge_color="#ef6c00",
                width=[edge_width_lookup[f"{min(u,v)}--{max(u,v)}"] + 1.4 for u, v in surviving_degraded],
                alpha=0.96,
            )
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_size=[14 + 3.0 * np.sqrt(max(1.0, graph.nodes[n].get("member_count", 1.0))) for n in graph.nodes()],
        node_color="#455a64",
        alpha=0.82,
        linewidths=0.0,
    )
    for location in location_points:
        ax.scatter(location["x"], location["y"], marker="s", s=70, c="#37474f", edgecolors="white", zorder=6)
    for strike in strike_points:
        marker = "X" if strike["attack_type"] == "missile" else "o"
        size = 90 if strike["attack_type"] == "missile" else 55
        color = "#8e24aa" if strike["attack_type"] == "missile" else "#00897b"
        ax.scatter(strike["x"], strike["y"], marker=marker, s=size, c=color, edgecolors="black", zorder=6)
    legend_items = [
        Line2D([0], [0], color="#9e9e9e", lw=2, label="Surviving edge"),
        Line2D([0], [0], color="#ef6c00", lw=3, label="Degraded edge"),
        Line2D([0], [0], marker="s", color="#37474f", lw=0, markersize=8, label="Chosen location"),
        Line2D([0], [0], marker="X", color="#8e24aa", lw=0, markersize=10, label="Missile strike point"),
        Line2D([0], [0], marker="o", color="#00897b", lw=0, markersize=8, label="Bomb strike point"),
    ]
    ax.legend(handles=legend_items, loc="lower left")
    ax.set_title("Scenario Graph After Edge Removals")
    ax.set_axis_off()


def _project_positions(graph: nx.Graph) -> dict:
    node_ids = list(graph.nodes())
    node_points = gpd.GeoSeries(
        gpd.points_from_xy(
            [graph.nodes[n]["lon"] for n in node_ids],
            [graph.nodes[n]["lat"] for n in node_ids],
        ),
        crs="EPSG:4326",
    ).to_crs(3857)
    return {int(node): (float(node_points.iloc[idx].x), float(node_points.iloc[idx].y)) for idx, node in enumerate(node_ids)}


def _set_extent(ax, positions: dict) -> None:
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    if not xs or not ys:
        return
    x_pad = (max(xs) - min(xs)) * 0.06 if max(xs) > min(xs) else 1000.0
    y_pad = (max(ys) - min(ys)) * 0.06 if max(ys) > min(ys) else 1000.0
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_aspect("equal", adjustable="box")


def _draw_geography(ax, geography) -> None:
    if not geography:
        return
    geography["sovereign_shape"].boundary.to_crs(3857).plot(ax=ax, color="black", linewidth=0.9, alpha=0.7, zorder=0)
    geography["frontline_boundary_gs"].to_crs(3857).plot(ax=ax, color="#c62828", linewidth=1.2, alpha=0.75, zorder=0)
    geography["ukraine_russia_border_gs"].to_crs(3857).plot(ax=ax, color="#1565c0", linewidth=1.2, alpha=0.75, zorder=0)


def _project_strike_points(strike_events: list[dict]) -> list[dict]:
    return _project_points(strike_events)


def _project_points(items: list[dict]) -> list[dict]:
    if not items:
        return []
    gseries = gpd.GeoSeries(
        gpd.points_from_xy([item["lon"] for item in items], [item["lat"] for item in items]),
        crs="EPSG:4326",
    ).to_crs(3857)
    projected = []
    for idx, item in enumerate(items):
        projected.append(
            {
                **item,
                "x": float(gseries.iloc[idx].x),
                "y": float(gseries.iloc[idx].y),
            }
        )
    return projected
