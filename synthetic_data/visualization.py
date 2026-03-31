from __future__ import annotations

from pathlib import Path

import contextily as ctx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox

from .utils import ensure_parent


def _maybe_add_basemap(ax, use_basemap: bool, zoom: int):
    if not use_basemap:
        return
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.CartoDB.Positron,
            zoom=zoom,
            alpha=0.8,
        )
    except Exception as exc:
        print(f"Basemap unavailable, continuing without tiles: {exc}")


def plot_occupied_filter(
    raw_graph,
    filtered_graph,
    filter_debug,
    ukraine_shape,
    land_only_border_gs,
    occupied_gs,
    save_path: str | Path,
    config: dict,
):
    fig_cfg = config["visualization"]
    figsize = tuple(fig_cfg["occupied_filter_figsize"])

    base_sample = ox.convert.to_undirected(raw_graph)
    if base_sample.number_of_nodes() > 0:
        base_sample = base_sample.subgraph(max(nx.connected_components(base_sample), key=len)).copy()

    filtered_sample = ox.convert.to_undirected(filtered_graph)
    if filtered_sample.number_of_nodes() > 0:
        filtered_sample = filtered_sample.subgraph(max(nx.connected_components(filtered_sample), key=len)).copy()

    base_pos = {n: (data["x"], data["y"]) for n, data in base_sample.nodes(data=True)}
    filtered_pos = {n: (data["x"], data["y"]) for n, data in filtered_sample.nodes(data=True)}

    removed_nodes = [n for n in filter_debug["removed_outside_new_border_nodes"] if n in base_sample.nodes()]
    kept_occupied_nodes = [
        n for n in filtered_sample.nodes() if filtered_sample.nodes[n].get("territory_occupation_fraction", 0.0) > 0.5
    ]
    removed_pos = {n: base_pos[n] for n in removed_nodes}
    kept_pos = {n: filtered_pos[n] for n in kept_occupied_nodes}

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        (axes[0], base_sample, base_pos, "Raw major-road graph with sovereign-held Ukraine overlay"),
        (
            axes[1],
            filtered_sample,
            filtered_pos,
            f"Road graph clipped to sovereign-held Ukraine (+{config['occupied_filter']['outside_new_border_tolerance_km']:.0f} km tolerance)",
        ),
    ]

    for ax, graph, pos, title in panels:
        ukraine_shape.boundary.plot(ax=ax, color="black", linewidth=1.0, alpha=0.8, zorder=1)
        land_only_border_gs.plot(ax=ax, color="navy", linewidth=1.0, alpha=0.5, zorder=2)
        occupied_gs.boundary.plot(ax=ax, color="crimson", linewidth=1.2, alpha=0.9, zorder=3)
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="gray", alpha=0.16, width=0.5)

        if ax is axes[1] and removed_pos:
            ax.scatter(
                [removed_pos[n][0] for n in removed_nodes],
                [removed_pos[n][1] for n in removed_nodes],
                s=10,
                c="crimson",
                alpha=0.65,
                zorder=4,
                label="Removed outside new border",
            )
        if ax is axes[1] and kept_pos:
            ax.scatter(
                [kept_pos[n][0] for n in kept_occupied_nodes],
                [kept_pos[n][1] for n in kept_occupied_nodes],
                s=9,
                c="orange",
                alpha=0.55,
                zorder=5,
                label="Kept occupied-adjacent nodes",
            )

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower left")

    plt.suptitle("Ukraine Road Graph Clipped to the Sovereign-Held Border", fontsize=14)
    plt.tight_layout()
    ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adaptive_graph(
    coarse_graph,
    demand_nodes,
    ukraine_shape,
    occupied_gs,
    save_path: str | Path,
    config: dict,
):
    fig_cfg = config["visualization"]
    zone_colors = fig_cfg["zone_colors"]
    adaptive_pos = {n: (data["lon"], data["lat"]) for n, data in coarse_graph.nodes(data=True)}
    minx, miny, maxx, maxy = ukraine_shape.total_bounds
    x_pad = (maxx - minx) * 0.03
    y_pad = (maxy - miny) * 0.03

    fig, ax = plt.subplots(figsize=tuple(fig_cfg["adaptive_figsize"]))
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)
    _maybe_add_basemap(ax, bool(fig_cfg["use_basemap"]), int(fig_cfg["basemap_zoom"]))

    ukraine_shape.boundary.plot(ax=ax, color="black", linewidth=1.0, alpha=0.8, zorder=1)
    occupied_gs.boundary.plot(ax=ax, color="crimson", linewidth=1.1, alpha=0.85, zorder=2)

    edge_list = list(coarse_graph.edges(data=True))
    edge_path_counts = [data.get("abstracted_path_count", 1) for _, _, data in edge_list] or [1]
    edge_widths = [
        0.6 + 4.4 * np.sqrt(data.get("abstracted_path_count", 1)) / np.sqrt(max(edge_path_counts))
        for _, _, data in edge_list
    ]
    edge_collection = nx.draw_networkx_edges(
        coarse_graph,
        adaptive_pos,
        ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        alpha=0.28,
        edge_color="gray",
        width=edge_widths,
    )
    edge_collection.set_zorder(3)

    node_sizes = [20 + 4 * np.sqrt(coarse_graph.nodes[n]["member_count"]) for n in coarse_graph.nodes()]
    node_colors = [zone_colors[coarse_graph.nodes[n]["zone"]] for n in coarse_graph.nodes()]
    node_collection = nx.draw_networkx_nodes(
        coarse_graph,
        adaptive_pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
    )
    node_collection.set_zorder(4)

    demand_plot = demand_nodes.drop_duplicates(subset=["coarse_node"])
    ax.scatter(
        demand_plot["plot_lon"].to_list(),
        demand_plot["plot_lat"].to_list(),
        marker="*",
        c="blue",
        s=170,
        zorder=5,
        label="Demand node (snapped)",
    )

    for zone in ["near", "mid", "far"]:
        ax.scatter([], [], c=zone_colors[zone], s=80, label=f"{zone} zone")
    ax.plot([], [], color="gray", linewidth=0.8, alpha=0.5, label="few abstracted paths")
    ax.plot([], [], color="gray", linewidth=5.0, alpha=0.5, label="many abstracted paths")

    ax.set_title("Adaptive Coarsening After Occupied-Area Filtering")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left")
    plt.tight_layout()
    ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_adaptive_edge_metrics(
    coarse_graph,
    demand_nodes,
    ukraine_shape,
    occupied_gs,
    save_path: str | Path,
    config: dict,
):
    fig_cfg = config["visualization"]
    zone_colors = fig_cfg["zone_colors"]
    adaptive_pos = {n: (data["lon"], data["lat"]) for n, data in coarse_graph.nodes(data=True)}
    minx, miny, maxx, maxy = ukraine_shape.total_bounds
    x_pad = (maxx - minx) * 0.03
    y_pad = (maxy - miny) * 0.03

    fig, ax = plt.subplots(figsize=tuple(fig_cfg["adaptive_figsize"]))
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)
    _maybe_add_basemap(ax, bool(fig_cfg["use_basemap"]), int(fig_cfg["basemap_zoom"]))

    ukraine_shape.boundary.plot(ax=ax, color="black", linewidth=1.0, alpha=0.8, zorder=1)
    occupied_gs.boundary.plot(ax=ax, color="crimson", linewidth=1.1, alpha=0.85, zorder=2)

    edge_list = list(coarse_graph.edges(data=True))
    edge_path_counts = [data.get("abstracted_path_count", 1) for _, _, data in edge_list] or [1]
    edge_widths = [
        0.6 + 4.4 * np.sqrt(data.get("abstracted_path_count", 1)) / np.sqrt(max(edge_path_counts))
        for _, _, data in edge_list
    ]
    edge_collection = nx.draw_networkx_edges(
        coarse_graph,
        adaptive_pos,
        ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        alpha=0.30,
        edge_color="gray",
        width=edge_widths,
    )
    edge_collection.set_zorder(3)

    node_sizes = [20 + 4 * np.sqrt(coarse_graph.nodes[n]["member_count"]) for n in coarse_graph.nodes()]
    node_colors = [zone_colors[coarse_graph.nodes[n]["zone"]] for n in coarse_graph.nodes()]
    node_collection = nx.draw_networkx_nodes(
        coarse_graph,
        adaptive_pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.2,
    )
    node_collection.set_zorder(4)

    demand_plot = demand_nodes.drop_duplicates(subset=["coarse_node"])
    ax.scatter(
        demand_plot["plot_lon"].to_list(),
        demand_plot["plot_lat"].to_list(),
        marker="*",
        c="royalblue",
        s=150,
        zorder=5,
        label="Demand node (snapped)",
    )

    edge_labels = {}
    for u, v, data in edge_list:
        travel_hr = data.get("travel_time", np.nan) / 3600.0 if np.isfinite(data.get("travel_time", np.nan)) else np.nan
        road_km = data.get("length_m", np.nan) / 1000.0 if np.isfinite(data.get("length_m", np.nan)) else np.nan
        travel_str = f"{travel_hr:.2f}h" if np.isfinite(travel_hr) else "NAh"
        road_str = f"{road_km:.0f}km road" if np.isfinite(road_km) else "NA road"
        edge_labels[(u, v)] = f"{travel_str}\n{road_str}"

    nx.draw_networkx_edge_labels(
        coarse_graph,
        adaptive_pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=6,
        rotate=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.15},
    )

    for zone in ["near", "mid", "far"]:
        ax.scatter([], [], c=zone_colors[zone], s=80, label=f"{zone} zone")

    ax.set_title("Adaptive Coarsening with Routed Road Distances and Times")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left")
    plt.tight_layout()
    ensure_parent(save_path)
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_cost_score(
    coarse_graph,
    demand_nodes,
    cost_score,
    member_count,
    ukraine_shape,
    occupied_gs,
    save_path: str | Path,
    config: dict,
):
    fig_cfg = config["visualization"]
    adaptive_pos = {n: (data["lon"], data["lat"]) for n, data in coarse_graph.nodes(data=True)}
    minx, miny, maxx, maxy = ukraine_shape.total_bounds
    x_pad = (maxx - minx) * 0.03
    y_pad = (maxy - miny) * 0.03

    edge_list = list(coarse_graph.edges(data=True))
    edge_path_counts = [data.get("abstracted_path_count", 1) for _, _, data in edge_list] or [1]
    edge_widths = [
        0.5 + 3.5 * np.sqrt(data.get("abstracted_path_count", 1)) / np.sqrt(max(edge_path_counts))
        for _, _, data in edge_list
    ]

    costs = [cost_score[n] for n in coarse_graph.nodes()]
    norm = mcolors.Normalize(vmin=min(costs), vmax=max(costs))
    cmap = cm.RdYlGn_r

    fig, ax = plt.subplots(1, 1, figsize=tuple(fig_cfg["cost_figsize"]))
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)
    _maybe_add_basemap(ax, bool(fig_cfg["use_basemap"]), int(fig_cfg["basemap_zoom"]))

    ukraine_shape.boundary.plot(ax=ax, color="black", linewidth=1.0, alpha=0.8, zorder=1)
    occupied_gs.boundary.plot(ax=ax, color="crimson", linewidth=1.2, alpha=0.9, zorder=2)
    edge_collection = nx.draw_networkx_edges(
        coarse_graph,
        adaptive_pos,
        ax=ax,
        edgelist=[(u, v) for u, v, _ in edge_list],
        edge_color="gray",
        alpha=0.20,
        width=edge_widths,
    )
    edge_collection.set_zorder(3)

    node_collection = nx.draw_networkx_nodes(
        coarse_graph,
        adaptive_pos,
        ax=ax,
        node_color=costs,
        cmap=cmap,
        node_size=[30 + 5 * np.sqrt(member_count[n]) for n in coarse_graph.nodes()],
        vmin=min(costs),
        vmax=max(costs),
    )
    node_collection.set_zorder(4)

    ax.scatter(
        demand_nodes["plot_lon"].to_list(),
        demand_nodes["plot_lat"].to_list(),
        marker="*",
        c="blue",
        s=150,
        zorder=5,
        label="Demand node (snapped)",
    )

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="cost_score")
    ax.set_title("Adaptive Graph Main Cost Score")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left")

    plt.tight_layout()
    ensure_parent(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
