from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

from .coarsen import adaptive_coarsen_graph, map_demand_to_coarse_nodes
from .config import load_config
from .costs import compute_costs
from .demand import load_demand_nodes, snap_demand_nodes_to_graph
from .occupied import (
    annotate_graph_with_border_metrics,
    clip_graph_to_sovereign_border,
    load_current_occupied_snapshot,
    load_ukraine_sovereign_geometry,
)
from .raw_graph import load_raw_graph
from .utils import (
    ensure_dirs,
    read_pickle,
    sanitize_graph_for_graphml,
    to_serializable,
    write_json,
    write_pickle,
    write_yaml,
)
from .visualization import plot_adaptive_edge_metrics, plot_adaptive_graph, plot_cost_score, plot_occupied_filter


def build_synthetic_dataset(config_path: str | Path | None = None, generate_visuals: bool | None = None):
    config = load_config(config_path)
    ensure_dirs(config)

    raw_graph = load_raw_graph(config)

    occupied = load_current_occupied_snapshot(config)
    borders = load_ukraine_sovereign_geometry(config, occupied["occupied_geom"])
    clipped_graph, clip_debug = clip_graph_to_sovereign_border(
        raw_graph,
        borders["sovereign_geom"],
        occupied["occupied_geom"],
        config,
    )
    filtered_graph, filter_debug = annotate_graph_with_border_metrics(
        clipped_graph,
        borders["sovereign_geom"],
        occupied["occupied_geom"],
        borders["sovereign_border_metric"],
        borders["frontline_boundary_metric"],
        borders["exterior_boundary_metric"],
        clip_debug,
        config,
    )

    acled_events, demand_nodes, demand_medoid_indices = load_demand_nodes(config)
    snapped_demand_nodes = snap_demand_nodes_to_graph(demand_nodes, filtered_graph)

    coarse_graph, raw_dist_to_demand, raw_zone_membership, raw_node_to_cluster = adaptive_coarsen_graph(
        filtered_graph,
        snapped_demand_nodes[["lat", "lon"]],
        config["adaptive_coarsening"],
    )
    coarse_demand_nodes = map_demand_to_coarse_nodes(snapped_demand_nodes, coarse_graph, raw_node_to_cluster)

    cost_details = compute_costs(coarse_graph, config)

    snapshot_date = occupied["snapshot_date"]
    snapshot_label = snapshot_date.strftime("%Y-%m-%d") if snapshot_date is not None else "undated"
    edge_path_counts = [data.get("abstracted_path_count", 1) for _, _, data in coarse_graph.edges(data=True)] or [1]
    edge_lengths_km = [
        float(data.get("length_m", float("nan"))) / 1000.0
        for _, _, data in coarse_graph.edges(data=True)
        if pd.notna(data.get("length_m", float("nan")))
    ]
    edge_times_hr = [
        float(data.get("travel_time", float("nan"))) / 3600.0
        for _, _, data in coarse_graph.edges(data=True)
        if pd.notna(data.get("travel_time", float("nan")))
    ]

    summary = {
        "snapshot_date": snapshot_label,
        "raw_graph": {
            "nodes": raw_graph.number_of_nodes(),
            "edges": raw_graph.number_of_edges(),
        },
        "filtered_graph": {
            "nodes": filtered_graph.number_of_nodes(),
            "edges": filtered_graph.number_of_edges(),
            "removed_outside_new_border_nodes": len(filter_debug["removed_outside_new_border_nodes"]),
            "removed_occupied_nodes": len(filter_debug["removed_occupied_nodes"]),
            "removed_russia_side_nodes": len(filter_debug["removed_russia_side_nodes"]),
        },
        "adaptive_graph": {
            "nodes": coarse_graph.number_of_nodes(),
            "edges": coarse_graph.number_of_edges(),
            "raw_zone_counts": {zone: len(nodes) for zone, nodes in raw_zone_membership.items()},
            "coarse_zone_counts": {
                zone: int(pd.Series([coarse_graph.nodes[n]["zone"] for n in coarse_graph.nodes()]).value_counts().get(zone, 0))
                for zone in ["near", "mid", "far"]
            },
            "abstracted_path_count": {
                "min": min(edge_path_counts),
                "mean": float(sum(edge_path_counts) / len(edge_path_counts)),
                "max": max(edge_path_counts),
            },
            "road_length_km": _finite_list_range(edge_lengths_km),
            "travel_time_hr": _finite_list_range(edge_times_hr),
        },
        "demand_nodes": {
            "count": len(coarse_demand_nodes),
            "unique_coarse_nodes": int(coarse_demand_nodes["coarse_node"].nunique()),
            "medoid_indices": [int(idx) for idx in demand_medoid_indices],
        },
        "costs": {
            "a_i": _cost_range(cost_details["node_params"], "a_i"),
            "b_i": _cost_range(cost_details["node_params"], "b_i"),
            "cost_score": _mapping_range(cost_details["cost_score"]),
        },
    }

    paths = config["paths"]
    ox.save_graphml(sanitize_graph_for_graphml(filtered_graph), paths["filtered_raw_graph"])
    nx.write_graphml(sanitize_graph_for_graphml(coarse_graph), paths["adaptive_graph"])
    coarse_demand_nodes.to_csv(paths["demand_nodes_csv"], index=False)
    write_json(paths["summary_json"], summary)
    write_yaml(paths["config_used_yaml"], config)

    bundle = {
        "config": to_serializable(config),
        "summary": summary,
        "graphs": {
            "raw_graph": raw_graph,
            "filtered_graph": filtered_graph,
            "adaptive_graph": coarse_graph,
        },
        "demand_nodes": coarse_demand_nodes,
        "acled_events": acled_events,
        "metadata": {
            "raw_dist_to_demand": raw_dist_to_demand,
            "raw_zone_membership": raw_zone_membership,
            "raw_node_to_cluster": raw_node_to_cluster,
            "cost_details": cost_details,
        },
    }
    write_pickle(paths["bundle_pickle"], bundle)

    make_visuals = bool(config["visualization"]["enabled"]) if generate_visuals is None else bool(generate_visuals)
    if make_visuals:
        plot_occupied_filter(
            raw_graph=raw_graph,
            filtered_graph=filtered_graph,
            filter_debug=filter_debug,
            ukraine_shape=borders["sovereign_shape"],
            land_only_border_gs=borders["sovereign_border_gs"],
            occupied_gs=occupied["occupied_gs"],
            save_path=paths["occupied_filter_figure"],
            config=config,
        )
        plot_adaptive_graph(
            coarse_graph=coarse_graph,
            demand_nodes=coarse_demand_nodes,
            ukraine_shape=borders["sovereign_shape"],
            occupied_gs=occupied["occupied_gs"],
            save_path=paths["adaptive_figure"],
            config=config,
        )
        plot_adaptive_edge_metrics(
            coarse_graph=coarse_graph,
            demand_nodes=coarse_demand_nodes,
            ukraine_shape=borders["sovereign_shape"],
            occupied_gs=occupied["occupied_gs"],
            save_path=paths["edge_metric_figure"],
            config=config,
        )
        plot_cost_score(
            coarse_graph=coarse_graph,
            demand_nodes=coarse_demand_nodes.drop_duplicates(subset=["coarse_node"]).copy(),
            cost_score=cost_details["cost_score"],
            member_count=cost_details["member_count"],
            ukraine_shape=borders["sovereign_shape"],
            occupied_gs=occupied["occupied_gs"],
            save_path=paths["cost_figure"],
            config=config,
        )

    return bundle


def load_saved_synthetic_bundle(bundle_path: str | Path | None = None):
    config = load_config()
    path = Path(bundle_path or config["paths"]["bundle_pickle"])
    return read_pickle(path)


def _cost_range(node_params: dict, key: str) -> dict:
    values = [details[key] for details in node_params.values()]
    return {
        "min": min(values),
        "mean": float(sum(values) / len(values)),
        "max": max(values),
    }


def _mapping_range(mapping: dict) -> dict:
    values = list(mapping.values())
    return {
        "min": min(values),
        "mean": float(sum(values) / len(values)),
        "max": max(values),
    }


def _finite_list_range(values: list[float]) -> dict:
    finite_values = [float(value) for value in values if pd.notna(value)]
    if not finite_values:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": min(finite_values),
        "mean": float(sum(finite_values) / len(finite_values)),
        "max": max(finite_values),
    }
