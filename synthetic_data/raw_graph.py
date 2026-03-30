from __future__ import annotations

from pathlib import Path

import osmnx as ox

from .utils import migrate_if_missing


def load_raw_graph(config: dict):
    paths = config["paths"]
    raw_cfg = config["raw_graph"]
    cache_path = Path(paths["raw_graph_cache"])

    migrate_if_missing(
        cache_path,
        [
            config["_project_root"] / "cache" / "ukraine_major_roads.graphml",
        ],
    )

    ox.settings.max_query_area_size = int(raw_cfg["max_query_area_size"])

    if cache_path.exists():
        graph = ox.load_graphml(cache_path)
    else:
        graph = ox.graph_from_place(
            config["data_sources"]["ukraine_place"],
            network_type=raw_cfg["network_type"],
            custom_filter=raw_cfg["custom_filter"],
        )
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, cache_path)

    return graph
