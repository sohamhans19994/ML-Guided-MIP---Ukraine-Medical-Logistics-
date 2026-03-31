from __future__ import annotations

import networkx as nx
import numpy as np
import osmnx as ox
from sklearn.cluster import DBSCAN


def _cluster_representative_node(graph, members: list[int]) -> tuple[int, float, float]:
    centroid_lat = float(np.mean([graph.nodes[n]["y"] for n in members]))
    centroid_lon = float(np.mean([graph.nodes[n]["x"] for n in members]))
    rep_node = min(
        members,
        key=lambda n: (graph.nodes[n]["y"] - centroid_lat) ** 2 + (graph.nodes[n]["x"] - centroid_lon) ** 2,
    )
    return rep_node, centroid_lat, centroid_lon


def _routed_path_metrics(graph, source: int, target: int) -> tuple[float, float]:
    path = nx.shortest_path(graph, source, target, weight="travel_time")
    total_time = 0.0
    total_length = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_bundle = graph.get_edge_data(u, v)
        if edge_bundle is None:
            raise nx.NetworkXNoPath(f"Missing edge data for segment {(u, v)}")

        if isinstance(edge_bundle, dict) and all(isinstance(k, int) for k in edge_bundle.keys()):
            candidates = list(edge_bundle.values())
        else:
            candidates = [edge_bundle]

        finite_candidates = [
            data
            for data in candidates
            if np.isfinite(data.get("travel_time", np.nan)) and np.isfinite(data.get("length", np.nan))
        ]
        best = min(finite_candidates if finite_candidates else candidates, key=lambda data: float(data.get("travel_time", np.inf)))
        total_time += float(best.get("travel_time", np.nan))
        total_length += float(best.get("length", np.nan))

    return float(total_time), float(total_length)


def adaptive_coarsen_graph(graph, demand_points, params: dict):
    if hasattr(demand_points, "loc"):
        demand_coords = demand_points[["lat", "lon"]].to_numpy(dtype=float)
    else:
        demand_coords = np.asarray(demand_points, dtype=float)

    if len(demand_coords) == 0:
        raise ValueError("adaptive_coarsen_graph needs at least one demand point")

    node_ids = list(graph.nodes())
    coords = np.array([[graph.nodes[n]["y"], graph.nodes[n]["x"]] for n in node_ids], dtype=float)

    node_to_demand_dist = {}
    zone_to_nodes = {"near": [], "mid": [], "far": []}
    for index, node in enumerate(node_ids):
        dist = np.sqrt(((demand_coords - coords[index]) ** 2).sum(axis=1)).min()
        node_to_demand_dist[node] = float(dist)
        if dist <= float(params["near_radius"]):
            zone_to_nodes["near"].append(node)
        elif dist <= float(params["mid_radius"]):
            zone_to_nodes["mid"].append(node)
        else:
            zone_to_nodes["far"].append(node)

    zone_eps = {
        "near": float(params["eps_near"]),
        "mid": float(params["eps_mid"]),
        "far": float(params["eps_far"]),
    }
    node_to_cluster = {}
    cluster_info = {}
    cluster_id = 0

    for zone in ["near", "mid", "far"]:
        zone_nodes = zone_to_nodes[zone]
        if not zone_nodes:
            continue

        zone_coords = np.array([[graph.nodes[n]["y"], graph.nodes[n]["x"]] for n in zone_nodes], dtype=float)
        labels = DBSCAN(eps=zone_eps[zone], min_samples=1).fit(zone_coords).labels_

        for label in sorted(set(labels)):
            members = [zone_nodes[i] for i, lab in enumerate(labels) if lab == label]
            cid = cluster_id
            cluster_id += 1

            for node in members:
                node_to_cluster[node] = cid

            rep_node, centroid_lat, centroid_lon = _cluster_representative_node(graph, members)
            cluster_info[cid] = {
                "zone": zone,
                "lat": float(graph.nodes[rep_node]["y"]),
                "lon": float(graph.nodes[rep_node]["x"]),
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "rep_node": rep_node,
                "member_count": len(members),
                "mean_demand_dist": float(np.mean([node_to_demand_dist[n] for n in members])),
                "territory_occupation_fraction": float(np.mean([graph.nodes[n].get("territory_occupation_fraction", 0.0) for n in members])),
                "territory_border_risk": float(max([graph.nodes[n].get("territory_border_risk", 0.0) for n in members])),
                "territory_border_distance_km": float(min([graph.nodes[n].get("territory_border_distance_km", float("inf")) for n in members])),
                "territory_frontline_distance_km": float(min([graph.nodes[n].get("territory_frontline_distance_km", float("inf")) for n in members])),
                "territory_interior_depth_km": float(max([graph.nodes[n].get("territory_interior_depth_km", 0.0) for n in members])),
            }

    coarse_graph = nx.Graph()
    for cid, attrs in cluster_info.items():
        coarse_graph.add_node(cid, **attrs)

    routing_graph = ox.convert.to_undirected(graph)
    edge_aggregates: dict[tuple[int, int], dict[str, list[float] | int]] = {}
    for u, v, data in graph.edges(data=True):
        cu = node_to_cluster.get(u)
        cv = node_to_cluster.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        key = tuple(sorted((cu, cv)))
        agg = edge_aggregates.setdefault(key, {"travel_times": [], "lengths": [], "count": 0})
        agg["count"] += 1
        agg["travel_times"].append(float(data.get("travel_time", np.nan)))
        agg["lengths"].append(float(data.get("length", np.nan)))

    for (cu, cv), agg in edge_aggregates.items():
        rep_u = cluster_info[cu]["rep_node"]
        rep_v = cluster_info[cv]["rep_node"]
        finite_boundary_times = [value for value in agg["travel_times"] if np.isfinite(value)]
        finite_boundary_lengths = [value for value in agg["lengths"] if np.isfinite(value)]

        try:
            routed_travel_time, routed_length_m = _routed_path_metrics(routing_graph, rep_u, rep_v)
            route_source = "rep_shortest_path"
        except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
            routed_travel_time = float(np.nanmedian(finite_boundary_times)) if finite_boundary_times else float("inf")
            routed_length_m = float(np.nanmedian(finite_boundary_lengths)) if finite_boundary_lengths else float("inf")
            route_source = "boundary_median_fallback"

        coarse_graph.add_edge(
            cu,
            cv,
            travel_time=routed_travel_time,
            length_m=routed_length_m,
            abstracted_path_count=int(agg["count"]),
            metric_source=route_source,
            boundary_time_min_s=float(np.nanmin(finite_boundary_times)) if finite_boundary_times else float("inf"),
            boundary_time_mean_s=float(np.nanmean(finite_boundary_times)) if finite_boundary_times else float("inf"),
            boundary_time_median_s=float(np.nanmedian(finite_boundary_times)) if finite_boundary_times else float("inf"),
            boundary_length_min_m=float(np.nanmin(finite_boundary_lengths)) if finite_boundary_lengths else float("inf"),
            boundary_length_mean_m=float(np.nanmean(finite_boundary_lengths)) if finite_boundary_lengths else float("inf"),
            boundary_length_median_m=float(np.nanmedian(finite_boundary_lengths)) if finite_boundary_lengths else float("inf"),
        )

    coarse_graph = coarse_graph.subgraph(max(nx.connected_components(coarse_graph), key=len)).copy()
    surviving_clusters = set(coarse_graph.nodes())
    node_to_cluster = {node: cid for node, cid in node_to_cluster.items() if cid in surviving_clusters}
    return coarse_graph, node_to_demand_dist, zone_to_nodes, node_to_cluster


def map_demand_to_coarse_nodes(demand_nodes, coarse_graph, raw_node_to_cluster):
    mapped = demand_nodes.copy()
    mapped["coarse_node"] = mapped["graph_node"].map(raw_node_to_cluster)
    mapped = mapped[mapped["coarse_node"].notna()].copy()
    mapped["coarse_node"] = mapped["coarse_node"].astype(int)
    mapped["plot_lat"] = mapped["coarse_node"].map(lambda n: coarse_graph.nodes[n]["lat"])
    mapped["plot_lon"] = mapped["coarse_node"].map(lambda n: coarse_graph.nodes[n]["lon"])
    return mapped
