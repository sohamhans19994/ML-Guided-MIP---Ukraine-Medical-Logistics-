from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN


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

            cluster_info[cid] = {
                "zone": zone,
                "lat": float(np.mean([graph.nodes[n]["y"] for n in members])),
                "lon": float(np.mean([graph.nodes[n]["x"] for n in members])),
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

    for u, v, data in graph.edges(data=True):
        cu = node_to_cluster.get(u)
        cv = node_to_cluster.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        travel_time = data.get("travel_time", float("inf"))
        if coarse_graph.has_edge(cu, cv):
            coarse_graph[cu][cv]["abstracted_path_count"] += 1
            if travel_time < coarse_graph[cu][cv]["travel_time"]:
                coarse_graph[cu][cv]["travel_time"] = travel_time
        else:
            coarse_graph.add_edge(cu, cv, travel_time=travel_time, abstracted_path_count=1)

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
