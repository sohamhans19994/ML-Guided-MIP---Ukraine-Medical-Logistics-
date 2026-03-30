from __future__ import annotations

import networkx as nx
import numpy as np


def normalize(mapping):
    values = np.array(list(mapping.values()), dtype=float)
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return {key: 0.5 for key in mapping}
    minimum = finite_values.min()
    maximum = finite_values.max()
    if maximum == minimum:
        return {key: 0.5 for key in mapping}

    output = {}
    for key, value in mapping.items():
        current = maximum if not np.isfinite(value) else float(value)
        output[key] = (current - minimum) / (maximum - minimum)
    return output


def _distance_to_large_hub(graph, node, hub_ids):
    node_lat = graph.nodes[node]["lat"]
    node_lon = graph.nodes[node]["lon"]
    distances = []
    for hub in hub_ids:
        if hub == node:
            continue
        hub_lat = graph.nodes[hub]["lat"]
        hub_lon = graph.nodes[hub]["lon"]
        base_dist = ((node_lat - hub_lat) ** 2 + (node_lon - hub_lon) ** 2) ** 0.5
        hub_scale = np.sqrt(graph.nodes[hub].get("member_count", 1))
        distances.append(base_dist / hub_scale)
    return min(distances) if distances else 0.0


def compute_costs(coarse_graph, config: dict):
    cost_cfg = config["costs"]
    degree = dict(coarse_graph.degree())
    centrality = nx.betweenness_centrality(coarse_graph, weight="travel_time", normalized=True)

    member_count = {n: coarse_graph.nodes[n].get("member_count", 1) for n in coarse_graph.nodes()}
    mean_demand_dist = {n: coarse_graph.nodes[n].get("mean_demand_dist", 0.0) for n in coarse_graph.nodes()}
    territory_border_risk = {n: coarse_graph.nodes[n].get("territory_border_risk", 0.0) for n in coarse_graph.nodes()}
    territory_occupation_fraction = {
        n: coarse_graph.nodes[n].get("territory_occupation_fraction", 0.0) for n in coarse_graph.nodes()
    }
    territory_frontline_distance_km = {
        n: coarse_graph.nodes[n].get("territory_frontline_distance_km", np.inf) for n in coarse_graph.nodes()
    }
    incident_edge_support = {
        n: sum(data.get("abstracted_path_count", 1) for _, _, data in coarse_graph.edges(n, data=True))
        for n in coarse_graph.nodes()
    }

    large_hub_threshold = np.quantile(list(member_count.values()), float(cost_cfg["large_hub_quantile"]))
    large_hub_ids = [n for n, count in member_count.items() if count >= large_hub_threshold]
    if not large_hub_ids:
        large_hub_ids = list(coarse_graph.nodes())

    large_hub_dist = {n: _distance_to_large_hub(coarse_graph, n, large_hub_ids) for n in coarse_graph.nodes()}

    norm_degree = normalize(degree)
    norm_centrality = normalize(centrality)
    norm_member = normalize(member_count)
    norm_large_hub_dist = normalize(large_hub_dist)
    norm_incident_support = normalize(incident_edge_support)
    norm_demand_dist = normalize(mean_demand_dist)
    norm_occ_frontline_dist = normalize(territory_frontline_distance_km)

    large_hub_proximity = {n: 1.0 - norm_large_hub_dist[n] for n in coarse_graph.nodes()}
    demand_proximity = {n: 1.0 - norm_demand_dist[n] for n in coarse_graph.nodes()}
    occupied_proximity = {n: 1.0 - norm_occ_frontline_dist[n] for n in coarse_graph.nodes()}

    threat_weights = cost_cfg["occupied_threat_weights"]
    frontline_cfg = cost_cfg["frontline_exponential"]
    decay_km = float(frontline_cfg["decay_km"])
    frontline_exponential_pressure = {
        n: (
            float(np.exp(-max(float(territory_frontline_distance_km[n]), 0.0) / decay_km))
            if np.isfinite(territory_frontline_distance_km[n])
            else 0.0
        )
        for n in coarse_graph.nodes()
    }
    occupied_threat = {
        n: (
            float(threat_weights["occupied_proximity"]) * occupied_proximity[n]
            + float(threat_weights["border_risk"]) * territory_border_risk[n]
            + float(threat_weights["occupation_fraction"]) * territory_occupation_fraction[n]
            + float(frontline_cfg["threat_weight"]) * frontline_exponential_pressure[n]
        )
        for n in coarse_graph.nodes()
    }

    scale_min = float(cost_cfg["scale_min"])
    scale_max = float(cost_cfg["scale_max"])
    scale_span = scale_max - scale_min

    a_weights = cost_cfg["a_i"]
    b_weights = cost_cfg["b_i"]
    node_params = {}
    for node in coarse_graph.nodes():
        base_a_score = (
            float(a_weights["low_degree"]) * (1.0 - norm_degree[node])
            + float(a_weights["low_centrality"]) * (1.0 - norm_centrality[node])
            + float(a_weights["small_cluster"]) * (1.0 - norm_member[node])
            + float(a_weights["far_from_large_hub"]) * (1.0 - large_hub_proximity[node])
            + float(a_weights["low_edge_support"]) * (1.0 - norm_incident_support[node])
            + float(a_weights["demand_penalty"]) * demand_proximity[node]
            + float(a_weights["occupied_penalty"]) * occupied_threat[node]
        )
        base_b_score = (
            float(b_weights["low_centrality"]) * (1.0 - norm_centrality[node])
            + float(b_weights["small_cluster"]) * (1.0 - norm_member[node])
            + float(b_weights["far_from_large_hub"]) * (1.0 - large_hub_proximity[node])
            + float(b_weights["low_edge_support"]) * (1.0 - norm_incident_support[node])
            + float(b_weights["demand_penalty"]) * demand_proximity[node]
            + float(b_weights["occupied_penalty"]) * occupied_threat[node]
        )
        base_a_cost = scale_min + scale_span * base_a_score
        base_b_cost = scale_min + scale_span * base_b_score

        a_frontline_factor = float(
            np.exp(float(frontline_cfg["a_exponent"]) * frontline_exponential_pressure[node])
        )
        b_frontline_factor = float(
            np.exp(float(frontline_cfg["b_exponent"]) * frontline_exponential_pressure[node])
        )
        node_params[node] = {
            "a_i": round(base_a_cost * a_frontline_factor, 3),
            "b_i": round(base_b_cost * b_frontline_factor, 3),
        }

    nx.set_node_attributes(coarse_graph, {n: values["a_i"] for n, values in node_params.items()}, "a_i")
    nx.set_node_attributes(coarse_graph, {n: values["b_i"] for n, values in node_params.items()}, "b_i")
    nx.set_node_attributes(coarse_graph, degree, "degree")
    nx.set_node_attributes(coarse_graph, centrality, "betweenness_centrality")
    nx.set_node_attributes(coarse_graph, member_count, "member_count")
    nx.set_node_attributes(coarse_graph, mean_demand_dist, "mean_demand_dist")
    nx.set_node_attributes(coarse_graph, territory_frontline_distance_km, "territory_frontline_distance_km")
    nx.set_node_attributes(coarse_graph, territory_border_risk, "territory_border_risk")
    nx.set_node_attributes(coarse_graph, territory_occupation_fraction, "territory_occupation_fraction")
    nx.set_node_attributes(coarse_graph, incident_edge_support, "incident_edge_support")
    nx.set_node_attributes(coarse_graph, occupied_threat, "occupied_threat")
    nx.set_node_attributes(coarse_graph, frontline_exponential_pressure, "frontline_exponential_pressure")
    nx.set_node_attributes(
        coarse_graph,
        {n: float(np.exp(float(frontline_cfg["a_exponent"]) * frontline_exponential_pressure[n])) for n in coarse_graph.nodes()},
        "a_frontline_factor",
    )
    nx.set_node_attributes(
        coarse_graph,
        {n: float(np.exp(float(frontline_cfg["b_exponent"]) * frontline_exponential_pressure[n])) for n in coarse_graph.nodes()},
        "b_frontline_factor",
    )

    return {
        "node_params": node_params,
        "degree": degree,
        "centrality": centrality,
        "member_count": member_count,
        "mean_demand_dist": mean_demand_dist,
        "territory_border_risk": territory_border_risk,
        "territory_occupation_fraction": territory_occupation_fraction,
        "territory_frontline_distance_km": territory_frontline_distance_km,
        "incident_edge_support": incident_edge_support,
        "large_hub_ids": large_hub_ids,
        "large_hub_threshold": float(large_hub_threshold),
        "large_hub_dist": large_hub_dist,
        "large_hub_proximity": large_hub_proximity,
        "demand_proximity": demand_proximity,
        "occupied_proximity": occupied_proximity,
        "occupied_threat": occupied_threat,
        "frontline_exponential_pressure": frontline_exponential_pressure,
    }
