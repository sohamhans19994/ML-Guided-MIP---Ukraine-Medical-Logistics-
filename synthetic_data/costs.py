from __future__ import annotations

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


def compute_frontline_cost_component(
    distance_km: float,
    is_ukraine_side: bool,
    band_cfg: dict,
    value_cfg: dict,
) -> float:
    if not is_ukraine_side:
        return float(value_cfg["outside_border_flat"])

    near_km = float(band_cfg["near_km"])
    mid_km = float(band_cfg["mid_km"])
    far_scale_km = float(band_cfg["far_log_scale_km"])

    near_floor = float(value_cfg["near_floor"])
    near_ceiling = float(value_cfg["near_ceiling"])
    mid_floor = float(value_cfg["mid_floor"])
    far_floor = float(value_cfg["far_floor"])

    distance_km = max(float(distance_km), 0.0)

    if distance_km <= near_km:
        t = distance_km / max(near_km, 1e-9)
        return near_floor + (near_ceiling - near_floor) * ((1.0 - t) ** 3)

    if distance_km <= mid_km:
        t = (distance_km - near_km) / max(mid_km - near_km, 1e-9)
        return mid_floor + (near_floor - mid_floor) * (1.0 - t)

    return far_floor / (1.0 + np.log1p((distance_km - mid_km) / max(far_scale_km, 1e-9)))


def compute_costs(coarse_graph, config: dict):
    cost_cfg = config["costs"]
    member_count = {n: coarse_graph.nodes[n].get("member_count", 1) for n in coarse_graph.nodes()}
    incident_edge_support = {
        n: sum(data.get("abstracted_path_count", 1) for _, _, data in coarse_graph.edges(n, data=True))
        for n in coarse_graph.nodes()
    }
    territory_frontline_distance_km = {
        n: coarse_graph.nodes[n].get("territory_frontline_distance_km", np.inf) for n in coarse_graph.nodes()
    }
    territory_is_ukraine_side = {
        n: bool(coarse_graph.nodes[n].get("territory_is_ukraine_side", True)) for n in coarse_graph.nodes()
    }

    norm_member = normalize(member_count)
    norm_support = normalize(incident_edge_support)

    member_cost_component = {n: 1.0 - norm_member[n] for n in coarse_graph.nodes()}
    edge_support_cost_component = {n: 1.0 - norm_support[n] for n in coarse_graph.nodes()}

    raw_frontline_cost_component = {
        n: compute_frontline_cost_component(
            territory_frontline_distance_km[n],
            territory_is_ukraine_side[n],
            cost_cfg["frontline_distance_bands_km"],
            cost_cfg["frontline_component_values"],
        )
        for n in coarse_graph.nodes()
    }
    frontline_cost_component = normalize(raw_frontline_cost_component)

    weight_cfg = cost_cfg["score_weights"]
    cost_score = {
        n: (
            float(weight_cfg["small_cluster"]) * member_cost_component[n]
            + float(weight_cfg["low_edge_support"]) * edge_support_cost_component[n]
            + float(weight_cfg["frontline_danger"]) * frontline_cost_component[n]
        )
        for n in coarse_graph.nodes()
    }

    alpha = float(cost_cfg["alpha"])
    beta = float(cost_cfg["beta"])

    node_params = {
        n: {
            "a_i": round(alpha * cost_score[n], 3),
            "b_i": round(beta * cost_score[n], 3),
        }
        for n in coarse_graph.nodes()
    }

    nx_attrs_a = {n: values["a_i"] for n, values in node_params.items()}
    nx_attrs_b = {n: values["b_i"] for n, values in node_params.items()}
    import networkx as nx

    nx.set_node_attributes(coarse_graph, nx_attrs_a, "a_i")
    nx.set_node_attributes(coarse_graph, nx_attrs_b, "b_i")
    nx.set_node_attributes(coarse_graph, member_count, "member_count")
    nx.set_node_attributes(coarse_graph, incident_edge_support, "incident_edge_support")
    nx.set_node_attributes(coarse_graph, territory_frontline_distance_km, "territory_frontline_distance_km")
    nx.set_node_attributes(coarse_graph, territory_is_ukraine_side, "territory_is_ukraine_side")
    nx.set_node_attributes(coarse_graph, member_cost_component, "member_cost_component")
    nx.set_node_attributes(coarse_graph, edge_support_cost_component, "edge_support_cost_component")
    nx.set_node_attributes(coarse_graph, raw_frontline_cost_component, "raw_frontline_cost_component")
    nx.set_node_attributes(coarse_graph, frontline_cost_component, "frontline_cost_component")
    nx.set_node_attributes(coarse_graph, cost_score, "cost_score")

    return {
        "node_params": node_params,
        "member_count": member_count,
        "incident_edge_support": incident_edge_support,
        "territory_frontline_distance_km": territory_frontline_distance_km,
        "territory_is_ukraine_side": territory_is_ukraine_side,
        "member_cost_component": member_cost_component,
        "edge_support_cost_component": edge_support_cost_component,
        "raw_frontline_cost_component": raw_frontline_cost_component,
        "frontline_cost_component": frontline_cost_component,
        "cost_score": cost_score,
    }
