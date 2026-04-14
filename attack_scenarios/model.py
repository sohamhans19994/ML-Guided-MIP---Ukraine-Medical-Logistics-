from __future__ import annotations

import math
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .config import ScenarioParameters, StrikeCenter


def generate_attack_bundle(
    base_graph: nx.Graph,
    demand_nodes: pd.DataFrame | None,
    params: ScenarioParameters,
    geography: dict,
) -> dict:
    if base_graph.number_of_nodes() == 0 or base_graph.number_of_edges() == 0:
        raise ValueError("Base graph must contain nodes and edges")

    graph = base_graph.copy()
    metric_positions, _ = _project_graph_nodes(graph)
    edge_table = _build_edge_table(graph, metric_positions, params)
    d_max_km = _max_frontline_depth_km(graph)
    candidates = _generate_candidate_strike_centers(edge_table, geography, params)
    selected_locations, strike_events, budget_summary = _plan_randomized_attack(
        graph=graph,
        edge_table=edge_table,
        metric_positions=metric_positions,
        geography=geography,
        params=params,
        candidates=candidates,
        d_max_km=d_max_km,
    )
    impact_table = _apply_strikes(edge_table, strike_events, params)
    scenario_graph, removed_nodes = _build_scenario_graph(graph, impact_table, params)

    base_c_ij = _build_cost_matrix_for_graph(graph).rename(columns={"c_ij": "c_ij_base"})
    c_ij = _build_cost_matrix_for_graph(scenario_graph).rename(columns={"c_ij": "c_ij_scenario"})
    c_ij = base_c_ij.merge(c_ij, on=["source", "target"], how="left")
    c_ij["c_ij_scenario"] = c_ij["c_ij_scenario"].fillna(np.inf)
    c_ij["delta_c_ij"] = c_ij["c_ij_scenario"] - c_ij["c_ij_base"]

    serializable_locations = _serializable_locations(selected_locations)
    serializable_strike_events = _serializable_strike_events(strike_events)
    summary = _build_summary(
        graph=graph,
        scenario_graph=scenario_graph,
        impact_table=impact_table,
        c_ij=c_ij,
        selected_locations=serializable_locations,
        strike_events=serializable_strike_events,
        budget_summary=budget_summary,
        params=params,
        removed_nodes=removed_nodes,
    )

    scenario_graph.graph["scenario_parameters"] = params.to_dict()
    scenario_graph.graph["selected_locations"] = serializable_locations
    scenario_graph.graph["strike_events"] = serializable_strike_events
    scenario_graph.graph["scenario_id"] = params.scenario_id

    return {
        "parameters": params.to_dict(),
        "selected_locations": serializable_locations,
        "strike_events": serializable_strike_events,
        "budget_summary": budget_summary,
        "edge_impacts": impact_table.sort_values(["status_rank", "power_total", "edge_value"], ascending=[True, False, False])
        .drop(columns=["status_rank", "line_metric"])
        .reset_index(drop=True),
        "c_ij": c_ij,
        "graphs": {
            "base_graph": graph,
            "scenario_graph": scenario_graph,
        },
        "geography": geography,
        "demand_nodes": demand_nodes.copy() if demand_nodes is not None else None,
        "summary": summary,
    }


def _project_graph_nodes(graph: nx.Graph) -> tuple[dict[int, tuple[float, float]], gpd.GeoSeries]:
    node_ids = list(graph.nodes())
    node_points = gpd.GeoSeries(
        gpd.points_from_xy([graph.nodes[n]["lon"] for n in node_ids], [graph.nodes[n]["lat"] for n in node_ids]),
        index=node_ids,
        crs="EPSG:4326",
    )
    node_points_metric = node_points.to_crs(3857)
    metric_positions = {int(n): (float(node_points_metric.loc[n].x), float(node_points_metric.loc[n].y)) for n in node_ids}
    return metric_positions, node_points_metric


def _build_edge_table(graph: nx.Graph, metric_positions: dict[int, tuple[float, float]], params: ScenarioParameters) -> pd.DataFrame:
    rows: list[dict] = []
    for u, v, data in graph.edges(data=True):
        uu = int(min(u, v))
        vv = int(max(u, v))
        ux, uy = metric_positions[uu]
        vx, vy = metric_positions[vv]
        member_avg = 0.5 * (
            float(graph.nodes[uu].get("member_count", 1.0)) + float(graph.nodes[vv].get("member_count", 1.0))
        )
        rows.append(
            {
                "u": uu,
                "v": vv,
                "edge_key": f"{uu}--{vv}",
                "u_lat": float(graph.nodes[uu]["lat"]),
                "u_lon": float(graph.nodes[uu]["lon"]),
                "v_lat": float(graph.nodes[vv]["lat"]),
                "v_lon": float(graph.nodes[vv]["lon"]),
                "mid_lat": 0.5 * (float(graph.nodes[uu]["lat"]) + float(graph.nodes[vv]["lat"])),
                "mid_lon": 0.5 * (float(graph.nodes[uu]["lon"]) + float(graph.nodes[vv]["lon"])),
                "travel_time_base": float(data.get("travel_time", np.inf)),
                "length_m": float(data.get("length_m", np.nan)),
                "abstracted_path_count": float(data.get("abstracted_path_count", 1.0)),
                "member_count_avg": float(member_avg),
                "line_metric": LineString([(ux, uy), (vx, vy)]),
            }
        )

    edge_table = pd.DataFrame(rows)
    edge_table["path_count_norm"] = _min_max_normalize(edge_table["abstracted_path_count"].to_numpy(dtype=float))
    edge_table["member_count_norm"] = _min_max_normalize(edge_table["member_count_avg"].to_numpy(dtype=float))
    edge_table["travel_time_norm"] = _min_max_normalize(edge_table["travel_time_base"].to_numpy(dtype=float))
    edge_table["defense_score"] = (
        float(params.defense_alpha) * edge_table["path_count_norm"] + float(params.defense_beta) * edge_table["member_count_norm"]
    ).clip(lower=float(params.minimum_defense_score))
    edge_table["edge_value"] = (
        0.60 * edge_table["travel_time_norm"] + 0.25 * edge_table["path_count_norm"] + 0.15 * edge_table["member_count_norm"]
    )
    return edge_table


def _max_frontline_depth_km(graph: nx.Graph) -> float:
    return max(
        float(attrs.get("territory_frontline_distance_km", 0.0))
        for _, attrs in graph.nodes(data=True)
        if math.isfinite(float(attrs.get("territory_frontline_distance_km", 0.0)))
    )


def _plan_randomized_attack(
    graph: nx.Graph,
    edge_table: pd.DataFrame,
    metric_positions: dict[int, tuple[float, float]],
    geography: dict,
    params: ScenarioParameters,
    candidates: list[dict],
    d_max_km: float,
) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(params.budget.random_seed)
    remaining_budget = float(params.base_budget)
    selected_locations: list[dict] = []
    strike_events: list[dict] = []
    current_impacts = _apply_strikes(edge_table, strike_events, params)

    manual_locations = _collect_manual_locations(
        graph=graph,
        metric_positions=metric_positions,
        geography=geography,
        params=params,
        d_max_km=d_max_km,
    )
    for location in manual_locations:
        if remaining_budget < params.budget.location_cost or len(selected_locations) >= params.budget.max_locations:
            break
        if _violates_separation(
            location["point_metric"],
            selected_locations,
            params.budget.min_location_separation_km,
        ):
            continue
        selected_locations.append(location)
        remaining_budget -= params.budget.location_cost

    while True:
        feasible_actions = _feasible_actions(params, remaining_budget, selected_locations, strike_events, candidates)
        if not feasible_actions:
            break
        action_name = _sample_action(feasible_actions, rng, selected_locations)

        if action_name == "location":
            location = _choose_random_location_candidate(candidates, selected_locations, params, rng)
            if location is None:
                break
            selected_locations.append(
                _build_location_record(location, graph, metric_positions, geography, params, d_max_km, len(selected_locations) + 1)
            )
            remaining_budget -= params.budget.location_cost
            continue

        strike_event = _choose_random_strike_event(
            attack_type=action_name,
            selected_locations=selected_locations,
            existing_strike_events=strike_events,
            candidates=candidates,
            graph=graph,
            metric_positions=metric_positions,
            geography=geography,
            params=params,
            d_max_km=d_max_km,
            rng=rng,
        )
        if strike_event is None:
            break
        strike_events.append(strike_event)
        current_impacts = _apply_strikes(edge_table, strike_events, params)
        remaining_budget -= params.budget.missile_cost if action_name == "missile" else params.budget.bomb_cost

        if _all_strike_caps_reached(params, strike_events):
            break

    return selected_locations, strike_events, _budget_summary(params, selected_locations, strike_events, remaining_budget)


def _collect_manual_locations(
    graph: nx.Graph,
    metric_positions: dict[int, tuple[float, float]],
    geography: dict,
    params: ScenarioParameters,
    d_max_km: float,
) -> list[dict]:
    manual_centers: list[StrikeCenter] = []
    for attack_type in params.active_attack_types():
        for center in getattr(params, attack_type).manual_strike_centers:
            if not any(math.isclose(center.lat, seen.lat) and math.isclose(center.lon, seen.lon) for seen in manual_centers):
                manual_centers.append(center)

    manual_locations: list[dict] = []
    for idx, center in enumerate(manual_centers, start=1):
        point_metric = gpd.GeoSeries([Point(center.lon, center.lat)], crs="EPSG:4326").to_crs(3857).iloc[0]
        manual_locations.append(
            _annotate_location_point(
                point_metric=point_metric,
                lat=float(center.lat),
                lon=float(center.lon),
                label=f"manual_location_{idx}",
                source="manual",
                graph=graph,
                metric_positions=metric_positions,
                geography=geography,
                params=params,
                d_max_km=d_max_km,
                location_index=idx,
            )
        )
    return manual_locations


def _feasible_actions(
    params: ScenarioParameters,
    remaining_budget: float,
    selected_locations: list[dict],
    strike_events: list[dict],
    candidates: list[dict],
) -> list[str]:
    actions: list[str] = []
    if (
        remaining_budget >= params.budget.location_cost
        and len(selected_locations) < params.budget.max_locations
        and candidates
    ):
        actions.append("location")
    if (
        "missile" in params.active_attack_types()
        and remaining_budget >= params.budget.missile_cost
        and len(selected_locations) > 0
        and _strike_count(strike_events, "missile") < params.missile.max_strikes
    ):
        actions.append("missile")
    if (
        "bomb" in params.active_attack_types()
        and remaining_budget >= params.budget.bomb_cost
        and len(selected_locations) > 0
        and _strike_count(strike_events, "bomb") < params.bomb.max_strikes
    ):
        actions.append("bomb")
    return actions


def _sample_action(feasible_actions: list[str], rng, selected_locations: list[dict]) -> str:
    base_weights = {
        "location": 1.3 if not selected_locations else 0.8,
        "missile": 1.0,
        "bomb": 1.1,
    }
    weights = np.asarray([base_weights[action] for action in feasible_actions], dtype=float)
    probabilities = weights / weights.sum()
    return str(rng.choice(feasible_actions, p=probabilities))


def _choose_random_location_candidate(candidates: list[dict], selected_locations: list[dict], params: ScenarioParameters, rng):
    feasible = [
        candidate
        for candidate in candidates
        if not _violates_separation(
            candidate["point_metric"],
            selected_locations,
            params.budget.min_location_separation_km,
        )
    ]
    if not feasible:
        return None
    weights = np.asarray([max(1e-6, candidate["location_score"]) for candidate in feasible], dtype=float)
    probabilities = weights / weights.sum()
    return feasible[int(rng.choice(np.arange(len(feasible)), p=probabilities))]


def _build_location_record(candidate: dict, graph, metric_positions, geography, params, d_max_km, location_index: int) -> dict:
    return _annotate_location_point(
        point_metric=candidate["point_metric"],
        lat=float(candidate["lat"]),
        lon=float(candidate["lon"]),
        label=f"location_{location_index}",
        source=str(candidate["source"]),
        graph=graph,
        metric_positions=metric_positions,
        geography=geography,
        params=params,
        d_max_km=d_max_km,
        location_index=location_index,
    )


def _annotate_location_point(
    point_metric: Point,
    lat: float,
    lon: float,
    label: str,
    source: str,
    graph: nx.Graph,
    metric_positions: dict[int, tuple[float, float]],
    geography: dict,
    params: ScenarioParameters,
    d_max_km: float,
    location_index: int,
) -> dict:
    if not geography["sovereign_metric"].covers(point_metric):
        raise ValueError("Selected location must lie on sovereign-held Ukraine ground")
    nearest_node = min(
        graph.nodes(),
        key=lambda n: _point_distance(metric_positions[int(n)], (point_metric.x, point_metric.y)),
    )
    frontline_distance_km = float(point_metric.distance(geography["launch_interface_metric"]) / 1000.0)
    russia_border_distance_km = float(point_metric.distance(geography["ukraine_russia_border_metric"]) / 1000.0)
    depth_factor = max(0.0, 1.0 - frontline_distance_km / d_max_km) ** float(params.depth_penalty_gamma)
    return {
        "location_id": f"location_{location_index}",
        "label": label,
        "source": source,
        "lat": float(lat),
        "lon": float(lon),
        "point_metric": point_metric,
        "nearest_node": int(nearest_node),
        "frontline_distance_km": frontline_distance_km,
        "russia_border_distance_km": russia_border_distance_km,
        "depth_factor": depth_factor,
        "link_radius_km": float(params.budget.location_link_radius_km),
    }


def _choose_random_strike_event(
    attack_type: str,
    selected_locations: list[dict],
    existing_strike_events: list[dict],
    candidates: list[dict],
    graph: nx.Graph,
    metric_positions: dict[int, tuple[float, float]],
    geography: dict,
    params: ScenarioParameters,
    d_max_km: float,
    rng,
) -> dict | None:
    attack_cfg = getattr(params, attack_type)
    event_candidates: list[dict] = []
    for location in selected_locations:
        nearby_candidates = [
            candidate
            for candidate in candidates
            if candidate["point_metric"].distance(location["point_metric"]) <= params.budget.location_link_radius_km * 1000.0
            and not _violates_separation(candidate["point_metric"], existing_strike_events, attack_cfg.min_strike_separation_km)
        ]
        for candidate in nearby_candidates:
            event_candidates.append(
                _annotate_strike_point(
                    point_metric=candidate["point_metric"],
                    lat=float(candidate["lat"]),
                    lon=float(candidate["lon"]),
                    label=str(candidate["label"]),
                    source=str(candidate["source"]),
                    attack_type=attack_type,
                    strike_index=_strike_count(existing_strike_events, attack_type) + 1,
                    radius_km=float(attack_cfg.radius_km),
                    base_budget=float(params.budget.missile_cost if attack_type == "missile" else params.budget.bomb_cost),
                    graph=graph,
                    metric_positions=metric_positions,
                    geography=geography,
                    params=params,
                    d_max_km=d_max_km,
                    parent_location=location,
                )
            )

    if not event_candidates:
        return None

    weights = np.asarray([max(1e-6, candidate["selection_score"]) for candidate in event_candidates], dtype=float)
    probabilities = weights / weights.sum()
    return event_candidates[int(rng.choice(np.arange(len(event_candidates)), p=probabilities))]


def _annotate_strike_point(
    point_metric: Point,
    lat: float,
    lon: float,
    label: str,
    source: str,
    attack_type: str,
    strike_index: int,
    radius_km: float,
    base_budget: float,
    graph: nx.Graph,
    metric_positions: dict[int, tuple[float, float]],
    geography: dict,
    params: ScenarioParameters,
    d_max_km: float,
    parent_location: dict,
) -> dict:
    if not geography["sovereign_metric"].covers(point_metric):
        raise ValueError("Strike center must lie on sovereign-held Ukraine ground")
    nearest_node = min(
        graph.nodes(),
        key=lambda n: _point_distance(metric_positions[int(n)], (point_metric.x, point_metric.y)),
    )
    frontline_distance_km = float(point_metric.distance(geography["launch_interface_metric"]) / 1000.0)
    russia_border_distance_km = float(point_metric.distance(geography["ukraine_russia_border_metric"]) / 1000.0)
    depth_factor = max(0.0, 1.0 - frontline_distance_km / d_max_km) ** float(params.depth_penalty_gamma)
    effective_power = float(base_budget * depth_factor)
    distance_to_location_km = float(point_metric.distance(parent_location["point_metric"]) / 1000.0)
    selection_score = max(1e-6, depth_factor * (1.4 if attack_type == "missile" else 1.0))
    return {
        "attack_type": attack_type,
        "strike_id": f"{attack_type}_{strike_index}",
        "strike_index": int(strike_index),
        "label": label,
        "source": source,
        "lat": float(lat),
        "lon": float(lon),
        "point_metric": point_metric,
        "nearest_node": int(nearest_node),
        "frontline_distance_km": frontline_distance_km,
        "russia_border_distance_km": russia_border_distance_km,
        "d_max_km": float(d_max_km),
        "radius_km": float(radius_km),
        "base_budget": float(base_budget),
        "depth_factor": depth_factor,
        "effective_power": effective_power,
        "parent_location_id": parent_location["location_id"],
        "parent_location_label": parent_location["label"],
        "distance_to_location_km": distance_to_location_km,
        "selection_score": selection_score,
    }


def _apply_strikes(edge_table: pd.DataFrame, strike_events: list[dict], params: ScenarioParameters) -> pd.DataFrame:
    impacts = edge_table.copy()
    impacted_power: dict[str, float] = defaultdict(float)
    attack_types: dict[str, list[str]] = defaultdict(list)
    strike_ids: dict[str, list[str]] = defaultdict(list)
    strike_hits: dict[str, int] = defaultdict(int)
    strike_radii: dict[str, list[float]] = defaultdict(list)
    min_distance: dict[str, float] = {key: float("inf") for key in impacts["edge_key"]}

    for strike in strike_events:
        distance_lookup = {
            row.edge_key: float(row.line_metric.distance(strike["point_metric"]) / 1000.0)
            for row in impacts.itertuples(index=False)
        }
        for edge_key, distance_km in distance_lookup.items():
            min_distance[edge_key] = min(min_distance[edge_key], distance_km)

        candidates = impacts.loc[impacts["edge_key"].map(lambda key: distance_lookup[key] <= strike["radius_km"])].copy()
        if candidates.empty:
            continue

        if strike["attack_type"] == "missile":
            target = candidates.sort_values(
                by=["edge_value", "travel_time_base", "path_count_norm"],
                ascending=[False, False, False],
            ).iloc[0]
            _record_strike_hit(
                impacted_power,
                attack_types,
                strike_ids,
                strike_hits,
                strike_radii,
                target["edge_key"],
                strike,
                float(strike["effective_power"]),
            )
            continue

        per_edge_power = float(strike["effective_power"]) / float(params.bomb_reduction_factor)
        for target in candidates.itertuples(index=False):
            _record_strike_hit(
                impacted_power,
                attack_types,
                strike_ids,
                strike_hits,
                strike_radii,
                target.edge_key,
                strike,
                per_edge_power,
            )

    impacts["min_distance_to_any_strike_km"] = impacts["edge_key"].map(lambda key: float(min_distance[key]))
    impacts["power_total"] = impacts["edge_key"].map(lambda key: float(impacted_power.get(key, 0.0))).astype(float)
    impacts["attack_types"] = impacts["edge_key"].map(lambda key: ",".join(attack_types.get(key, [])))
    impacts["strike_ids"] = impacts["edge_key"].map(lambda key: ",".join(strike_ids.get(key, [])))
    impacts["strike_count"] = impacts["edge_key"].map(lambda key: int(strike_hits.get(key, 0)))
    impacts["max_strike_radius_km"] = impacts["edge_key"].map(
        lambda key: float(max(strike_radii.get(key, [0.0])))
    ).astype(float)
    impacts["ratio"] = impacts["power_total"] / impacts["defense_score"].clip(lower=1e-9)
    impacts["status"] = "unaffected"
    impacts.loc[impacts["power_total"] > 0.0, "status"] = "struck_no_change"
    impacts.loc[impacts["ratio"] > float(params.theta_degrade), "status"] = "degraded"
    impacts.loc[impacts["ratio"] > float(params.theta_remove), "status"] = "removed"
    impacts["travel_time_scenario"] = impacts["travel_time_base"]
    impacts.loc[impacts["status"] == "degraded", "travel_time_scenario"] = (
        impacts.loc[impacts["status"] == "degraded", "travel_time_base"] * float(params.degrade_multiplier)
    )
    impacts.loc[impacts["status"] == "removed", "travel_time_scenario"] = np.inf
    impacts["status_rank"] = impacts["status"].map(
        {"removed": 0, "degraded": 1, "struck_no_change": 2, "unaffected": 3}
    ).astype(int)
    return impacts


def _record_strike_hit(
    impacted_power: dict[str, float],
    attack_types: dict[str, list[str]],
    strike_ids: dict[str, list[str]],
    strike_hits: dict[str, int],
    strike_radii: dict[str, list[float]],
    edge_key: str,
    strike: dict,
    power: float,
) -> None:
    impacted_power[edge_key] += float(power)
    attack_types[edge_key].append(strike["attack_type"])
    strike_ids[edge_key].append(strike["strike_id"])
    strike_hits[edge_key] += 1
    strike_radii[edge_key].append(float(strike["radius_km"]))


def _build_scenario_graph(
    graph: nx.Graph,
    impact_table: pd.DataFrame,
    params: ScenarioParameters,
) -> tuple[nx.Graph, list[int]]:
    scenario_graph = graph.copy()
    removed_edges: list[tuple[int, int]] = []
    for row in impact_table.itertuples(index=False):
        if not scenario_graph.has_edge(row.u, row.v):
            continue
        edge_attrs = scenario_graph[row.u][row.v]
        edge_attrs["travel_time_base"] = float(row.travel_time_base)
        edge_attrs["travel_time_scenario"] = float(row.travel_time_scenario)
        edge_attrs["scenario_status"] = row.status
        edge_attrs["scenario_power"] = float(row.power_total)
        edge_attrs["scenario_ratio"] = float(row.ratio)
        edge_attrs["scenario_defense"] = float(row.defense_score)
        edge_attrs["scenario_attack_types"] = row.attack_types
        edge_attrs["scenario_strike_ids"] = row.strike_ids
        edge_attrs["scenario_strike_count"] = int(row.strike_count)
        if row.status == "degraded":
            edge_attrs["travel_time"] = float(row.travel_time_scenario)
        elif row.status == "removed":
            removed_edges.append((int(row.u), int(row.v)))

    scenario_graph.remove_edges_from(removed_edges)
    removed_nodes: list[int] = []
    if params.remove_isolated_nodes:
        removed_nodes = [int(node) for node, degree in dict(scenario_graph.degree()).items() if degree == 0]
        scenario_graph.remove_nodes_from(removed_nodes)
    return scenario_graph, removed_nodes


def _build_cost_matrix_for_graph(graph: nx.Graph) -> pd.DataFrame:
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="travel_time"))
    rows: list[dict] = []
    for source in sorted(graph.nodes()):
        reachable = lengths.get(source, {})
        for target in sorted(graph.nodes()):
            rows.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "c_ij": float(reachable.get(target, np.inf)),
                }
            )
    return pd.DataFrame(rows)


def _build_summary(
    graph: nx.Graph,
    scenario_graph: nx.Graph,
    impact_table: pd.DataFrame,
    c_ij: pd.DataFrame,
    selected_locations: list[dict],
    strike_events: list[dict],
    budget_summary: dict,
    params: ScenarioParameters,
    removed_nodes: list[int],
) -> dict:
    impacted = impact_table[impact_table["power_total"] > 0.0].copy()
    degraded = impact_table[impact_table["status"] == "degraded"].copy()
    removed = impact_table[impact_table["status"] == "removed"].copy()
    finite_costs = c_ij[np.isfinite(c_ij["c_ij_scenario"])].copy()
    return {
        "scenario_id": params.scenario_id,
        "attack_mode": params.attack_mode,
        "base_budget": float(params.base_budget),
        "budget_summary": budget_summary,
        "location_count": int(len(selected_locations)),
        "selected_locations": selected_locations,
        "strike_counts": {
            "total": int(len(strike_events)),
            "missile": int(sum(1 for event in strike_events if event["attack_type"] == "missile")),
            "bomb": int(sum(1 for event in strike_events if event["attack_type"] == "bomb")),
        },
        "strike_events": strike_events,
        "base_graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        },
        "scenario_graph": {
            "nodes": int(scenario_graph.number_of_nodes()),
            "edges": int(scenario_graph.number_of_edges()),
            "removed_isolated_nodes": int(len(removed_nodes)),
        },
        "edge_impacts": {
            "struck_edges": int(len(impacted)),
            "degraded_edges": int(len(degraded)),
            "removed_edges": int(len(removed)),
            "max_ratio": float(impact_table["ratio"].max()),
            "max_power": float(impact_table["power_total"].max()),
        },
        "c_ij": {
            "finite_min": float(finite_costs["c_ij_scenario"].min()) if not finite_costs.empty else None,
            "finite_mean": float(finite_costs["c_ij_scenario"].mean()) if not finite_costs.empty else None,
            "finite_max": float(finite_costs["c_ij_scenario"].max()) if not finite_costs.empty else None,
            "infinite_pairs": int((~np.isfinite(c_ij["c_ij_scenario"])).sum()),
        },
        "removed_nodes": removed_nodes,
    }


def _generate_candidate_strike_centers(edge_table: pd.DataFrame, geography: dict, params: ScenarioParameters) -> list[dict]:
    candidates: list[dict] = []
    seen_keys: set[tuple[float, float]] = set()
    spacing_m = float(params.candidate_grid_spacing_km) * 1000.0
    minx, miny, maxx, maxy = geography["sovereign_metric"].bounds
    x_values = np.arange(minx, maxx + spacing_m, spacing_m)
    y_values = np.arange(miny, maxy + spacing_m, spacing_m)

    for ix, x_coord in enumerate(x_values):
        for iy, y_coord in enumerate(y_values):
            point_metric = Point(float(x_coord), float(y_coord))
            if not geography["sovereign_metric"].covers(point_metric):
                continue
            _append_candidate(candidates, seen_keys, point_metric, geography, f"grid_{ix}_{iy}", "grid")

    top_edges = edge_table.sort_values("edge_value", ascending=False).head(int(params.edge_midpoint_candidate_count))
    for row in top_edges.itertuples(index=False):
        for frac, source in ((0.5, "edge_midpoint"), (0.25, "edge_sample"), (0.75, "edge_sample")):
            point_metric = row.line_metric.interpolate(frac, normalized=True)
            if geography["sovereign_metric"].covers(point_metric):
                _append_candidate(
                    candidates,
                    seen_keys,
                    point_metric,
                    geography,
                    f"{source}_{row.u}_{row.v}_{int(frac * 100)}",
                    source,
                )

    if not candidates:
        raise ValueError("Could not generate any valid candidates inside sovereign-held Ukraine")
    return candidates


def _append_candidate(
    candidates: list[dict],
    seen_keys: set[tuple[float, float]],
    point_metric: Point,
    geography: dict,
    label: str,
    source: str,
) -> None:
    lon_lat_point = gpd.GeoSeries([point_metric], crs=3857).to_crs("EPSG:4326").iloc[0]
    key = (round(float(lon_lat_point.x), 5), round(float(lon_lat_point.y), 5))
    if key in seen_keys:
        return
    seen_keys.add(key)
    frontline_distance_km = float(point_metric.distance(geography["launch_interface_metric"]) / 1000.0)
    russia_border_distance_km = float(point_metric.distance(geography["ukraine_russia_border_metric"]) / 1000.0)
    location_score = max(1e-6, 1.0 / (1.0 + frontline_distance_km))
    candidates.append(
        {
            "lat": float(lon_lat_point.y),
            "lon": float(lon_lat_point.x),
            "label": label,
            "source": source,
            "point_metric": point_metric,
            "frontline_distance_km": frontline_distance_km,
            "russia_border_distance_km": russia_border_distance_km,
            "location_score": location_score,
        }
    )


def _violates_separation(point_metric: Point, selected_items: list[dict], min_separation_km: float) -> bool:
    threshold_m = float(min_separation_km) * 1000.0
    for item in selected_items:
        if point_metric.distance(item["point_metric"]) < threshold_m:
            return True
    return False


def _strike_count(strike_events: list[dict], attack_type: str) -> int:
    return sum(1 for event in strike_events if event["attack_type"] == attack_type)


def _all_strike_caps_reached(params: ScenarioParameters, strike_events: list[dict]) -> bool:
    missile_done = _strike_count(strike_events, "missile") >= params.missile.max_strikes or "missile" not in params.active_attack_types()
    bomb_done = _strike_count(strike_events, "bomb") >= params.bomb.max_strikes or "bomb" not in params.active_attack_types()
    return missile_done and bomb_done


def _budget_summary(params: ScenarioParameters, selected_locations: list[dict], strike_events: list[dict], remaining_budget: float) -> dict:
    missile_count = _strike_count(strike_events, "missile")
    bomb_count = _strike_count(strike_events, "bomb")
    return {
        "base_budget": float(params.base_budget),
        "spent_on_locations": float(len(selected_locations) * params.budget.location_cost),
        "spent_on_missiles": float(missile_count * params.budget.missile_cost),
        "spent_on_bombs": float(bomb_count * params.budget.bomb_cost),
        "remaining_budget": float(remaining_budget),
        "location_cost": float(params.budget.location_cost),
        "missile_cost": float(params.budget.missile_cost),
        "bomb_cost": float(params.budget.bomb_cost),
        "location_link_radius_km": float(params.budget.location_link_radius_km),
    }


def _serializable_locations(selected_locations: list[dict]) -> list[dict]:
    return [{key: value for key, value in location.items() if key != "point_metric"} for location in selected_locations]


def _serializable_strike_events(strike_events: list[dict]) -> list[dict]:
    return [
        {
            key: value
            for key, value in event.items()
            if key not in {"point_metric", "selection_score"}
        }
        for event in strike_events
    ]


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    normalized = np.zeros_like(values, dtype=float)
    if not finite_mask.any():
        return normalized
    finite_values = values[finite_mask]
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if math.isclose(vmin, vmax):
        normalized[finite_mask] = 1.0
        return normalized
    normalized[finite_mask] = (finite_values - vmin) / (vmax - vmin)
    return normalized
