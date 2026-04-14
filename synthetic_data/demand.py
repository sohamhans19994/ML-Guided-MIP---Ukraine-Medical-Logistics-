from __future__ import annotations

from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd


def cluster_demand_kmedoids(coords, n_clusters, random_state=42, max_iter=8, candidate_cap=200):
    coords = np.asarray(coords, dtype=float)
    n_samples = len(coords)
    if n_samples == 0:
        raise ValueError("cluster_demand_kmedoids needs at least one point")
    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot exceed the number of points")

    rng = np.random.default_rng(random_state)
    medoid_indices = [int(rng.integers(n_samples))]
    closest_dist = ((coords - coords[medoid_indices[0]]) ** 2).sum(axis=1)

    for _ in range(1, n_clusters):
        total = closest_dist.sum()
        if total <= 0:
            remaining = np.setdiff1d(np.arange(n_samples), np.array(medoid_indices), assume_unique=False)
            next_idx = int(rng.choice(remaining))
        else:
            probs = closest_dist / total
            next_idx = int(rng.choice(n_samples, p=probs))
            while next_idx in medoid_indices:
                next_idx = int(rng.choice(n_samples, p=probs))
        medoid_indices.append(next_idx)
        new_dist = ((coords - coords[next_idx]) ** 2).sum(axis=1)
        closest_dist = np.minimum(closest_dist, new_dist)

    medoid_indices = np.array(medoid_indices, dtype=int)

    for _ in range(max_iter):
        medoids = coords[medoid_indices]
        dist_to_medoids = ((coords[:, None, :] - medoids[None, :, :]) ** 2).sum(axis=2)
        labels = dist_to_medoids.argmin(axis=1)
        updated = medoid_indices.copy()

        for k in range(n_clusters):
            cluster_idx = np.where(labels == k)[0]
            if len(cluster_idx) == 0:
                fallback = int(np.argmax(np.min(dist_to_medoids, axis=1)))
                updated[k] = fallback
                continue

            cluster_points = coords[cluster_idx]
            if len(cluster_idx) > candidate_cap:
                centroid = cluster_points.mean(axis=0)
                order = np.argsort(((cluster_points - centroid) ** 2).sum(axis=1))
                near_count = min(candidate_cap // 2, len(cluster_idx))
                chosen_local = order[:near_count]
                remaining = np.setdiff1d(np.arange(len(cluster_idx)), chosen_local, assume_unique=False)
                extra_count = min(candidate_cap - len(chosen_local), len(remaining))
                if extra_count > 0:
                    extra_local = rng.choice(remaining, size=extra_count, replace=False)
                    chosen_local = np.unique(np.concatenate([chosen_local, extra_local]))
                candidate_idx = cluster_idx[chosen_local]
            else:
                candidate_idx = cluster_idx

            candidate_points = coords[candidate_idx]
            candidate_costs = ((cluster_points[:, None, :] - candidate_points[None, :, :]) ** 2).sum(axis=2).sum(axis=0)
            updated[k] = int(candidate_idx[int(np.argmin(candidate_costs))])

        if np.array_equal(updated, medoid_indices):
            break
        medoid_indices = updated

    final_medoids = coords[medoid_indices]
    final_dist = ((coords[:, None, :] - final_medoids[None, :, :]) ** 2).sum(axis=2)
    labels = final_dist.argmin(axis=1)
    return labels, medoid_indices


def build_demand_nodes_kmedoids(df, n_clusters, averaging_days=730, random_state=42, candidate_cap=200):
    working = df.copy()
    coords = working[["latitude", "longitude"]].to_numpy(dtype=float)
    labels, medoid_indices = cluster_demand_kmedoids(
        coords,
        n_clusters=n_clusters,
        random_state=random_state,
        candidate_cap=candidate_cap,
    )
    working["cluster"] = labels

    medoid_lookup = {
        cluster_id: {
            "lat": float(coords[medoid_indices[cluster_id], 0]),
            "lon": float(coords[medoid_indices[cluster_id], 1]),
        }
        for cluster_id in range(n_clusters)
    }

    demand_raw = (
        working.groupby("cluster")
        .agg(
            n_events=("event_type", "count"),
            total_fatalities=("fatalities", "sum"),
        )
        .reset_index()
    )
    averaging_days = float(averaging_days)
    if averaging_days <= 0:
        raise ValueError("averaging_days must be positive")
    demand_raw["daily_demand"] = demand_raw["n_events"] / averaging_days
    demand_raw["daily_fatalities"] = demand_raw["total_fatalities"] / averaging_days
    demand_raw["lat"] = demand_raw["cluster"].map(lambda c: medoid_lookup[c]["lat"])
    demand_raw["lon"] = demand_raw["cluster"].map(lambda c: medoid_lookup[c]["lon"])
    demand_raw = demand_raw[
        ["lat", "lon", "n_events", "daily_demand", "total_fatalities", "daily_fatalities"]
    ].reset_index(drop=True)
    return working, demand_raw, medoid_indices


def load_demand_nodes(config: dict):
    demand_cfg = config["demand"]
    acled_path = Path(config["paths"]["acled_csv"])
    if not acled_path.exists():
        raise FileNotFoundError(f"Could not find ACLED data at {acled_path}")

    acled_df = pd.read_csv(acled_path)
    acled_df = acled_df[
        (acled_df["country"] == demand_cfg["country"])
        & (acled_df["year"] >= int(demand_cfg["min_year"]))
        & (acled_df["event_type"].isin(demand_cfg["event_types"]))
    ].copy()

    filtered_df, demand_nodes, medoid_indices = build_demand_nodes_kmedoids(
        acled_df,
        n_clusters=int(demand_cfg["n_clusters"]),
        averaging_days=float(demand_cfg.get("averaging_days", 730)),
        random_state=int(demand_cfg["random_state"]),
        candidate_cap=int(demand_cfg["candidate_cap"]),
    )
    return filtered_df, demand_nodes, medoid_indices


def snap_demand_nodes_to_graph(demand_nodes: pd.DataFrame, graph) -> pd.DataFrame:
    snapped = demand_nodes.copy()
    snapped["centroid_lat"] = snapped["lat"]
    snapped["centroid_lon"] = snapped["lon"]
    snapped_nodes = ox.distance.nearest_nodes(
        graph,
        X=snapped["lon"].to_list(),
        Y=snapped["lat"].to_list(),
    )
    snapped["graph_node"] = snapped_nodes
    snapped["lat"] = [graph.nodes[n]["y"] for n in snapped_nodes]
    snapped["lon"] = [graph.nodes[n]["x"] for n in snapped_nodes]
    return snapped
