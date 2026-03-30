from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .utils import migrate_if_missing


def polygon_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    parts = []
    for piece in getattr(geom, "geoms", []):
        if piece.geom_type == "Polygon":
            parts.append(piece)
        elif piece.geom_type == "MultiPolygon":
            parts.extend(list(piece.geoms))
    return parts


def load_current_occupied_snapshot(config: dict):
    paths = config["paths"]
    remote_url = config["data_sources"]["occupied_remote_url"]
    cache_gz = Path(paths["occupied_cache_gz"])
    cache_geojson = Path(paths["occupied_cache_geojson"])

    migrate_if_missing(
        cache_geojson,
        [config["_project_root"] / "data" / "deepstate-map-data.geojson"],
    )
    migrate_if_missing(
        cache_gz,
        [config["_project_root"] / "data" / "deepstate-map-data.geojson.gz"],
    )

    cache_gz.parent.mkdir(parents=True, exist_ok=True)

    if cache_geojson.exists():
        occupied_all = gpd.read_file(cache_geojson)
    else:
        if not cache_gz.exists():
            with urlopen(remote_url) as response, cache_gz.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)

        with gzip.open(cache_gz, "rb") as src, cache_geojson.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        occupied_all = gpd.read_file(cache_geojson)

    if occupied_all.crs is None:
        occupied_all = occupied_all.set_crs("EPSG:4326")
    else:
        occupied_all = occupied_all.to_crs("EPSG:4326")

    snapshot_date = None
    if "date" in occupied_all.columns:
        occupied_all["date"] = pd.to_datetime(occupied_all["date"], errors="coerce")
        valid_dates = occupied_all["date"].dropna()
        if not valid_dates.empty:
            snapshot_date = valid_dates.max()
            occupied_all = occupied_all[occupied_all["date"] == snapshot_date].copy()

    if occupied_all.empty:
        raise ValueError("Occupied-territory dataset did not contain any usable current geometry")

    occupied_geom = (
        occupied_all.geometry.union_all()
        if hasattr(occupied_all.geometry, "union_all")
        else occupied_all.geometry.unary_union
    )
    return {
        "occupied_gdf": occupied_all,
        "snapshot_date": snapshot_date,
        "occupied_geom": occupied_geom,
        "occupied_components": polygon_parts(occupied_geom),
        "occupied_gs": gpd.GeoSeries([occupied_geom], crs="EPSG:4326"),
    }


def load_ukraine_boundary_and_land_border(config: dict):
    paths = config["paths"]
    occ_cfg = config["occupied_filter"]
    boundary_cache = Path(paths["boundary_cache"])
    coastline_cache = Path(paths["coastline_cache"])

    migrate_if_missing(
        coastline_cache,
        [config["_project_root"] / "cache" / "ukraine_coastline.geojson"],
    )

    boundary_cache.parent.mkdir(parents=True, exist_ok=True)
    coastline_cache.parent.mkdir(parents=True, exist_ok=True)

    if boundary_cache.exists():
        ukraine_shape = gpd.read_file(boundary_cache).to_crs("EPSG:4326")
    else:
        ukraine_shape = ox.geocode_to_gdf(config["data_sources"]["ukraine_place"])
        ukraine_shape.to_file(boundary_cache, driver="GeoJSON")

    ukraine_geom = (
        ukraine_shape.geometry.union_all()
        if hasattr(ukraine_shape.geometry, "union_all")
        else ukraine_shape.geometry.unary_union
    )
    ukraine_metric = gpd.GeoSeries([ukraine_geom], crs="EPSG:4326").to_crs(3857).iloc[0]
    full_boundary_metric = ukraine_metric.boundary

    if coastline_cache.exists():
        coastline_gdf = gpd.read_file(coastline_cache)
    else:
        coastline_gdf = ox.features_from_place(
            config["data_sources"]["ukraine_place"],
            tags={"natural": "coastline"},
        ).copy()
        coastline_gdf = coastline_gdf[
            coastline_gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()
        coastline_gdf.to_file(coastline_cache, driver="GeoJSON")

    coastline_gdf = coastline_gdf.to_crs("EPSG:4326")
    coastline_metric = coastline_gdf.to_crs(3857)
    coastline_union_metric = (
        coastline_metric.geometry.union_all()
        if hasattr(coastline_metric.geometry, "union_all")
        else coastline_metric.geometry.unary_union
    )

    land_only_border_metric = full_boundary_metric.difference(
        coastline_union_metric.buffer(float(occ_cfg["coastline_strip_km"]) * 1000.0)
    )

    return {
        "ukraine_shape": ukraine_shape,
        "ukraine_geom": ukraine_geom,
        "land_only_border_metric": land_only_border_metric,
        "coastline_metric": coastline_union_metric,
        "land_only_border_gs": gpd.GeoSeries([land_only_border_metric], crs=3857).to_crs("EPSG:4326"),
    }


def build_forgiving_current_filter(
    graph,
    occupied_geom,
    occupied_components,
    land_only_border_metric,
    coastline_metric,
    config: dict,
):
    occ_cfg = config["occupied_filter"]
    keep_band_km = float(occ_cfg["frontline_keep_band_km"])
    russia_side_match_km = float(occ_cfg["russia_side_match_km"])
    border_risk_buffer_km = float(occ_cfg["border_risk_buffer_km"])

    node_ids = list(graph.nodes())
    node_points = gpd.GeoSeries(
        gpd.points_from_xy([graph.nodes[n]["x"] for n in node_ids], [graph.nodes[n]["y"] for n in node_ids]),
        index=node_ids,
        crs="EPSG:4326",
    )
    node_points_metric = node_points.to_crs(3857)
    inside_occupied = node_points.intersects(occupied_geom).to_dict()

    russia_side_reference_metric = land_only_border_metric.union(coastline_metric)
    russia_side_buffer_metric = russia_side_reference_metric.buffer(russia_side_match_km * 1000.0)

    territory_frontline_distance_km = {n: np.inf for n in node_ids}
    territory_rear_distance_km = {n: np.inf for n in node_ids}
    territory_interior_depth_km = {n: 0.0 for n in node_ids}
    territory_cross_width_km = {n: np.inf for n in node_ids}
    territory_removal_threshold_km = {n: keep_band_km for n in node_ids}
    territory_is_ukraine_side = {n: False for n in node_ids}
    frontline_boundaries_metric = []

    for poly in occupied_components:
        poly_nodes = [n for n in node_ids if inside_occupied[n] and node_points.loc[n].intersects(poly)]
        if not poly_nodes:
            continue

        poly_metric = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(3857).iloc[0]
        poly_boundary = poly_metric.boundary
        rear_boundary = poly_boundary.intersection(russia_side_buffer_metric)
        frontline_boundary = poly_boundary.difference(russia_side_buffer_metric)

        if rear_boundary.is_empty:
            rear_boundary = poly_boundary
        if not frontline_boundary.is_empty:
            frontline_boundaries_metric.append(frontline_boundary)

        if frontline_boundary.is_empty:
            poly_front_dists = {n: float("inf") for n in poly_nodes}
        else:
            poly_front_dists = {
                n: float(node_points_metric.loc[n].distance(frontline_boundary) / 1000.0)
                for n in poly_nodes
            }
        poly_rear_dists = {
            n: float(node_points_metric.loc[n].distance(rear_boundary) / 1000.0)
            for n in poly_nodes
        }

        finite_cross_widths = [
            poly_front_dists[n] + poly_rear_dists[n]
            for n in poly_nodes
            if np.isfinite(poly_front_dists[n]) and np.isfinite(poly_rear_dists[n])
        ]
        component_cross_width_km = max(finite_cross_widths) if finite_cross_widths else np.inf

        for n in poly_nodes:
            territory_frontline_distance_km[n] = poly_front_dists[n]
            territory_rear_distance_km[n] = poly_rear_dists[n]
            territory_interior_depth_km[n] = poly_front_dists[n]
            territory_cross_width_km[n] = component_cross_width_km
            territory_is_ukraine_side[n] = np.isfinite(poly_front_dists[n]) and (
                poly_front_dists[n] <= poly_rear_dists[n]
            )

    if frontline_boundaries_metric:
        frontline_series = gpd.GeoSeries(frontline_boundaries_metric, crs=3857)
        frontline_boundary_metric = (
            frontline_series.union_all() if hasattr(frontline_series, "union_all") else frontline_series.unary_union
        )
    else:
        frontline_boundary_metric = gpd.GeoSeries([occupied_geom], crs="EPSG:4326").to_crs(3857).boundary.iloc[0]

    border_distance_km = {}
    for n in node_ids:
        if inside_occupied[n]:
            border_distance_km[n] = territory_frontline_distance_km[n]
        else:
            border_distance_km[n] = float(node_points_metric.loc[n].distance(frontline_boundary_metric) / 1000.0)

    removed_occupied_nodes = [
        n
        for n in node_ids
        if inside_occupied[n]
        and (
            not territory_is_ukraine_side[n]
            or not np.isfinite(territory_frontline_distance_km[n])
            or territory_frontline_distance_km[n] > keep_band_km
        )
    ]
    removed_russia_side_nodes = [n for n in node_ids if inside_occupied[n] and (not territory_is_ukraine_side[n])]

    border_proximity = {
        n: max(0.0, 1.0 - min(border_distance_km[n], border_risk_buffer_km) / border_risk_buffer_km)
        for n in node_ids
    }
    territory_border_risk = {}
    for n in node_ids:
        if inside_occupied[n]:
            if territory_is_ukraine_side[n] and np.isfinite(territory_frontline_distance_km[n]):
                inside_pressure = min(1.0, territory_frontline_distance_km[n] / keep_band_km)
                territory_border_risk[n] = min(1.0, 0.55 + 0.45 * inside_pressure)
            else:
                territory_border_risk[n] = 1.0
        else:
            territory_border_risk[n] = 0.85 * border_proximity[n]

    filtered_graph = graph.copy()
    filtered_graph.remove_nodes_from(removed_occupied_nodes)
    isolated_nodes = [n for n, deg in dict(filtered_graph.degree()).items() if deg == 0]
    filtered_graph.remove_nodes_from(isolated_nodes)

    surviving_nodes = list(filtered_graph.nodes())
    nx.set_node_attributes(
        filtered_graph,
        {n: float(inside_occupied[n]) for n in surviving_nodes},
        "territory_occupation_fraction",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(border_distance_km[n]) for n in surviving_nodes},
        "territory_border_distance_km",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(territory_interior_depth_km[n]) for n in surviving_nodes},
        "territory_interior_depth_km",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(territory_cross_width_km[n]) for n in surviving_nodes},
        "territory_cross_width_km",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(territory_removal_threshold_km[n]) for n in surviving_nodes},
        "territory_removal_threshold_km",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(territory_frontline_distance_km[n]) for n in surviving_nodes},
        "territory_frontline_distance_km",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(territory_rear_distance_km[n]) for n in surviving_nodes},
        "territory_rear_distance_km",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: bool(territory_is_ukraine_side[n]) for n in surviving_nodes},
        "territory_is_ukraine_side",
    )
    nx.set_node_attributes(
        filtered_graph,
        {n: float(territory_border_risk[n]) for n in surviving_nodes},
        "territory_border_risk",
    )

    return filtered_graph, {
        "inside_occupied": inside_occupied,
        "occupied_geom": occupied_geom,
        "border_distance_km": border_distance_km,
        "territory_frontline_distance_km": territory_frontline_distance_km,
        "territory_rear_distance_km": territory_rear_distance_km,
        "territory_interior_depth_km": territory_interior_depth_km,
        "territory_cross_width_km": territory_cross_width_km,
        "territory_removal_threshold_km": territory_removal_threshold_km,
        "territory_border_risk": territory_border_risk,
        "territory_is_ukraine_side": territory_is_ukraine_side,
        "removed_occupied_nodes": removed_occupied_nodes,
        "removed_russia_side_nodes": removed_russia_side_nodes,
    }
