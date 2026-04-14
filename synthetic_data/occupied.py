from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlopen

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.ops import unary_union

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

    migrate_if_missing(cache_geojson, [config["_project_root"] / "data" / "deepstate-map-data.geojson"])
    migrate_if_missing(cache_gz, [config["_project_root"] / "data" / "deepstate-map-data.geojson.gz"])

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


def _ensure_zip_downloaded(url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        with urlopen(url) as response, zip_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)


def _ensure_zip_extracted(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    shp_files = sorted(extract_dir.glob("*.shp"))
    if shp_files:
        return shp_files[0]

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    shp_files = sorted(extract_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No shapefile found after extracting {zip_path}")
    return shp_files[0]


def load_ukraine_sovereign_geometry(config: dict, occupied_geom):
    paths = config["paths"]
    occ_cfg = config["occupied_filter"]
    countries_zip = Path(paths["natural_earth_countries_zip"])
    countries_dir = Path(paths["natural_earth_countries_dir"])
    countries_url = config["data_sources"]["natural_earth_countries_url"]

    _ensure_zip_downloaded(countries_url, countries_zip)
    countries_shp = _ensure_zip_extracted(countries_zip, countries_dir)

    ne_countries = gpd.read_file(countries_shp).to_crs("EPSG:4326")
    ukraine_shape = ne_countries[
        (ne_countries.get("ADMIN") == "Ukraine")
        | (ne_countries.get("SOVEREIGNT") == "Ukraine")
        | (ne_countries.get("NAME") == "Ukraine")
    ].copy()
    if ukraine_shape.empty:
        raise ValueError("Could not find Ukraine in the Natural Earth country polygon layer")

    russia_labels = {"Russia", "Russian Federation"}
    russia_shape = ne_countries[
        ne_countries.get("ADMIN", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
        | ne_countries.get("SOVEREIGNT", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
        | ne_countries.get("NAME", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
        | ne_countries.get("NAME_LONG", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
    ].copy()
    if russia_shape.empty:
        raise ValueError("Could not find Russia in the Natural Earth country polygon layer")

    ukraine_geom = (
        ukraine_shape.geometry.union_all()
        if hasattr(ukraine_shape.geometry, "union_all")
        else ukraine_shape.geometry.unary_union
    )
    russia_geom = (
        russia_shape.geometry.union_all()
        if hasattr(russia_shape.geometry, "union_all")
        else russia_shape.geometry.unary_union
    )
    occupied_metric = gpd.GeoSeries([occupied_geom], crs="EPSG:4326").to_crs(3857).iloc[0]
    ukraine_metric = gpd.GeoSeries([ukraine_geom], crs="EPSG:4326").to_crs(3857).iloc[0]
    russia_metric = gpd.GeoSeries([russia_geom], crs="EPSG:4326").to_crs(3857).iloc[0]

    sovereign_metric = ukraine_metric.difference(occupied_metric)
    if sovereign_metric.is_empty:
        raise ValueError("Ukraine sovereign-held geometry became empty after removing occupied territory")

    sovereign_geom = gpd.GeoSeries([sovereign_metric], crs=3857).to_crs("EPSG:4326").iloc[0]
    sovereign_shape = gpd.GeoDataFrame(geometry=[sovereign_geom], crs="EPSG:4326")
    sovereign_border_metric = sovereign_metric.boundary

    frontline_match_km = float(occ_cfg["frontline_match_km"])
    frontline_boundary_metric = sovereign_border_metric.intersection(
        occupied_metric.buffer(frontline_match_km * 1000.0)
    )
    if frontline_boundary_metric.is_empty:
        frontline_boundary_metric = occupied_metric.boundary

    border_match_m = frontline_match_km * 1000.0
    ukraine_russia_border_metric = ukraine_metric.boundary.intersection(
        russia_metric.boundary.buffer(border_match_m)
    )
    if ukraine_russia_border_metric.is_empty:
        ukraine_russia_border_metric = ukraine_metric.boundary.intersection(
            russia_metric.buffer(border_match_m)
        )

    unsafe_frontier_metric = unary_union(
        [geom for geom in [frontline_boundary_metric, ukraine_russia_border_metric] if not geom.is_empty]
    )
    if unsafe_frontier_metric.is_empty:
        unsafe_frontier_metric = frontline_boundary_metric

    exterior_boundary_metric = sovereign_border_metric.difference(
        occupied_metric.buffer(frontline_match_km * 1000.0)
    )
    if exterior_boundary_metric.is_empty:
        exterior_boundary_metric = sovereign_border_metric

    return {
        "ukraine_shape": ukraine_shape,
        "ukraine_geom": ukraine_geom,
        "sovereign_shape": sovereign_shape,
        "sovereign_geom": sovereign_geom,
        "sovereign_border_metric": sovereign_border_metric,
        "sovereign_border_gs": gpd.GeoSeries([sovereign_border_metric], crs=3857).to_crs("EPSG:4326"),
        "frontline_boundary_metric": frontline_boundary_metric,
        "frontline_boundary_gs": gpd.GeoSeries([frontline_boundary_metric], crs=3857).to_crs("EPSG:4326"),
        "ukraine_russia_border_metric": ukraine_russia_border_metric,
        "ukraine_russia_border_gs": gpd.GeoSeries([ukraine_russia_border_metric], crs=3857).to_crs("EPSG:4326"),
        "unsafe_frontier_metric": unsafe_frontier_metric,
        "unsafe_frontier_gs": gpd.GeoSeries([unsafe_frontier_metric], crs=3857).to_crs("EPSG:4326"),
        "exterior_boundary_metric": exterior_boundary_metric,
    }


def clip_graph_to_sovereign_border(graph, sovereign_geom, occupied_geom, config: dict):
    tolerance_km = float(config["occupied_filter"]["outside_new_border_tolerance_km"])
    tolerance_m = tolerance_km * 1000.0

    node_ids = list(graph.nodes())
    node_points = gpd.GeoSeries(
        gpd.points_from_xy([graph.nodes[n]["x"] for n in node_ids], [graph.nodes[n]["y"] for n in node_ids]),
        index=node_ids,
        crs="EPSG:4326",
    )
    node_points_metric = node_points.to_crs(3857)
    sovereign_metric = gpd.GeoSeries([sovereign_geom], crs="EPSG:4326").to_crs(3857).iloc[0]
    sovereign_buffer_metric = sovereign_metric.buffer(tolerance_m)

    inside_sovereign = node_points.intersects(sovereign_geom).to_dict()
    inside_occupied_raw = node_points.intersects(occupied_geom).to_dict()
    within_tolerance = {
        n: bool(node_points_metric.loc[n].intersects(sovereign_buffer_metric)) for n in node_ids
    }
    removed_outside_nodes = [n for n in node_ids if not within_tolerance[n]]

    clipped_graph = graph.copy()
    clipped_graph.remove_nodes_from(removed_outside_nodes)
    isolated_nodes = [n for n, deg in dict(clipped_graph.degree()).items() if deg == 0]
    clipped_graph.remove_nodes_from(isolated_nodes)

    return clipped_graph, {
        "inside_sovereign": inside_sovereign,
        "inside_occupied_raw": inside_occupied_raw,
        "within_tolerance": within_tolerance,
        "removed_outside_new_border_nodes": removed_outside_nodes,
        "outside_new_border_tolerance_km": tolerance_km,
    }


def annotate_graph_with_border_metrics(
    graph,
    sovereign_geom,
    occupied_geom,
    sovereign_border_metric,
    unsafe_frontier_metric,
    exterior_boundary_metric,
    clip_debug: dict,
    config: dict,
):
    occ_cfg = config["occupied_filter"]
    border_risk_buffer_km = float(occ_cfg["border_risk_buffer_km"])

    node_ids = list(graph.nodes())
    node_points = gpd.GeoSeries(
        gpd.points_from_xy([graph.nodes[n]["x"] for n in node_ids], [graph.nodes[n]["y"] for n in node_ids]),
        index=node_ids,
        crs="EPSG:4326",
    )
    node_points_metric = node_points.to_crs(3857)

    inside_sovereign = {n: bool(node_points.loc[n].intersects(sovereign_geom)) for n in node_ids}
    inside_occupied = {n: bool(node_points.loc[n].intersects(occupied_geom)) for n in node_ids}

    border_distance_km = {
        n: float(node_points_metric.loc[n].distance(sovereign_border_metric) / 1000.0)
        for n in node_ids
    }
    territory_frontline_distance_km = {
        n: float(node_points_metric.loc[n].distance(unsafe_frontier_metric) / 1000.0)
        for n in node_ids
    }
    territory_rear_distance_km = {
        n: float(node_points_metric.loc[n].distance(exterior_boundary_metric) / 1000.0)
        for n in node_ids
    }
    territory_interior_depth_km = {
        n: territory_frontline_distance_km[n] if inside_occupied[n] else 0.0
        for n in node_ids
    }
    territory_cross_width_km = {
        n: territory_frontline_distance_km[n] + territory_rear_distance_km[n]
        for n in node_ids
    }
    territory_removal_threshold_km = {
        n: float(occ_cfg["outside_new_border_tolerance_km"]) for n in node_ids
    }
    territory_is_ukraine_side = {
        n: bool(inside_sovereign[n]) for n in node_ids
    }

    frontline_proximity = {
        n: max(0.0, 1.0 - min(territory_frontline_distance_km[n], border_risk_buffer_km) / border_risk_buffer_km)
        for n in node_ids
    }
    border_proximity = {
        n: max(0.0, 1.0 - min(border_distance_km[n], border_risk_buffer_km) / border_risk_buffer_km)
        for n in node_ids
    }

    territory_border_risk = {}
    for n in node_ids:
        if inside_occupied[n]:
            territory_border_risk[n] = min(1.0, 0.75 + 0.25 * frontline_proximity[n])
        elif inside_sovereign[n]:
            territory_border_risk[n] = max(0.85 * frontline_proximity[n], 0.20 * border_proximity[n])
        else:
            territory_border_risk[n] = min(1.0, 0.60 + 0.40 * frontline_proximity[n])

    nx.set_node_attributes(graph, {n: float(inside_occupied[n]) for n in node_ids}, "territory_occupation_fraction")
    nx.set_node_attributes(graph, border_distance_km, "territory_border_distance_km")
    nx.set_node_attributes(graph, territory_interior_depth_km, "territory_interior_depth_km")
    nx.set_node_attributes(graph, territory_cross_width_km, "territory_cross_width_km")
    nx.set_node_attributes(graph, territory_removal_threshold_km, "territory_removal_threshold_km")
    nx.set_node_attributes(graph, territory_frontline_distance_km, "territory_frontline_distance_km")
    nx.set_node_attributes(graph, territory_rear_distance_km, "territory_rear_distance_km")
    nx.set_node_attributes(graph, territory_is_ukraine_side, "territory_is_ukraine_side")
    nx.set_node_attributes(graph, territory_border_risk, "territory_border_risk")

    removed_outside_nodes = clip_debug["removed_outside_new_border_nodes"]
    removed_occupied_nodes = [n for n in removed_outside_nodes if clip_debug["inside_occupied_raw"].get(n, False)]

    return graph, {
        "inside_sovereign": inside_sovereign,
        "inside_occupied": inside_occupied,
        "border_distance_km": border_distance_km,
        "territory_frontline_distance_km": territory_frontline_distance_km,
        "territory_rear_distance_km": territory_rear_distance_km,
        "territory_interior_depth_km": territory_interior_depth_km,
        "territory_cross_width_km": territory_cross_width_km,
        "territory_removal_threshold_km": territory_removal_threshold_km,
        "territory_border_risk": territory_border_risk,
        "territory_is_ukraine_side": territory_is_ukraine_side,
        "removed_occupied_nodes": removed_occupied_nodes,
        "removed_russia_side_nodes": removed_occupied_nodes,
        "removed_outside_new_border_nodes": removed_outside_nodes,
    }
