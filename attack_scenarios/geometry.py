from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


def load_attack_geography(config: dict) -> dict:
    paths = config["paths"]
    occ_cfg = config["occupied_filter"]

    occupied_gdf = gpd.read_file(Path(paths["occupied_cache_geojson"]))
    if occupied_gdf.crs is None:
        occupied_gdf = occupied_gdf.set_crs("EPSG:4326")
    else:
        occupied_gdf = occupied_gdf.to_crs("EPSG:4326")

    snapshot_date = None
    if "date" in occupied_gdf.columns:
        occupied_gdf["date"] = pd.to_datetime(occupied_gdf["date"], errors="coerce")
        valid_dates = occupied_gdf["date"].dropna()
        if not valid_dates.empty:
            snapshot_date = valid_dates.max()
            occupied_gdf = occupied_gdf[occupied_gdf["date"] == snapshot_date].copy()

    occupied_geom = (
        occupied_gdf.geometry.union_all()
        if hasattr(occupied_gdf.geometry, "union_all")
        else occupied_gdf.geometry.unary_union
    )

    countries_dir = Path(paths["natural_earth_countries_dir"])
    shp_files = sorted(countries_dir.glob("*.shp"))
    if not shp_files:
        zip_path = Path(paths["natural_earth_countries_zip"])
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing Natural Earth archive: {zip_path}")
        countries_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(countries_dir)
        shp_files = sorted(countries_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No Natural Earth shapefile found in {countries_dir}")

    ne_countries = gpd.read_file(shp_files[0]).to_crs("EPSG:4326")
    ukraine_shape = ne_countries[
        (ne_countries.get("ADMIN") == "Ukraine")
        | (ne_countries.get("SOVEREIGNT") == "Ukraine")
        | (ne_countries.get("NAME") == "Ukraine")
    ].copy()
    russia_labels = {"Russia", "Russian Federation"}
    russia_shape = ne_countries[
        ne_countries.get("ADMIN", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
        | ne_countries.get("SOVEREIGNT", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
        | ne_countries.get("NAME", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
        | ne_countries.get("NAME_LONG", pd.Series(index=ne_countries.index, dtype=object)).isin(russia_labels)
    ].copy()
    if ukraine_shape.empty or russia_shape.empty:
        raise ValueError("Could not reconstruct Ukraine/Russia geometry from cached Natural Earth data")

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
        raise ValueError("Sovereign-held Ukraine geometry is empty")
    sovereign_geom = gpd.GeoSeries([sovereign_metric], crs=3857).to_crs("EPSG:4326").iloc[0]
    sovereign_shape = gpd.GeoDataFrame(geometry=[sovereign_geom], crs="EPSG:4326")

    frontline_match_km = float(occ_cfg["frontline_match_km"])
    border_match_m = frontline_match_km * 1000.0
    sovereign_border_metric = sovereign_metric.boundary
    frontline_boundary_metric = sovereign_border_metric.intersection(occupied_metric.buffer(border_match_m))
    if frontline_boundary_metric.is_empty:
        frontline_boundary_metric = occupied_metric.boundary

    ukraine_russia_border_metric = ukraine_metric.boundary.intersection(russia_metric.boundary.buffer(border_match_m))
    if ukraine_russia_border_metric.is_empty:
        ukraine_russia_border_metric = ukraine_metric.boundary.intersection(russia_metric.buffer(border_match_m))

    launch_interface_metric = unary_union(
        [geom for geom in [frontline_boundary_metric, ukraine_russia_border_metric] if not geom.is_empty]
    )
    if launch_interface_metric.is_empty:
        launch_interface_metric = frontline_boundary_metric

    return {
        "snapshot_date": snapshot_date,
        "occupied_geom": occupied_geom,
        "occupied_gs": gpd.GeoSeries([occupied_geom], crs="EPSG:4326"),
        "ukraine_shape": ukraine_shape,
        "sovereign_geom": sovereign_geom,
        "sovereign_shape": sovereign_shape,
        "sovereign_metric": sovereign_metric,
        "sovereign_border_metric": sovereign_border_metric,
        "frontline_boundary_metric": frontline_boundary_metric,
        "frontline_boundary_gs": gpd.GeoSeries([frontline_boundary_metric], crs=3857).to_crs("EPSG:4326"),
        "ukraine_russia_border_metric": ukraine_russia_border_metric,
        "ukraine_russia_border_gs": gpd.GeoSeries([ukraine_russia_border_metric], crs=3857).to_crs("EPSG:4326"),
        "launch_interface_metric": launch_interface_metric,
        "launch_interface_gs": gpd.GeoSeries([launch_interface_metric], crs=3857).to_crs("EPSG:4326"),
    }
