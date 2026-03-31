from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "synthetic_data" / "config.yaml"


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: str | Path | None = None) -> dict:
    path = _resolve_path(PROJECT_ROOT, config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    paths = config.setdefault("paths", {})
    for key in [
        "data_dir",
        "cache_dir",
        "output_dir",
        "figures_dir",
        "acled_csv",
        "raw_graph_cache",
        "occupied_cache_gz",
        "occupied_cache_geojson",
        "coastline_cache",
        "boundary_cache",
        "natural_earth_countries_zip",
        "natural_earth_countries_dir",
        "filtered_raw_graph",
        "adaptive_graph",
        "demand_nodes_csv",
        "summary_json",
        "bundle_pickle",
        "config_used_yaml",
        "occupied_filter_figure",
        "adaptive_figure",
        "edge_metric_figure",
        "cost_figure",
    ]:
        if key in paths:
            paths[key] = _resolve_path(PROJECT_ROOT, paths[key])

    config["_project_root"] = PROJECT_ROOT
    config["_config_path"] = path
    return config
