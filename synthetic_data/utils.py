from __future__ import annotations

import json
import math
import pickle
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import yaml


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(config: dict) -> None:
    for key in ["data_dir", "cache_dir", "output_dir", "figures_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def migrate_if_missing(target: str | Path, candidates: list[str | Path]) -> Path:
    target = Path(target)
    if target.exists():
        return target

    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            ensure_parent(target)
            shutil.copy2(candidate, target)
            return target
    return target


def _clean_scalar_for_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def to_serializable(value):
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    return _clean_scalar_for_json(value)


def sanitize_graph_for_graphml(graph: nx.Graph) -> nx.Graph:
    clean_graph = graph.copy()
    for _, attrs in clean_graph.nodes(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = sanitize_graphml_value(value)
    for _, _, attrs in clean_graph.edges(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = sanitize_graphml_value(value)
    for key, value in list(clean_graph.graph.items()):
        clean_graph.graph[key] = sanitize_graphml_value(value)
    return clean_graph


def sanitize_graphml_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(to_serializable(value), sort_keys=True)
    if hasattr(value, "wkt"):
        return value.wkt
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return -1.0
    return value


def write_json(path: str | Path, payload: dict) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def write_yaml(path: str | Path, payload: dict) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_serializable(payload), handle, sort_keys=False)


def write_pickle(path: str | Path, payload) -> None:
    path = ensure_parent(path)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def read_pickle(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
