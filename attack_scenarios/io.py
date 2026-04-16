from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import ScenarioParameters


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_base_bundle(bundle_path: str | Path):
    from synthetic_data.pipeline import load_saved_synthetic_bundle
    return load_saved_synthetic_bundle(bundle_path)


def sanitize_graph_for_graphml(graph: nx.Graph) -> nx.Graph:
    clean_graph = graph.copy()
    for _, attrs in clean_graph.nodes(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = _sanitize_graphml_value(value)
    for _, _, attrs in clean_graph.edges(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = _sanitize_graphml_value(value)
    for key, value in list(clean_graph.graph.items()):
        clean_graph.graph[key] = _sanitize_graphml_value(value)
    return clean_graph


def _sanitize_graphml_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_serializable(value), sort_keys=True)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return -1.0
    return value


def _to_serializable(value):
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
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


def write_scenario_outputs(
    params: ScenarioParameters,
    bundle: dict,
) -> dict:
    output_dir = ensure_dir(params.output_dir)

    graphml_path = output_dir / "scenario_graph.graphml"
    nx.write_graphml(sanitize_graph_for_graphml(bundle["graphs"]["scenario_graph"]), graphml_path)

    bundle_pickle_path = output_dir / "scenario_bundle.pkl"
    with bundle_pickle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    edge_impacts_path = output_dir / "edge_impacts.csv"
    bundle["edge_impacts"].to_csv(edge_impacts_path, index=False)

    locations_path = output_dir / "selected_locations.csv"
    pd.DataFrame(bundle["selected_locations"]).to_csv(locations_path, index=False)

    strike_events_path = output_dir / "strike_events.csv"
    pd.DataFrame(bundle["strike_events"]).to_csv(strike_events_path, index=False)

    c_ij_path = output_dir / "c_ij_costs.csv"
    bundle["c_ij"].to_csv(c_ij_path, index=False)

    summary_path = output_dir / "scenario_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(bundle["summary"]), handle, indent=2, sort_keys=True)

    return {
        "output_dir": output_dir,
        "graphml": graphml_path,
        "bundle_pickle": bundle_pickle_path,
        "edge_impacts_csv": edge_impacts_path,
        "selected_locations_csv": locations_path,
        "strike_events_csv": strike_events_path,
        "c_ij_csv": c_ij_path,
        "summary_json": summary_path,
    }


def load_saved_attack_scenario(bundle_path: str | Path):
    path = Path(bundle_path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_cost_matrix(graph: nx.Graph) -> pd.DataFrame:
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="travel_time"))
    rows: list[dict] = []
    for source in sorted(graph.nodes()):
        source_lengths = lengths.get(source, {})
        for target in sorted(graph.nodes()):
            rows.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "c_ij": float(source_lengths.get(target, np.inf)),
                }
            )
    return pd.DataFrame(rows)
