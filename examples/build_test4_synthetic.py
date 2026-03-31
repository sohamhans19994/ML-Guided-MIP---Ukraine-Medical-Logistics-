from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_data import build_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Recreate the synthetic-data generation flow from test4.ipynb.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "synthetic_data" / "config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip figure generation.",
    )
    args = parser.parse_args()

    bundle = build_synthetic_dataset(
        config_path=args.config,
        generate_visuals=not args.no_visuals,
    )

    summary = bundle["summary"]
    coarse_graph = bundle["graphs"]["adaptive_graph"]
    demand_nodes = bundle["demand_nodes"]
    cost_details = bundle["metadata"]["cost_details"]
    config = bundle["config"]

    edge_path_counts = [
        data.get("abstracted_path_count", 1) for _, _, data in coarse_graph.edges(data=True)
    ] or [1]
    top_edges = sorted(
        coarse_graph.edges(data=True),
        key=lambda edge: edge[2].get("abstracted_path_count", 1),
        reverse=True,
    )[:8]
    cheapest_nodes = sorted(
        coarse_graph.nodes(),
        key=lambda node: coarse_graph.nodes[node].get("a_i", float("inf")),
    )[:5]

    print("\nSynthetic data build complete")
    print("=" * 80)
    print(f"Config used: {args.config}")
    print(f"Snapshot date: {summary['snapshot_date']}")
    print(f"Raw graph: {summary['raw_graph']['nodes']} nodes, {summary['raw_graph']['edges']} edges")
    print(
        "Filtered graph: "
        f"{summary['filtered_graph']['nodes']} nodes, {summary['filtered_graph']['edges']} edges "
        f"(removed occupied={summary['filtered_graph']['removed_occupied_nodes']}, "
        f"removed Russia-side={summary['filtered_graph']['removed_russia_side_nodes']})"
    )
    print(
        "Adaptive CG: "
        f"{summary['adaptive_graph']['nodes']} nodes, {summary['adaptive_graph']['edges']} edges"
    )
    print(f"Raw zone counts: {summary['adaptive_graph']['raw_zone_counts']}")
    print(f"Coarse zone counts: {summary['adaptive_graph']['coarse_zone_counts']}")
    print(
        "Abstracted paths per coarse edge: "
        f"min={min(edge_path_counts)}, "
        f"mean={sum(edge_path_counts) / len(edge_path_counts):.2f}, "
        f"max={max(edge_path_counts)}"
    )
    if "road_length_km" in summary["adaptive_graph"]:
        print(
            "Coarse-edge road length (km): "
            f"min={summary['adaptive_graph']['road_length_km']['min']:.2f}, "
            f"mean={summary['adaptive_graph']['road_length_km']['mean']:.2f}, "
            f"max={summary['adaptive_graph']['road_length_km']['max']:.2f}"
        )
    if "travel_time_hr" in summary["adaptive_graph"]:
        print(
            "Coarse-edge travel time (hr): "
            f"min={summary['adaptive_graph']['travel_time_hr']['min']:.2f}, "
            f"mean={summary['adaptive_graph']['travel_time_hr']['mean']:.2f}, "
            f"max={summary['adaptive_graph']['travel_time_hr']['max']:.2f}"
        )
    print(
        "Demand nodes: "
        f"{summary['demand_nodes']['count']} rows, "
        f"{summary['demand_nodes']['unique_coarse_nodes']} unique coarse nodes"
    )
    print(
        f"a_i range: {summary['costs']['a_i']['min']:.2f} to {summary['costs']['a_i']['max']:.2f}, "
        f"mean={summary['costs']['a_i']['mean']:.2f}"
    )
    print(
        f"b_i range: {summary['costs']['b_i']['min']:.2f} to {summary['costs']['b_i']['max']:.2f}, "
        f"mean={summary['costs']['b_i']['mean']:.2f}"
    )
    print(
        f"cost_score range: {summary['costs']['cost_score']['min']:.2f} to {summary['costs']['cost_score']['max']:.2f}, "
        f"mean={summary['costs']['cost_score']['mean']:.2f}"
    )

    print("\nTop coarse edges by abstracted path count:")
    for u, v, data in top_edges:
        print(
            f"  ({u}, {v}) paths={data.get('abstracted_path_count', 1)}, "
            f"travel_time_hr={data.get('travel_time', 0.0) / 3600.0:.2f}, "
            f"road_km={data.get('length_m', float('nan')) / 1000.0:.1f}, "
            f"source={data.get('metric_source', 'unknown')}"
        )

    print("\nTop 5 cheapest coarse nodes by a_i:")
    for node in cheapest_nodes:
        attrs = coarse_graph.nodes[node]
        print(
            f"  node={node} a_i={attrs.get('a_i'):.3f} b_i={attrs.get('b_i'):.3f} "
            f"members={attrs.get('member_count')} zone={attrs.get('zone')} "
            f"frontline_km={attrs.get('territory_frontline_distance_km', float('nan')):.2f} "
            f"member_component={attrs.get('member_cost_component', 0.0):.3f} "
            f"support_component={attrs.get('edge_support_cost_component', 0.0):.3f} "
            f"raw_frontline={attrs.get('raw_frontline_cost_component', 0.0):.3f} "
            f"frontline_component={attrs.get('frontline_cost_component', 0.0):.3f}"
        )

    print("\nSaved outputs:")
    print(f"  bundle: {config['paths']['bundle_pickle']}")
    print(f"  filtered graph: {config['paths']['filtered_raw_graph']}")
    print(f"  adaptive graph: {config['paths']['adaptive_graph']}")
    print(f"  demand nodes: {config['paths']['demand_nodes_csv']}")
    print(f"  summary: {config['paths']['summary_json']}")
    if not args.no_visuals:
        print(f"  occupied filter figure: {config['paths']['occupied_filter_figure']}")
        print(f"  adaptive figure: {config['paths']['adaptive_figure']}")
        print(f"  edge metric figure: {config['paths']['edge_metric_figure']}")
        print(f"  cost figure: {config['paths']['cost_figure']}")


if __name__ == "__main__":
    main()
