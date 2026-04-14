from __future__ import annotations

import argparse

from .pipeline import generate_attack_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate attack scenarios on the saved adaptive Ukraine graph.")
    parser.add_argument("--config", default="attack_scenarios/config.yaml")
    parser.add_argument("--bundle-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--attack-mode", default=None, choices=["missile", "bomb", "combo"])
    parser.add_argument("--base-budget", type=float, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--strike-lat", type=float, default=None)
    parser.add_argument("--strike-lon", type=float, default=None)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-save", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = generate_attack_scenario(
        config_path=args.config,
        bundle_path=args.bundle_path,
        attack_mode=args.attack_mode,
        base_budget=args.base_budget,
        strike_lat=args.strike_lat,
        strike_lon=args.strike_lon,
        scenario_id=args.scenario_id,
        output_root=args.output_root,
        save_outputs=not args.skip_save if args.skip_save else None,
        generate_visual=not args.skip_visual if args.skip_visual else None,
    )
    summary = bundle["summary"]
    print(
        "Attack scenario ready:",
        f"id={summary['scenario_id']},",
        f"locations={summary['location_count']},",
        f"removed_edges={summary['edge_impacts']['removed_edges']},",
        f"degraded_edges={summary['edge_impacts']['degraded_edges']},",
        f"missile_strikes={summary['strike_counts']['missile']},",
        f"bomb_strikes={summary['strike_counts']['bomb']},",
        f"nodes={summary['scenario_graph']['nodes']},",
        f"edges={summary['scenario_graph']['edges']}",
    )
    if bundle.get("output_paths"):
        print("Outputs:")
        for key, value in sorted(bundle["output_paths"].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
