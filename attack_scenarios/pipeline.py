from __future__ import annotations

from pathlib import Path

from .config import apply_parameter_overrides, load_scenario_parameters
from .geometry import load_attack_geography
from .io import load_base_bundle, load_saved_attack_scenario, write_scenario_outputs
from .model import generate_attack_bundle


def generate_attack_scenario(
    config_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    attack_mode: str | None = None,
    base_budget: float | None = None,
    strike_lat: float | None = None,
    strike_lon: float | None = None,
    scenario_id: str | None = None,
    output_root: str | Path | None = None,
    save_outputs: bool | None = None,
    generate_visual: bool | None = None,
) -> dict:
    params = load_scenario_parameters(config_path)
    apply_parameter_overrides(
        params,
        {
            "bundle_path": bundle_path,
            "attack_mode": attack_mode,
            "base_budget": base_budget,
            "strike_lat": strike_lat,
            "strike_lon": strike_lon,
            "scenario_id": scenario_id,
            "output_root": output_root,
            "save_outputs": save_outputs,
            "generate_visual": generate_visual,
        },
    )

    base_bundle = load_base_bundle(params.bundle_path)
    geography = load_attack_geography(base_bundle["config"])
    attack_bundle = generate_attack_bundle(
        base_graph=base_bundle["graphs"]["adaptive_graph"],
        demand_nodes=base_bundle.get("demand_nodes"),
        params=params,
        geography=geography,
    )

    output_paths = {}
    if params.save_outputs:
        output_paths = write_scenario_outputs(params, attack_bundle)

    if params.generate_visual:
        from .visualization import plot_attack_scenario

        figure_path = params.output_dir / "scenario_visualization.png"
        plot_attack_scenario(attack_bundle, figure_path)
        output_paths["figure"] = figure_path

    attack_bundle["output_paths"] = {key: str(value) for key, value in output_paths.items()}
    return attack_bundle
