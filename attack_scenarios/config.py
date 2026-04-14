from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


DEFAULT_BUNDLE_PATH = Path("data/synthetic/ukraine_synthetic_bundle.pkl")
DEFAULT_OUTPUT_ROOT = Path("data/attack_scenarios")
DEFAULT_CONFIG_PATH = Path("attack_scenarios/config.yaml")


@dataclass(slots=True)
class StrikeCenter:
    lat: float
    lon: float
    label: str = "custom_strike_center"
    source: str = "manual"

    @classmethod
    def from_dict(cls, payload: dict | None, default_label: str = "custom_strike_center", source: str = "manual"):
        payload = payload or {}
        return cls(
            lat=float(payload["lat"]),
            lon=float(payload["lon"]),
            label=str(payload.get("label", default_label)),
            source=str(payload.get("source", source)),
        )


@dataclass(slots=True)
class AttackTypeParameters:
    enabled: bool = True
    radius_km: float = 12.0
    max_strikes: int = 6
    min_strike_separation_km: float = 25.0
    manual_strike_centers: list[StrikeCenter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.radius_km <= 0:
            raise ValueError("attack radius_km must be positive")
        if self.max_strikes < 0:
            raise ValueError("max_strikes must be non-negative")
        if self.min_strike_separation_km < 0:
            raise ValueError("min_strike_separation_km must be non-negative")

    @classmethod
    def from_dict(cls, payload: dict | None, defaults: "AttackTypeParameters" | None = None):
        payload = payload or {}
        base = defaults or cls()
        centers = payload.get("manual_strike_centers", [])
        return cls(
            enabled=bool(payload.get("enabled", base.enabled)),
            radius_km=float(payload.get("radius_km", base.radius_km)),
            max_strikes=int(payload.get("max_strikes", base.max_strikes)),
            min_strike_separation_km=float(
                payload.get("min_strike_separation_km", base.min_strike_separation_km)
            ),
            manual_strike_centers=[
                StrikeCenter.from_dict(center, default_label=f"manual_center_{idx + 1}", source="manual")
                for idx, center in enumerate(centers)
            ],
        )


@dataclass(slots=True)
class BudgetParameters:
    location_cost: float = 1.0
    missile_cost: float = 1.0
    bomb_cost: float = 0.5
    location_link_radius_km: float = 100.0
    max_locations: int = 6
    min_location_separation_km: float = 60.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.location_cost <= 0:
            raise ValueError("location_cost must be positive")
        if self.missile_cost <= 0:
            raise ValueError("missile_cost must be positive")
        if self.bomb_cost <= 0:
            raise ValueError("bomb_cost must be positive")
        if self.location_link_radius_km <= 0:
            raise ValueError("location_link_radius_km must be positive")
        if self.max_locations < 0:
            raise ValueError("max_locations must be non-negative")
        if self.min_location_separation_km < 0:
            raise ValueError("min_location_separation_km must be non-negative")


@dataclass(slots=True)
class ScenarioParameters:
    attack_mode: str = "combo"
    base_budget: float = 2.0
    scenario_id: str | None = None
    depth_penalty_gamma: float = 2.0
    defense_alpha: float = 0.7
    defense_beta: float = 0.3
    minimum_defense_score: float = 0.05
    theta_degrade: float = 0.5
    theta_remove: float = 1.5
    degrade_multiplier: float = 3.0
    bomb_reduction_factor: float = 5.0
    candidate_grid_spacing_km: float = 30.0
    edge_midpoint_candidate_count: int = 80
    remove_isolated_nodes: bool = True
    bundle_path: Path = DEFAULT_BUNDLE_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    generate_visual: bool = True
    save_outputs: bool = True
    budget: BudgetParameters = field(default_factory=BudgetParameters)
    missile: AttackTypeParameters = field(
        default_factory=lambda: AttackTypeParameters(
            enabled=True,
            radius_km=12.0,
            max_strikes=6,
            min_strike_separation_km=20.0,
        )
    )
    bomb: AttackTypeParameters = field(
        default_factory=lambda: AttackTypeParameters(
            enabled=True,
            radius_km=30.0,
            max_strikes=8,
            min_strike_separation_km=30.0,
        )
    )

    def __post_init__(self) -> None:
        self.attack_mode = str(self.attack_mode).lower()
        if self.attack_mode not in {"missile", "bomb", "combo"}:
            raise ValueError("attack_mode must be one of: missile, bomb, combo")
        if self.base_budget < 0:
            raise ValueError("base_budget must be non-negative")
        if self.depth_penalty_gamma < 1:
            raise ValueError("depth_penalty_gamma must be at least 1")
        if self.minimum_defense_score <= 0:
            raise ValueError("minimum_defense_score must be positive")
        if self.degrade_multiplier < 1:
            raise ValueError("degrade_multiplier must be at least 1")
        if self.bomb_reduction_factor <= 0:
            raise ValueError("bomb_reduction_factor must be positive")
        if self.candidate_grid_spacing_km <= 0:
            raise ValueError("candidate_grid_spacing_km must be positive")
        if self.edge_midpoint_candidate_count < 0:
            raise ValueError("edge_midpoint_candidate_count must be non-negative")
        if self.theta_remove <= self.theta_degrade:
            raise ValueError("theta_remove must be greater than theta_degrade")
        self.bundle_path = Path(self.bundle_path)
        self.output_root = Path(self.output_root)
        if self.scenario_id is None:
            budget_label = str(self.base_budget).replace(".", "p")
            self.scenario_id = f"{self.attack_mode}_K{budget_label}"
        self._validate_attack_mix()

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.scenario_id

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["bundle_path"] = str(self.bundle_path)
        payload["output_root"] = str(self.output_root)
        return payload

    def _validate_attack_mix(self) -> None:
        active = self.active_attack_types()
        if not active:
            raise ValueError("No attack types are active for the chosen attack_mode/configuration")

    def active_attack_types(self) -> list[str]:
        if self.attack_mode == "missile":
            return ["missile"] if self.missile.enabled else []
        if self.attack_mode == "bomb":
            return ["bomb"] if self.bomb.enabled else []
        active: list[str] = []
        if self.missile.enabled:
            active.append("missile")
        if self.bomb.enabled:
            active.append("bomb")
        return active


def load_scenario_parameters(config_path: str | Path | None = None, overrides: dict | None = None) -> ScenarioParameters:
    path = Path(config_path or DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    paths = raw.get("paths", {})
    scenario = raw.get("scenario", {})
    attack_model = raw.get("attack_model", {})
    budget = raw.get("budget", {})
    candidate_generation = raw.get("candidate_generation", {})
    attacks = raw.get("attacks", {})
    visualization = raw.get("visualization", {})

    default_params = ScenarioParameters()
    params = ScenarioParameters(
        attack_mode=str(scenario.get("attack_mode", "combo")),
        base_budget=float(scenario.get("base_budget", 2.0)),
        scenario_id=scenario.get("scenario_id"),
        depth_penalty_gamma=float(attack_model.get("depth_penalty_gamma", 2.0)),
        defense_alpha=float(attack_model.get("defense_alpha", 0.7)),
        defense_beta=float(attack_model.get("defense_beta", 0.3)),
        minimum_defense_score=float(attack_model.get("minimum_defense_score", 0.05)),
        theta_degrade=float(attack_model.get("theta_degrade", 0.5)),
        theta_remove=float(attack_model.get("theta_remove", 1.5)),
        degrade_multiplier=float(attack_model.get("degrade_multiplier", 3.0)),
        bomb_reduction_factor=float(attack_model.get("bomb_reduction_factor", 5.0)),
        candidate_grid_spacing_km=float(candidate_generation.get("candidate_grid_spacing_km", 30.0)),
        edge_midpoint_candidate_count=int(candidate_generation.get("edge_midpoint_candidate_count", 80)),
        remove_isolated_nodes=bool(scenario.get("remove_isolated_nodes", True)),
        bundle_path=Path(paths.get("bundle_path", DEFAULT_BUNDLE_PATH)),
        output_root=Path(paths.get("output_root", DEFAULT_OUTPUT_ROOT)),
        generate_visual=bool(visualization.get("enabled", True)),
        save_outputs=bool(paths.get("save_outputs", True)),
        budget=BudgetParameters(
            location_cost=float(budget.get("location_cost", default_params.budget.location_cost)),
            missile_cost=float(budget.get("missile_cost", default_params.budget.missile_cost)),
            bomb_cost=float(budget.get("bomb_cost", default_params.budget.bomb_cost)),
            location_link_radius_km=float(
                budget.get("location_link_radius_km", default_params.budget.location_link_radius_km)
            ),
            max_locations=int(budget.get("max_locations", default_params.budget.max_locations)),
            min_location_separation_km=float(
                budget.get("min_location_separation_km", default_params.budget.min_location_separation_km)
            ),
            random_seed=int(budget.get("random_seed", default_params.budget.random_seed)),
        ),
        missile=AttackTypeParameters.from_dict(attacks.get("missile"), default_params.missile),
        bomb=AttackTypeParameters.from_dict(attacks.get("bomb"), default_params.bomb),
    )
    apply_parameter_overrides(params, overrides or {})
    return params


def apply_parameter_overrides(params: ScenarioParameters, overrides: dict) -> ScenarioParameters:
    if not overrides:
        return params

    for key, value in overrides.items():
        if value is None:
            continue
        if key == "scenario_id":
            params.scenario_id = str(value)
        elif key == "bundle_path":
            params.bundle_path = Path(value)
        elif key == "output_root":
            params.output_root = Path(value)
        elif key == "attack_mode":
            params.attack_mode = str(value).lower()
        elif key == "base_budget":
            params.base_budget = float(value)
        elif key == "generate_visual":
            params.generate_visual = bool(value)
        elif key == "save_outputs":
            params.save_outputs = bool(value)
        elif key == "strike_lat" or key == "strike_lon":
            continue
        else:
            raise ValueError(f"Unsupported override key: {key}")

    strike_lat = overrides.get("strike_lat")
    strike_lon = overrides.get("strike_lon")
    if strike_lat is not None and strike_lon is not None:
        manual_center = StrikeCenter(
            lat=float(strike_lat),
            lon=float(strike_lon),
            label="cli_manual_center",
            source="manual_cli",
        )
        active = params.active_attack_types()
        for attack_type in active:
            getattr(params, attack_type).manual_strike_centers = [manual_center]
            getattr(params, attack_type).max_strikes = max(1, getattr(params, attack_type).max_strikes)

    params.__post_init__()
    return params
