from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import MIPInstance
from .scenarios import ScenarioData


@dataclass
class DeterministicResult:
    """Extracted solution from the deterministic MIP."""

    model_status: int
    obj_val: float
    open_hubs: list[int]
    hub_capacity: dict[int, float]
    used_capacity: dict[int, float]
    assignments: pd.DataFrame        # rows: demand_node, hub_node, flow, travel_time_hr
    avg_travel_time: float
    max_travel_time: float
    fixed_cost: float
    capacity_cost: float
    T: float


@dataclass
class RobustResult:
    """Extracted solution from the robust (scenario-aware) MIP."""

    model_status: int
    obj_val: float
    open_hubs: list[int]
    hub_capacity: dict[int, float]
    used_capacity: dict[int, float]   # aggregated across all scenarios
    scenario_service_df: pd.DataFrame  # per-scenario: K, T_s, lhs_service, violated
    violated_scenarios: list[str]
    fixed_cost: float
    capacity_cost: float
    delta: float


def extract_deterministic_solution(
    model,
    variables: dict,
    instance: MIPInstance,
    c: dict[tuple[int, int], float],
    T: float,
) -> DeterministicResult:
    """Extract a :class:`DeterministicResult` from a solved Gurobi model.

    Returns a result with ``model_status`` set and all numeric fields set to
    0/empty if no feasible solution was found.
    """
    y, u, x = variables["y"], variables["u"], variables["x"]
    D, N = instance.D, instance.N
    a, b = instance.a, instance.b

    if model.SolCount == 0:
        return DeterministicResult(
            model_status=model.Status,
            obj_val=float("inf"),
            open_hubs=[],
            hub_capacity={},
            used_capacity={},
            assignments=pd.DataFrame(),
            avg_travel_time=float("inf"),
            max_travel_time=float("inf"),
            fixed_cost=0.0,
            capacity_cost=0.0,
            T=T,
        )

    open_hubs = [j for j in N if y[j].X > 0.5]
    hub_capacity = {j: u[j].X for j in open_hubs}
    used_capacity = {j: sum(x[i, j].X for i in D) for j in open_hubs}

    rows = []
    for i in D:
        for j in N:
            flow = x[i, j].X
            if flow > 1e-6:
                rows.append({"demand_node": i, "hub_node": j, "flow": flow, "travel_time_hr": c[(i, j)]})
    assignments = pd.DataFrame(rows).sort_values(["demand_node", "flow"], ascending=[True, False])

    total_demand = float(sum(instance.demand.values()))
    avg_tt = sum(c[(i, j)] * x[i, j].X for i in D for j in N) / total_demand
    max_tt = max((r["travel_time_hr"] for r in rows), default=0.0)

    return DeterministicResult(
        model_status=model.Status,
        obj_val=model.ObjVal,
        open_hubs=open_hubs,
        hub_capacity=hub_capacity,
        used_capacity=used_capacity,
        assignments=assignments,
        avg_travel_time=avg_tt,
        max_travel_time=max_tt,
        fixed_cost=sum(a[j] * y[j].X for j in N),
        capacity_cost=sum(b[j] * u[j].X for j in N),
        T=T,
    )


def extract_robust_solution(
    model,
    variables: dict,
    instance: MIPInstance,
    scenarios: list[ScenarioData],
    delta: float,
) -> RobustResult:
    """Extract a :class:`RobustResult` from a solved Gurobi robust model."""
    y, u, x, w = variables["y"], variables["u"], variables["x"], variables["w"]
    N = instance.N
    D = instance.D
    a, b = instance.a, instance.b

    if model.SolCount == 0:
        return RobustResult(
            model_status=model.Status,
            obj_val=float("inf"),
            open_hubs=[],
            hub_capacity={},
            used_capacity={},
            scenario_service_df=pd.DataFrame(),
            violated_scenarios=[],
            fixed_cost=0.0,
            capacity_cost=0.0,
            delta=delta,
        )

    open_hubs = [j for j in N if y[j].X > 0.5]
    hub_capacity = {j: u[j].X for j in open_hubs}
    used_capacity = {
        j: sum(
            x[(s.id, i, j)].X
            for s in scenarios
            for i in D
            if (s.id, i, j) in x
        )
        for j in open_hubs
    }

    service_rows = []
    for s in scenarios:
        lhs = (1.0 / len(N)) * sum(
            s.c[(i, j)] * x[(s.id, i, j)].X
            for i in D
            for j in s.surviving_nodes
            if (s.id, i, j) in x
        )
        service_rows.append(
            {
                "scenario": s.id,
                "K": s.K,
                "T_s": s.T,
                "lhs_service": lhs,
                "violated": int(round(w[s.id].X)),
            }
        )

    scenario_service_df = (
        pd.DataFrame(service_rows)
        .sort_values(["K", "scenario"])
        .reset_index(drop=True)
    )
    violated_scenarios = scenario_service_df.loc[
        scenario_service_df["violated"] == 1, "scenario"
    ].tolist()

    return RobustResult(
        model_status=model.Status,
        obj_val=model.ObjVal,
        open_hubs=open_hubs,
        hub_capacity=hub_capacity,
        used_capacity=used_capacity,
        scenario_service_df=scenario_service_df,
        violated_scenarios=violated_scenarios,
        fixed_cost=sum(a[j] * y[j].X for j in N),
        capacity_cost=sum(b[j] * u[j].X for j in N),
        delta=delta,
    )
