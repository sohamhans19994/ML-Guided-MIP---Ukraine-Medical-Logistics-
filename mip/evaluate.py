"""Fixed-design second-stage evaluator.

Given a FIXED first-stage design (which hubs are open `y` and their capacities
`u`), evaluate how that design performs on a single attack scenario by solving
only the second-stage routing LP — i.e. *without* re-optimising the hub/capacity
decisions. This is the separation oracle for the C&CG loop in
``mip.enrichment``: it tells us whether a scenario is a *violator* of the current
design's service guarantee.

A scenario is a **violator** when, even under the cheapest feasible routing the
fixed design allows, the average routing cost exceeds the scenario threshold
``T_s`` — or when the design cannot serve the demand at all (some demand node
cannot reach any open, surviving hub), which is the most severe violation.

The per-scenario service value matches the master's constraint [4] in
``mip.models.robust`` exactly: ``(1/|N|) Σ_{i,j} c^s_{ij} x^s_{ij}`` (see also
``mip.solution.extract_robust_solution``), so results are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .data import MIPInstance
from .scenarios import ScenarioData


@dataclass
class EvalResult:
    """Outcome of evaluating one fixed design on one scenario."""

    scenario_id: str
    T: float
    lhs_service: float   # (1/|N|) Σ c·x under cheapest feasible routing; inf if unservable
    feasible: bool       # could the design serve all demand in this scenario?
    violated: bool       # not feasible, or lhs_service > T
    severity: float      # lhs_service - T  (+inf if unservable) — used to rank violators


def evaluate_design_on_scenario(
    instance: MIPInstance,
    scenario: ScenarioData,
    y_vals: dict[int, int],
    u_vals: dict[int, float],
    *,
    tol: float = 1e-6,
) -> EvalResult:
    """Solve the routing-only LP for a fixed design on one scenario.

    Parameters
    ----------
    instance : MIPInstance
    scenario : ScenarioData (post-attack graph, surviving nodes, cost matrix c, threshold T)
    y_vals   : hub_id -> {0,1} open/close decision (first stage, fixed)
    u_vals   : hub_id -> capacity provisioned (first stage, fixed)
    tol      : numerical tolerance for the violation comparison

    Returns
    -------
    EvalResult
    """
    D = instance.D
    N = instance.N
    demand = instance.demand
    s = scenario

    # Hubs we may route to: open (y=1), surviving in this scenario, with capacity.
    open_hubs = [
        j for j in s.surviving_nodes
        if int(round(y_vals.get(j, 0))) == 1
    ]
    open_set = set(open_hubs)

    model = gp.Model("eval_routing")
    model.Params.OutputFlag = 0

    # x^s_{ij} only for (demand i, open surviving hub j) pairs that are reachable.
    x: dict[tuple[int, int], gp.Var] = {}
    for i in D:
        for j in open_hubs:
            if (i, j) in s.c:
                x[(i, j)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")
    model.update()

    # [3] serve all demand. If a demand node has no reachable open hub the LHS is
    #     empty -> 0 == demand[i] -> infeasible (the "unservable" case).
    for i in D:
        model.addConstr(
            gp.quicksum(x[(i, j)] for j in open_hubs if (i, j) in x) == demand[i],
            name=f"demand_{i}",
        )

    # [2] respect fixed capacity at each open hub.
    for j in open_hubs:
        model.addConstr(
            gp.quicksum(x[(i, j)] for i in D if (i, j) in x) <= u_vals.get(j, 0.0),
            name=f"cap_{j}",
        )

    # Cheapest feasible routing — the best the fixed design can do.
    model.setObjective(
        gp.quicksum(s.c[(i, j)] * x[(i, j)] for (i, j) in x),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.SolCount == 0:
        # Infeasible: the open hubs cannot serve all demand in this scenario.
        return EvalResult(
            scenario_id=s.id,
            T=s.T,
            lhs_service=float("inf"),
            feasible=False,
            violated=True,
            severity=float("inf"),
        )

    lhs = (1.0 / len(N)) * model.ObjVal
    violated = lhs > s.T + tol
    return EvalResult(
        scenario_id=s.id,
        T=s.T,
        lhs_service=lhs,
        feasible=True,
        violated=violated,
        severity=lhs - s.T,
    )


def evaluate_design_on_batch(
    instance: MIPInstance,
    scenarios: list[ScenarioData],
    y_vals: dict[int, int],
    u_vals: dict[int, float],
    *,
    tol: float = 1e-6,
) -> list[EvalResult]:
    """Evaluate one fixed design across a batch of scenarios (serial).

    Note: each call builds a small independent LP. This is fine for batches of a
    few hundred to ~1000 (LPs have ~|D|·|open hubs| vars and solve in ms). If this
    ever dominates runtime, parallelise across scenarios — the calls are independent.
    """
    return [
        evaluate_design_on_scenario(instance, s, y_vals, u_vals, tol=tol)
        for s in scenarios
    ]


def violation_rate(results: list[EvalResult]) -> float:
    """Fraction of scenarios the design violates (cost over threshold or unservable)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.violated) / len(results)
