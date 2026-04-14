from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from ..data import MIPInstance
from ..scenarios import ScenarioData


def build_robust_model(
    instance: MIPInstance,
    scenarios: list[ScenarioData],
    delta: float = 0.05,
    M_cap: float | None = None,
    M_service: float = 100.0,
    verbose: bool = False,
) -> tuple[gp.Model, dict]:
    """Build the scenario-aware (robust) chance-constraint MIP.

    Formulation
    -----------
    min  Σⱼ (aⱼ yⱼ + bⱼ uⱼ)

    [1]  uⱼ ≤ M_cap · yⱼ                                    ∀j ∈ N
    [2]  Σᵢ x^s_{ij} ≤ uⱼ                                   ∀j ∈ Nₛ, ∀s ∈ S
    [3]  Σⱼ x^s_{ij} = dᵢ                                   ∀i ∈ D, ∀s ∈ S
    [4]  (1/|N|) Σᵢⱼ c^s_{ij} x^s_{ij} ≤ Tₛ + M_service·wₛ  ∀s ∈ S
    [5]  Σₛ wₛ ≤ |S| · δ                                     (violation budget)

    Constraints [4]+[5] together enforce the chance constraint:
        P[ avg routing cost ≤ T ] ≥ 1 − δ

    Parameters
    ----------
    instance   : MIPInstance
    scenarios  : list of ScenarioData from load_scenario_batch()
    delta      : max fraction of scenarios allowed to violate threshold (e.g. 0.05)
    M_cap      : big-M for capacity link; defaults to total number of nodes
    M_service  : big-M for service slack; should be >> max possible routing cost
    verbose    : whether to show Gurobi output

    Returns
    -------
    model     : solved or unsolved gp.Model
    variables : dict with keys ``y``, ``u``, ``x``, ``w``
                x is keyed by (scenario_id, demand_node, hub_node)
                w is keyed by scenario_id
    """
    D = instance.D
    N = instance.N
    demand = instance.demand
    a = instance.a
    b = instance.b

    if M_cap is None:
        M_cap = float(len(N))

    model = gp.Model("robust_facility_location")
    model.Params.OutputFlag = int(verbose)

    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    u = model.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
    w = model.addVars([s.id for s in scenarios], vtype=GRB.BINARY, name="w")

    # x^s_{ij} only created for (i,j) pairs that are reachable in scenario s
    x: dict[tuple[str, int, int], gp.Var] = {}
    for s in scenarios:
        for i in D:
            for j in s.surviving_nodes:
                if (i, j) in s.c:
                    x[(s.id, i, j)] = model.addVar(
                        lb=0.0,
                        vtype=GRB.CONTINUOUS,
                        name=f"x_{s.id}_{i}_{j}",
                    )
    model.update()

    model.setObjective(
        gp.quicksum(a[j] * y[j] + b[j] * u[j] for j in N),
        GRB.MINIMIZE,
    )

    # [1] capacity requires open hub
    model.addConstrs((u[j] <= M_cap * y[j] for j in N), name="cap_open")

    for s in scenarios:
        sid = s.id
        Ns = s.surviving_nodes

        # [2] hub capacity respected per scenario
        model.addConstrs(
            (
                gp.quicksum(x[(sid, i, j)] for i in D if (sid, i, j) in x) <= u[j]
                for j in Ns
            ),
            name=f"cap_flow_{sid}",
        )

        # [3] all demand fully served per scenario
        model.addConstrs(
            (
                gp.quicksum(x[(sid, i, j)] for j in Ns if (sid, i, j) in x) == demand[i]
                for i in D
            ),
            name=f"demand_{sid}",
        )

        # [4] per-scenario service threshold (chance constraint big-M)
        model.addConstr(
            (1.0 / len(N))
            * gp.quicksum(
                s.c[(i, j)] * x[(sid, i, j)]
                for i in D
                for j in Ns
                if (sid, i, j) in x
            )
            <= s.T + M_service * w[sid],
            name=f"service_{sid}",
        )

    # [5] violation budget
    model.addConstr(
        gp.quicksum(w[s.id] for s in scenarios) <= len(scenarios) * delta,
        name="scenario_budget",
    )

    return model, {"y": y, "u": u, "x": x, "w": w}
