from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from ..data import MIPInstance


def build_deterministic_model(
    instance: MIPInstance,
    c: dict[tuple[int, int], float],
    T: float,
    verbose: bool = False,
) -> tuple[gp.Model, dict]:
    """Build the deterministic (no-scenario) facility-location MIP.

    Formulation
    -----------
    min  Σⱼ (aⱼ yⱼ + bⱼ uⱼ)

    [1]  uⱼ ≤ M · yⱼ                          ∀j ∈ N   (capacity requires open hub)
    [2]  Σⱼ xᵢⱼ ≤ uⱼ                           ∀j ∈ N   (hub capacity respected)
    [3]  Σⱼ xᵢⱼ = dᵢ                           ∀i ∈ D   (all demand fully served)
    [4]  Σᵢⱼ cᵢⱼ xᵢⱼ ≤ T · total_demand                 (weighted-avg travel time ≤ T)

    Parameters
    ----------
    instance : MIPInstance
    c        : cost matrix from compute_cost_matrix()
    T        : weighted-average travel-time threshold (hours)
    verbose  : whether to show Gurobi output

    Returns
    -------
    model    : solved or unsolved gp.Model
    variables: dict with keys ``y``, ``u``, ``x``
    """
    D = instance.D
    N = instance.N
    demand = instance.demand
    a = instance.a
    b = instance.b
    total_demand = float(sum(demand.values()))
    M = total_demand  # big-M for capacity link

    model = gp.Model("deterministic_facility_location")
    model.Params.OutputFlag = int(verbose)

    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    u = model.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
    x = model.addVars(D, N, lb=0.0, vtype=GRB.CONTINUOUS, name="x")

    model.setObjective(
        gp.quicksum(a[j] * y[j] + b[j] * u[j] for j in N),
        GRB.MINIMIZE,
    )

    # [1] capacity requires open hub
    model.addConstrs((u[j] <= M * y[j] for j in N), name="open_capacity_link")
    # [2] hub capacity respected
    model.addConstrs((gp.quicksum(x[i, j] for i in D) <= u[j] for j in N), name="facility_capacity")
    # [3] all demand fully served
    model.addConstrs((gp.quicksum(x[i, j] for j in N) == demand[i] for i in D), name="demand_balance")
    # [4] weighted-average travel-time threshold
    model.addConstr(
        gp.quicksum(c[(i, j)] * x[i, j] for i in D for j in N) <= T * total_demand,
        name="service_threshold",
    )

    return model, {"y": y, "u": u, "x": x}
