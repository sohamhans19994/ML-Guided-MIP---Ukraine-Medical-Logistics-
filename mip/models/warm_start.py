from __future__ import annotations

import gurobipy as gp


def apply_warm_start(model: gp.Model, variables: dict, hint: dict) -> None:
    """Inject an ML-predicted warm start into a Gurobi model before solving.

    This is the Phase 2/3/4 integration point. The MIP model and its variables
    are fully built before this function is called; it only sets initial hint
    values that guide branch-and-bound without changing the formulation.

    Parameters
    ----------
    model     : a built but not yet solved gp.Model
    variables : variable dict returned by build_deterministic_model or build_robust_model
    hint      : dict produced by the ML predictor, expected keys:
                  ``y_hint``  — dict[node_id, float in [0,1]]  open-hub probability
                  ``u_hint``  — dict[node_id, float >= 0]       capacity suggestion
                  ``w_hint``  — dict[scenario_id, float in [0,1]] bad-scenario prob (robust only)

    Notes
    -----
    - Gurobi uses VarHintVal and VarHintPri attributes.  A hint of 1.0 for a
      binary variable strongly encourages the solver to try that branch first.
    - VarHintPri controls branching priority; higher = branch on this variable
      sooner.  We use the predicted open probability as the priority signal.
    - This function is a no-op stub until Phase 2.  Replace the body below with
      the actual hint injection once the ML predictor is available.
    """
    # --- Phase 2+ implementation goes here ---
    # y = variables["y"]
    # y_hint: dict = hint.get("y_hint", {})
    # for j, prob in y_hint.items():
    #     if j in y:
    #         y[j].VarHintVal = round(prob)          # 0 or 1 hint
    #         y[j].VarHintPri = int(prob * 100)      # branch priority
    #
    # u = variables["u"]
    # u_hint: dict = hint.get("u_hint", {})
    # for j, val in u_hint.items():
    #     if j in u:
    #         u[j].Start = float(val)
    #
    # w = variables.get("w", {})
    # w_hint: dict = hint.get("w_hint", {})
    # for sid, prob in w_hint.items():
    #     if sid in w:
    #         w[sid].VarHintVal = round(prob)
    pass
