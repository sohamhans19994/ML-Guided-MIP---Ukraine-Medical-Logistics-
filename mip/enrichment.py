from __future__ import annotations

from .data import MIPInstance
from .scenarios import ScenarioData
from .solution import RobustResult


def iterative_enrichment_loop(
    instance: MIPInstance,
    initial_scenarios: list[ScenarioData],
    oracle,
    delta: float = 0.05,
    max_iterations: int = 20,
    ml_guide=None,
) -> RobustResult:
    """Iteratively grow the scenario set S until no new violations are found.

    This is the Phase 3/4 core loop.  At each iteration:
      1. Solve the robust MIP on the current S.
      2. Query the oracle to generate new attack scenarios targeting the
         current design (y*, u*).
      3. Identify scenarios where constraint [4] is violated under (y*, u*)
         with optimal second-stage routing.
      4. If ``ml_guide`` is provided, use it to rank the new scenarios and
         add only the top-ranked ones (scenario prioritisation).
         Otherwise add all violating scenarios.
      5. Re-solve.  Stop when no new violations or budget exhausted.

    Parameters
    ----------
    instance          : MIPInstance
    initial_scenarios : S₀ — starting scenario set
    oracle            : callable(instance, current_design, budget_k) → list[ScenarioData]
    delta             : violation fraction budget
    max_iterations    : hard cap on enrichment iterations
    ml_guide          : optional ML guide object with .rank_scenarios(scenarios) method
                        (Phase 3/4); pass None for pure scenario enrichment (Phase 1 extension)

    Returns
    -------
    RobustResult from the final solve on the converged scenario set S.

    Notes
    -----
    This function is a stub.  The full implementation requires:
    - An oracle interface (LLM-based or heuristic).
    - A second-stage feasibility checker to identify violated scenarios.
    - An ML guide interface for scenario prioritisation (Phase 3/4).
    """
    raise NotImplementedError(
        "iterative_enrichment_loop is not yet implemented. "
        "This will be the core of Phase 3 / Phase 4. "
        "For now, use build_robust_model() directly with a fixed scenario set."
    )
