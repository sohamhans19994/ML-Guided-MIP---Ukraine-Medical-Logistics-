"""Column-and-Constraint Generation (C&CG) scenario-enrichment loop.

This is the Phase 3 core (optimization half — no continual ML yet). We solve the
robust hub-location MIP against an effectively-infinite attack distribution by
growing a small *working set* of scenarios until a large held-out sample can no
longer break the design, yielding an out-of-sample robustness *certificate*.

The loop is **solver-agnostic**: the master solve and the scenario oracle are
injected as callables so this module stays free of any `ml/` dependency. The
GNN-accelerated master (Predict+Search / Alpha-Repair) and the real attack oracle
are wired up in `ml.ccg`.

Per iteration:
  1. ``solve_master(S)`` -> first-stage design (y, u).
  2. Evaluate the design on the fixed held-out set H -> out-of-sample violation
     rate (the certificate yardstick). Converge if rate <= delta - margin.
  3. Otherwise generate a fresh mining batch, find the scenarios the design
     violates, and add the worst ``add_per_round`` to S. Re-solve.

Two scenario sets are kept separate on purpose: H (held-out, never added to S) is
the honest certificate yardstick; the mining batches are what we "train" on by
adding their violators. This avoids measuring the certificate on scenarios we
enriched against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .data import MIPInstance
from .evaluate import evaluate_design_on_batch, violation_rate
from .scenarios import ScenarioData


@dataclass
class Design:
    """A first-stage design returned by a master-solve callable."""

    y_vals: dict[int, int]            # hub_id -> {0,1}
    u_vals: dict[int, float]          # hub_id -> capacity
    obj: float                        # build cost (objective)
    feasible: bool
    status: str                       # "optimal" | "time_limit" | "infeasible" | ...
    solve_time: float
    n_open: int = 0


@dataclass
class CCGIterationLog:
    iteration: int
    working_set_size: int
    design_cost: float
    master_status: str
    master_time: float
    in_sample_violations: int         # scenarios in S the design violates (<= delta*|S|)
    holdout_viol_rate: float
    n_violators_added: int


@dataclass
class CCGResult:
    converged: bool
    iterations: int
    reason: str                       # "converged" | "max_iterations" | "wall_clock" | "dry" | "master_failed"
    final_design: Design | None
    final_holdout_rate: float
    holdout_size: int
    delta: float
    margin: float
    logs: list[CCGIterationLog] = field(default_factory=list)


def iterative_enrichment_loop(
    instance: MIPInstance,
    initial_scenarios: list[ScenarioData],
    solve_master: Callable[[list[ScenarioData]], Design],
    gen_mining_batch: Callable[[int], list[ScenarioData]],
    holdout_scenarios: list[ScenarioData],
    *,
    delta: float = 0.10,
    margin: float = 0.02,
    add_per_round: int = 5,
    max_iterations: int = 25,
    wall_clock_s: float | None = None,
    max_dry_rounds: int = 3,
    log_fn: Callable[[str], None] = print,
) -> CCGResult:
    """Run the C&CG enrichment loop.

    Parameters
    ----------
    instance          : MIPInstance
    initial_scenarios : S0 — the (persisted) starting working set
    solve_master      : callable(scenarios) -> Design   (e.g. gurobi-exact or alpha-repair)
    gen_mining_batch  : callable(iteration) -> list[ScenarioData]  (fresh violator-mining batch)
    holdout_scenarios : H — fixed certificate set, never added to S
    delta             : chance-constraint violation budget (out-of-sample target)
    margin            : converge when holdout rate <= delta - margin (guards sampling noise)
    add_per_round     : how many worst violators to add to S each iteration
    max_iterations    : hard cap on outer iterations
    wall_clock_s      : optional total wall-clock backstop (seconds)
    max_dry_rounds    : stop if this many consecutive mining batches surface no violator
                        while the holdout still violates (mining can't find what to add)
    log_fn            : progress sink

    Returns
    -------
    CCGResult
    """
    S = list(initial_scenarios)
    logs: list[CCGIterationLog] = []
    t_start = time.perf_counter()

    last_design: Design | None = None
    last_rate = float("nan")
    dry_rounds = 0
    converged = False
    reason = "max_iterations"

    for it in range(max_iterations):
        # --- 1. solve master over the current working set ---
        design = solve_master(S)
        last_design = design

        if not design.feasible:
            # Structurally shouldn't happen (see mip.evaluate / master feasibility),
            # but guard so a surprise fails loud rather than silently.
            log_fn(f"[iter {it}] master returned no feasible design (status={design.status}) — stopping")
            logs.append(CCGIterationLog(
                iteration=it, working_set_size=len(S), design_cost=float("inf"),
                master_status=design.status, master_time=design.solve_time,
                in_sample_violations=-1, holdout_viol_rate=float("nan"), n_violators_added=0,
            ))
            reason = "master_failed"
            break

        # --- 2. certificate test on the fixed held-out set ---
        holdout_results = evaluate_design_on_batch(
            instance, holdout_scenarios, design.y_vals, design.u_vals
        )
        rate_H = violation_rate(holdout_results)
        last_rate = rate_H

        in_sample = evaluate_design_on_batch(instance, S, design.y_vals, design.u_vals)
        in_sample_viol = sum(1 for r in in_sample if r.violated)

        log_fn(
            f"[iter {it}] |S|={len(S)}  cost={design.obj:.2f}  open={design.n_open}  "
            f"master={design.status} ({design.solve_time:.1f}s)  "
            f"in-sample viol={in_sample_viol}/{len(S)}  holdout viol={rate_H:.1%}"
        )

        if rate_H <= delta - margin:
            converged = True
            reason = "converged"
            logs.append(CCGIterationLog(
                iteration=it, working_set_size=len(S), design_cost=design.obj,
                master_status=design.status, master_time=design.solve_time,
                in_sample_violations=in_sample_viol, holdout_viol_rate=rate_H,
                n_violators_added=0,
            ))
            break

        # --- backstop: wall-clock ---
        if wall_clock_s is not None and (time.perf_counter() - t_start) >= wall_clock_s:
            logs.append(CCGIterationLog(
                iteration=it, working_set_size=len(S), design_cost=design.obj,
                master_status=design.status, master_time=design.solve_time,
                in_sample_violations=in_sample_viol, holdout_viol_rate=rate_H,
                n_violators_added=0,
            ))
            reason = "wall_clock"
            break

        # --- 3. mine fresh violators and add the worst ---
        mining_batch = gen_mining_batch(it)
        mining_results = evaluate_design_on_batch(
            instance, mining_batch, design.y_vals, design.u_vals
        )
        violators = [
            (r, m) for r, m in zip(mining_results, mining_batch) if r.violated
        ]
        violators.sort(key=lambda rm: rm[0].severity, reverse=True)
        to_add = [m for (_, m) in violators[:add_per_round]]

        logs.append(CCGIterationLog(
            iteration=it, working_set_size=len(S), design_cost=design.obj,
            master_status=design.status, master_time=design.solve_time,
            in_sample_violations=in_sample_viol, holdout_viol_rate=rate_H,
            n_violators_added=len(to_add),
        ))

        if not to_add:
            dry_rounds += 1
            log_fn(
                f"[iter {it}] mining batch ({len(mining_batch)}) found no violator "
                f"despite holdout {rate_H:.1%} > delta — dry round {dry_rounds}/{max_dry_rounds}"
            )
            if dry_rounds >= max_dry_rounds:
                reason = "dry"
                break
            continue

        dry_rounds = 0
        S.extend(to_add)

    elapsed = time.perf_counter() - t_start
    log_fn(
        f"C&CG finished: reason={reason}  converged={converged}  "
        f"iterations={len(logs)}  final |S|={len(S)}  "
        f"final holdout viol={last_rate:.1%}  wall={elapsed:.1f}s"
    )

    return CCGResult(
        converged=converged,
        iterations=len(logs),
        reason=reason,
        final_design=last_design,
        final_holdout_rate=last_rate,
        holdout_size=len(holdout_scenarios),
        delta=delta,
        margin=margin,
        logs=logs,
    )
