"""C&CG scenario-enrichment experiment harness (Phase 3 — optimization half).

Wires the solver-agnostic loop in ``mip.enrichment`` to the real attack oracle,
the GNN, and the two master-solve drivers:

  - ``gurobi-exact`` : plain Gurobi master (ground-truth control)
  - ``alpha-repair`` : Predict+Search with alpha-repair (production driver, static GNN)

No continual learning yet — the GNN is used statically (re-predict each iteration,
no weight updates).

Reproducibility (see data/phase3_plan.md):
  - The initial working set S0 and the held-out certificate set H are each
    generated once and **persisted to data/ccg/**; every driver loads the identical
    saved files so runs differ only by solver.
  - All scenario seeds derive deterministically from a single ``--seed``.

Usage
-----
    uv run python -m ml.ccg --driver gurobi-exact --delta 0.10 \
        --holdout-size 1000 --mining-size 200 --add-per-round 5 \
        --max-iters 25 --per-master-limit 1200 --seed 500000

    uv run python -m ml.ccg --driver alpha-repair --model models/hub_gnn_bce.pt \
        --alpha 0.75 [same flags]
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from gurobipy import GRB

from attack_scenarios.config import load_scenario_parameters
from attack_scenarios.geometry import load_attack_geography
from attack_scenarios.io import load_base_bundle
from mip.data import load_instance
from mip.enrichment import Design, iterative_enrichment_loop
from mip.models.robust import build_robust_model
from mip.scenarios import DEFAULT_THRESHOLD_BY_K, ScenarioData
from mip.solution import extract_robust_solution
from ml.dataset import build_hub_features
from ml.predict_search import load_model, predict_probs, solve_with_alpha_repair
from ml.training import _generate_scenario_batch, _sample_K_sequence

# Distinct seed streams so init / held-out / mining scenarios never coincide.
SEED_INIT_OFFSET    = 0
SEED_HOLDOUT_OFFSET = 1_000_000
SEED_MINING_OFFSET  = 2_000_000
SEED_MINING_STEP    = 10_000

CCG_DATA_DIR = Path("data/ccg")


# ---------------------------------------------------------------------------
# Scenario generation + persistence
# ---------------------------------------------------------------------------

def _generate_set(
    instance, base_bundle, geography, params_template,
    attack_mode: str, size: int, base_seed: int, trial_id: int,
) -> list[ScenarioData]:
    """Generate `size` fresh scenarios from the oracle with a fixed seed stream."""
    rng = np.random.default_rng(base_seed)
    K_seq = _sample_K_sequence(size, rng)
    return _generate_scenario_batch(
        instance=instance,
        base_bundle=base_bundle,
        geography=geography,
        params_template=params_template,
        K_sequence=K_seq,
        attack_mode=attack_mode,
        trial_id=trial_id,
        base_seed=base_seed,
        threshold_by_k=DEFAULT_THRESHOLD_BY_K,
    )


def _load_or_generate(
    path: Path, gen_fn, log_fn=print,
) -> list[ScenarioData]:
    """Load a persisted scenario set if present, else generate, save, and return it."""
    if path.exists():
        log_fn(f"  loading persisted scenarios: {path}")
        with path.open("rb") as f:
            return pickle.load(f)
    log_fn(f"  generating + saving scenarios: {path}")
    scenarios = gen_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(scenarios, f)
    return scenarios


# ---------------------------------------------------------------------------
# Master-solve drivers (each returns a Design)
# ---------------------------------------------------------------------------

def _status_str(status: int) -> str:
    return {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbd",
    }.get(status, f"status_{status}")


def make_gurobi_exact_driver(instance, delta: float, per_master_limit: int):
    """Plain Gurobi master solve — ground-truth control."""
    def solve(scenarios: list[ScenarioData]) -> Design:
        model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
        model.Params.TimeLimit = per_master_limit
        t0 = time.perf_counter()
        model.optimize()
        st = time.perf_counter() - t0

        if model.SolCount == 0:
            return Design({}, {}, float("inf"), False, _status_str(model.Status), st, 0)

        res = extract_robust_solution(model, variables, instance, scenarios, delta)
        y_vals = {j: int(j in set(res.open_hubs)) for j in instance.N}
        u_vals = {j: float(res.hub_capacity.get(j, 0.0)) for j in instance.N}
        return Design(
            y_vals=y_vals, u_vals=u_vals, obj=float(res.obj_val), feasible=True,
            status=_status_str(model.Status), solve_time=st, n_open=len(res.open_hubs),
        )
    return solve


def make_alpha_repair_driver(
    instance, delta: float, per_master_limit: int,
    model_gnn, hub_feats, hubs, hub_idx, device,
    alpha: float, repair_step: float,
):
    """Predict+Search with alpha-repair — production driver (static GNN)."""
    def solve(scenarios: list[ScenarioData]) -> Design:
        # Re-predict every iteration: the scenario graph changes as S grows.
        probs = predict_probs(model_gnn, instance, scenarios, hub_feats, hubs, hub_idx, device)
        r = solve_with_alpha_repair(
            instance, scenarios, probs, hubs, hub_idx,
            alpha=alpha, delta=delta, time_limit=per_master_limit,
            snap_times=[], repair_step=repair_step,
        )
        if not r.feasible or r.y_vals is None or r.u_vals is None:
            return Design({}, {}, float("inf"), False, "infeasible", r.time_s, 0)
        y_vals = dict(r.y_vals)
        u_vals = dict(r.u_vals)
        n_open = int(sum(y_vals.values()))
        status = "optimal" if (r.gap is not None and r.gap <= 1e-6) else "feasible"
        return Design(
            y_vals=y_vals, u_vals=u_vals, obj=float(r.obj), feasible=True,
            status=status, solve_time=r.time_s, n_open=n_open,
        )
    return solve


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_ccg_experiment(
    *,
    driver: str,
    model_path: str,
    delta: float,
    margin: float,
    init_size: int,
    holdout_size: int,
    mining_size: int,
    add_per_round: int,
    max_iters: int,
    per_master_limit: int,
    wall_clock: float | None,
    attack_mode: str,
    seed: int,
    alpha: float,
    repair_step: float,
    out_path: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading instance, oracle, and (if needed) model ...")
    instance        = load_instance()
    params_template = load_scenario_parameters()
    base_bundle     = load_base_bundle(params_template.bundle_path)
    geography       = load_attack_geography(base_bundle["config"])

    # --- persisted, identical-across-drivers scenario sets ---
    init_seed    = seed + SEED_INIT_OFFSET
    holdout_seed = seed + SEED_HOLDOUT_OFFSET

    init_path    = CCG_DATA_DIR / f"init_scenarios_seed{seed}_n{init_size}.pkl"
    holdout_path = CCG_DATA_DIR / f"holdout_seed{seed}_n{holdout_size}.pkl"

    print(f"Initial working set S0 (|S0|={init_size}):")
    initial_scenarios = _load_or_generate(
        init_path,
        lambda: _generate_set(instance, base_bundle, geography, params_template,
                              attack_mode, init_size, init_seed, trial_id=0),
    )
    print(f"Held-out certificate set H (|H|={holdout_size}):")
    holdout_scenarios = _load_or_generate(
        holdout_path,
        lambda: _generate_set(instance, base_bundle, geography, params_template,
                              attack_mode, holdout_size, holdout_seed, trial_id=1),
    )

    # --- mining batch generator (fresh per iteration, fixed seed stream) ---
    def gen_mining_batch(iteration: int) -> list[ScenarioData]:
        mining_seed = seed + SEED_MINING_OFFSET + iteration * SEED_MINING_STEP
        return _generate_set(
            instance, base_bundle, geography, params_template,
            attack_mode, mining_size, mining_seed, trial_id=100 + iteration,
        )

    # --- driver ---
    if driver == "gurobi-exact":
        solve_master = make_gurobi_exact_driver(instance, delta, per_master_limit)
    elif driver == "alpha-repair":
        model_gnn, ckpt = load_model(model_path, device)
        hub_feats, hubs, hub_idx = build_hub_features(instance)
        hub_feats = hub_feats.to(device)
        print(f"  model: epoch={ckpt.get('epoch')}  val_acc={ckpt.get('val_acc')}")
        solve_master = make_alpha_repair_driver(
            instance, delta, per_master_limit,
            model_gnn, hub_feats, hubs, hub_idx, device, alpha, repair_step,
        )
    else:
        raise ValueError(f"unknown driver: {driver}")

    print("\n" + "=" * 70)
    print(f"C&CG  driver={driver}  delta={delta}  margin={margin}  "
          f"|S0|={init_size}  |H|={holdout_size}  mining={mining_size}  "
          f"add/round={add_per_round}  per_master_limit={per_master_limit}s")
    print("=" * 70 + "\n")

    result = iterative_enrichment_loop(
        instance, initial_scenarios,
        solve_master=solve_master,
        gen_mining_batch=gen_mining_batch,
        holdout_scenarios=holdout_scenarios,
        delta=delta, margin=margin, add_per_round=add_per_round,
        max_iterations=max_iters, wall_clock_s=wall_clock, log_fn=print,
    )

    # --- summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'iter':>4}  {'|S|':>4}  {'cost':>9}  {'master':>11}  "
          f"{'time':>8}  {'in-viol':>9}  {'holdout':>8}  {'+added':>6}")
    print(f"  {'-' * 72}")
    for lg in result.logs:
        cost_str = "inf" if lg.design_cost == float("inf") else f"{lg.design_cost:.2f}"
        inv_str = "n/a" if lg.in_sample_violations < 0 else f"{lg.in_sample_violations}/{lg.working_set_size}"
        print(f"  {lg.iteration:>4}  {lg.working_set_size:>4}  {cost_str:>9}  "
              f"{lg.master_status:>11}  {lg.master_time:>7.1f}s  {inv_str:>9}  "
              f"{lg.holdout_viol_rate:>7.1%}  {lg.n_violators_added:>6}")

    print()
    print(f"  driver={driver}  converged={result.converged}  reason={result.reason}  "
          f"iterations={result.iterations}")
    final_size = result.logs[-1].working_set_size if result.logs else init_size
    print(f"  final |S|={final_size}  "
          f"final holdout violation={result.final_holdout_rate:.1%}  "
          f"(target ≤ {delta - margin:.1%})")
    if result.final_design is not None and result.final_design.feasible:
        print(f"  final design: cost={result.final_design.obj:.2f}  "
              f"open hubs={result.final_design.n_open}")

    # --- optional dump ---
    if out_path:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("wb") as f:
            pickle.dump(result, f)
        print(f"\n  results written to {op}")


def main() -> None:
    p = argparse.ArgumentParser(description="C&CG scenario-enrichment loop")
    p.add_argument("--driver", choices=["gurobi-exact", "alpha-repair"], default="gurobi-exact")
    p.add_argument("--model", default="models/hub_gnn_bce.pt")
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--init-size", type=int, default=20)
    p.add_argument("--holdout-size", type=int, default=1000)
    p.add_argument("--mining-size", type=int, default=200)
    p.add_argument("--add-per-round", type=int, default=5)
    p.add_argument("--max-iters", type=int, default=25)
    p.add_argument("--per-master-limit", type=int, default=1200)
    p.add_argument("--wall-clock", type=float, default=None)
    p.add_argument("--attack-mode", default="combo")
    p.add_argument("--seed", type=int, default=500_000)
    p.add_argument("--alpha", type=float, default=0.75, help="initial alpha for alpha-repair driver")
    p.add_argument("--repair-step", type=float, default=0.10)
    p.add_argument("--out", default=None, help="optional path to pickle the CCGResult")
    args = p.parse_args()

    run_ccg_experiment(
        driver=args.driver,
        model_path=args.model,
        delta=args.delta,
        margin=args.margin,
        init_size=args.init_size,
        holdout_size=args.holdout_size,
        mining_size=args.mining_size,
        add_per_round=args.add_per_round,
        max_iters=args.max_iters,
        per_master_limit=args.per_master_limit,
        wall_clock=args.wall_clock,
        attack_mode=args.attack_mode,
        seed=args.seed,
        alpha=args.alpha,
        repair_step=args.repair_step,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
