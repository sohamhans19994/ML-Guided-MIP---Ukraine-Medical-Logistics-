"""Training data generation for the GNN hub-open predictor.

Pipeline
--------
For each trial:
  1. Sample 10 K values: guarantee one per K ∈ {2,3,4,5}, then sample 6 more
     from a center-weighted distribution so harder attacks appear more often.
  2. Generate each scenario in memory (no disk I/O), each with a unique seed.
  3. Solve the robust MIP (10 scenarios, delta=0.10 → 1 violation allowed).
  4. Collect near-optimal hub configs from Gurobi's solution pool.

Usage
-----
    from ml.training import generate_training_data
    records = generate_training_data(n_trials=100)

Or from the command line:
    uv run python -m ml.training --n-trials 100
"""
from __future__ import annotations

import argparse
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx
import numpy as np

from attack_scenarios.config import load_scenario_parameters
from attack_scenarios.geometry import load_attack_geography
from attack_scenarios.io import load_base_bundle
from attack_scenarios.model import generate_attack_bundle
from mip.data import MIPInstance, load_instance
from mip.models.robust import build_robust_model
from mip.scenarios import DEFAULT_THRESHOLD_BY_K, ScenarioData

DEFAULT_OUTPUT_PATH = Path("data/ml_training/records.pkl")
K_OPTIONS: list[float] = [2.0, 3.0, 4.0, 5.0]

# Center-weighted: medium attacks (K=3,4) are more common than extremes.
# Used when sampling the 6 "extra" scenarios beyond the 4 guaranteed ones.
K_SAMPLE_WEIGHTS = np.array([0.15, 0.35, 0.35, 0.15])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PoolSolution:
    """One near-optimal solution from Gurobi's solution pool."""
    pool_rank: int           # 0 = optimal, 1..N = progressively worse
    obj_val: float           # objective value for this pool solution
    y_open: dict[int, int]   # node_id → 0 or 1 (rounded)
    y_raw: dict[int, float]  # node_id → raw Xn (soft label / variable bias)


@dataclass
class TrainingRecord:
    """Everything needed to build one training example for the GNN."""
    trial_id: int
    delta: float
    scenarios: list[ScenarioData]   # 10 scenarios used in this solve
    pool_solutions: list[PoolSolution]
    solve_time_s: float
    status: int                     # Gurobi model status code


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_K_sequence(n_scenarios: int, rng: np.random.Generator) -> list[float]:
    """Sample a list of K values of length n_scenarios.

    Guarantees each K in {2,3,4,5} appears at least once, then fills the
    remainder by sampling from K_SAMPLE_WEIGHTS. The result is shuffled so
    the guaranteed scenarios aren't always first.

    With n_scenarios=10: 4 guaranteed + 6 sampled → balanced but random mix.
    """
    guaranteed = K_OPTIONS.copy()                          # one of each K
    n_extra = max(0, n_scenarios - len(K_OPTIONS))
    extra = rng.choice(K_OPTIONS, size=n_extra, p=K_SAMPLE_WEIGHTS).tolist()
    sequence = guaranteed + extra
    rng.shuffle(sequence)
    return sequence


def _make_scenario_data(
    bundle: dict,
    instance: MIPInstance,
    K: float,
    scenario_id: str,
    threshold_by_k: dict[float, float],
) -> ScenarioData:
    """Convert an in-memory attack bundle to a ScenarioData object.

    Mirrors the logic in mip.scenarios.load_scenario_batch but works directly
    from the bundle dict so no disk round-trip is needed.
    """
    Gs: nx.Graph = bundle["graphs"]["scenario_graph"]
    surviving_nodes = sorted(Gs.nodes())
    T = float(threshold_by_k.get(K, max(threshold_by_k.values())))

    node_pos = {
        n: (float(instance.CG.nodes[n]["lon"]), float(instance.CG.nodes[n]["lat"]))
        for n in instance.N
    }
    surviving_pos = {n: node_pos[n] for n in surviving_nodes}

    proxy_demand_node: dict[int, int] = {}
    c_s: dict[tuple[int, int], float] = {}

    for i in instance.D:
        if i in Gs.nodes:
            proxy = i
        else:
            lon_i, lat_i = node_pos[i]
            proxy = min(
                surviving_nodes,
                key=lambda n, lx=lon_i, ly=lat_i: (
                    (surviving_pos[n][0] - lx) ** 2 + (surviving_pos[n][1] - ly) ** 2
                ),
            )
        proxy_demand_node[i] = proxy
        lengths = nx.single_source_dijkstra_path_length(Gs, proxy, weight="travel_time")
        for j in surviving_nodes:
            if j in lengths and np.isfinite(lengths[j]):
                c_s[(i, j)] = float(lengths[j] / 3600.0)

    return ScenarioData(
        id=scenario_id,
        K=K,
        T=T,
        graph=Gs,
        surviving_nodes=surviving_nodes,
        proxy_demand_node=proxy_demand_node,
        c=c_s,
        summary=bundle.get("summary", {}),
    )


def _generate_scenario_batch(
    instance: MIPInstance,
    base_bundle: dict,
    geography: dict,
    params_template,
    K_sequence: list[float],
    attack_mode: str,
    trial_id: int,
    base_seed: int,
    threshold_by_k: dict[float, float],
) -> list[ScenarioData]:
    """Generate one ScenarioData per entry in K_sequence.

    Each scenario gets its own unique seed so same-K scenarios within a trial
    produce different attack placements.
    """
    scenarios: list[ScenarioData] = []

    for idx, K in enumerate(K_sequence):
        scenario_seed = base_seed + idx
        scenario_id = f"trial{trial_id:04d}_{idx:02d}_K{K:.0f}_s{scenario_seed}"

        new_budget = replace(params_template.budget, random_seed=scenario_seed)
        params = replace(
            params_template,
            base_budget=K,
            attack_mode=attack_mode,
            scenario_id=scenario_id,
            save_outputs=False,
            generate_visual=False,
            budget=new_budget,
        )

        bundle = generate_attack_bundle(
            base_graph=base_bundle["graphs"]["adaptive_graph"],
            demand_nodes=base_bundle.get("demand_nodes"),
            params=params,
            geography=geography,
        )

        sd = _make_scenario_data(bundle, instance, K, scenario_id, threshold_by_k)
        scenarios.append(sd)

    return scenarios


def _solve_with_pool(
    instance: MIPInstance,
    scenarios: list[ScenarioData],
    delta: float,
    n_pool: int,
    pool_gap: float,
    n_threads: int = 0,
) -> tuple[list[PoolSolution], float, int]:
    """Solve the robust MIP and collect near-optimal solutions from the pool.

    Returns
    -------
    pool_solutions : list of PoolSolution (empty if infeasible)
    solve_time_s   : wall-clock time for model.optimize()
    status         : Gurobi model status code
    """
    model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = n_pool
    model.Params.PoolGap = pool_gap
    if n_threads > 0:
        model.Params.Threads = n_threads

    t0 = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - t0

    pool_solutions: list[PoolSolution] = []
    if model.SolCount == 0:
        return pool_solutions, solve_time, model.Status

    y = variables["y"]
    N = instance.N

    for k in range(model.SolCount):
        model.Params.SolutionNumber = k
        pool_solutions.append(PoolSolution(
            pool_rank=k,
            obj_val=float(model.PoolObjVal),
            y_open={j: int(round(float(y[j].Xn))) for j in N},
            y_raw={j: float(y[j].Xn) for j in N},
        ))

    model.Params.SolutionNumber = 0
    return pool_solutions, solve_time, model.Status


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _run_trial_batch(
    trial_ids: list[int],
    delta: float,
    n_scenarios: int,
    attack_mode: str,
    n_pool: int,
    pool_gap: float,
    base_seed: int,
    n_threads: int,
) -> list[TrainingRecord]:
    """Process a batch of trials in a single worker process.

    Assets (instance, bundle, geography) are loaded once per worker, not once
    per trial, so the overhead is amortised across the batch.
    """
    instance = load_instance()
    params_template = load_scenario_parameters()
    base_bundle = load_base_bundle(params_template.bundle_path)
    geography = load_attack_geography(base_bundle["config"])
    threshold_by_k = DEFAULT_THRESHOLD_BY_K

    records: list[TrainingRecord] = []
    for trial_id in trial_ids:
        trial_seed = base_seed + trial_id * n_scenarios
        rng = np.random.default_rng(trial_seed)
        K_sequence = _sample_K_sequence(n_scenarios, rng)

        scenarios = _generate_scenario_batch(
            instance=instance,
            base_bundle=base_bundle,
            geography=geography,
            params_template=params_template,
            K_sequence=K_sequence,
            attack_mode=attack_mode,
            trial_id=trial_id,
            base_seed=trial_seed,
            threshold_by_k=threshold_by_k,
        )
        pool_solutions, solve_time, status = _solve_with_pool(
            instance=instance,
            scenarios=scenarios,
            delta=delta,
            n_pool=n_pool,
            pool_gap=pool_gap,
            n_threads=n_threads,
        )
        records.append(TrainingRecord(
            trial_id=trial_id,
            delta=delta,
            scenarios=scenarios,
            pool_solutions=pool_solutions,
            solve_time_s=solve_time,
            status=status,
        ))

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_training_data(
    n_trials: int = 100,
    output_path: str | Path | None = None,
    n_scenarios: int = 10,
    delta: float = 0.10,
    attack_mode: str = "combo",
    n_pool: int = 10,
    pool_gap: float = 0.05,
    base_seed: int = 0,
    n_workers: int = 1,
    verbose: bool = True,
) -> list[TrainingRecord]:
    """Generate training records for the GNN hub-open predictor.

    For each trial: sample 10 K values (guaranteed coverage of {2,3,4,5},
    remainder sampled from a center-weighted distribution), generate each
    scenario in memory, solve the robust MIP, and record all near-optimal
    hub configurations from Gurobi's solution pool.

    With n_scenarios=10 and delta=0.10: violation budget = 10 * 0.10 = 1,
    so exactly one scenario may be violated — a meaningful chance constraint.

    Parameters
    ----------
    n_trials    : number of independent solve trials
    output_path : pickle output path (default: data/ml_training/records.pkl)
    n_scenarios : scenarios per trial (default: 10)
    delta       : violation fraction (default: 0.10 → 1 violation per trial)
    attack_mode : attack type for all scenarios (default: "combo")
    n_pool      : max pool solutions per solve (Gurobi PoolSolutions)
    pool_gap    : relative gap for pool collection (Gurobi PoolGap)
    base_seed   : rng seed; trial i uses base_seed + i * n_scenarios
    n_workers   : parallel worker processes (default: 1 = sequential).
                  Each worker gets floor(cpu_count / n_workers) Gurobi threads.
                  Gurobi B&B parallelism saturates around 4-8 threads, so prefer
                  more workers with fewer threads: e.g. n_workers=8 on 32 cores
                  (4 threads/worker). Cap around 16 due to RAM (~1-2 GB/worker).
    verbose     : print progress

    Returns
    -------
    list of TrainingRecord
    """
    output_path = Path(output_path or DEFAULT_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_cpus = multiprocessing.cpu_count()
    n_workers = max(1, min(n_workers, n_trials))
    n_threads = max(1, total_cpus // n_workers)

    if verbose:
        allowed_violations = int(n_scenarios * delta)
        print(
            f"Config: {n_trials} trials  |  {n_scenarios} scenarios/trial  "
            f"|  delta={delta} ({allowed_violations} violation(s) allowed)  "
            f"|  pool={n_pool}  gap={pool_gap:.0%}  "
            f"|  workers={n_workers}  threads/worker={n_threads}"
        )

    # Split trials across workers as evenly as possible
    all_trial_ids = list(range(n_trials))
    batches = [all_trial_ids[i::n_workers] for i in range(n_workers)]

    worker_kwargs = dict(
        delta=delta,
        n_scenarios=n_scenarios,
        attack_mode=attack_mode,
        n_pool=n_pool,
        pool_gap=pool_gap,
        base_seed=base_seed,
        n_threads=n_threads,
    )

    records: list[TrainingRecord] = []

    if n_workers == 1:
        # Sequential — run in-process so verbose per-trial output works
        instance = load_instance()
        params_template = load_scenario_parameters()
        base_bundle = load_base_bundle(params_template.bundle_path)
        geography = load_attack_geography(base_bundle["config"])
        threshold_by_k = DEFAULT_THRESHOLD_BY_K

        if verbose:
            from mip.data import MIPInstance as _MI
            print(f"Instance: {len(instance.N)} nodes, {len(instance.D)} demand nodes\n")

        for trial_id in all_trial_ids:
            trial_seed = base_seed + trial_id * n_scenarios
            rng = np.random.default_rng(trial_seed)
            K_sequence = _sample_K_sequence(n_scenarios, rng)

            if verbose:
                from collections import Counter
                k_counts = dict(sorted(Counter(K_sequence).items()))
                print(f"[{trial_id+1:>3}/{n_trials}] seed={trial_seed}  K={k_counts}  ", end="", flush=True)

            scenarios = _generate_scenario_batch(
                instance=instance,
                base_bundle=base_bundle,
                geography=geography,
                params_template=params_template,
                K_sequence=K_sequence,
                attack_mode=attack_mode,
                trial_id=trial_id,
                base_seed=trial_seed,
                threshold_by_k=threshold_by_k,
            )
            pool_solutions, solve_time, status = _solve_with_pool(
                instance=instance,
                scenarios=scenarios,
                delta=delta,
                n_pool=n_pool,
                pool_gap=pool_gap,
                n_threads=n_threads,
            )
            records.append(TrainingRecord(
                trial_id=trial_id,
                delta=delta,
                scenarios=scenarios,
                pool_solutions=pool_solutions,
                solve_time_s=solve_time,
                status=status,
            ))
            if verbose:
                if pool_solutions:
                    print(
                        f"sols={len(pool_solutions)}  "
                        f"opt_obj={pool_solutions[0].obj_val:.1f}  "
                        f"open_hubs={sum(pool_solutions[0].y_open.values())}  "
                        f"t={solve_time:.1f}s"
                    )
                else:
                    print(f"INFEASIBLE (status={status})  t={solve_time:.1f}s")

    else:
        # Parallel — each worker handles a batch of trials
        if verbose:
            print(f"Dispatching {len(batches)} worker batches...\n")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_trial_batch, batch, **worker_kwargs): i
                for i, batch in enumerate(batches)
            }
            completed = 0
            for future in as_completed(futures):
                batch_records = future.result()   # raises if worker raised
                records.extend(batch_records)
                completed += len(batch_records)
                if verbose:
                    print(f"  {completed}/{n_trials} trials complete", flush=True)

        records.sort(key=lambda r: r.trial_id)

    with output_path.open("wb") as fh:
        pickle.dump(records, fh)

    if verbose:
        feasible = sum(1 for r in records if r.pool_solutions)
        avg_t = float(np.mean([r.solve_time_s for r in records]))
        print(f"\nDone. {feasible}/{n_trials} feasible  avg_solve={avg_t:.1f}s  →  {output_path}")

    return records


def load_training_records(path: str | Path = DEFAULT_OUTPUT_PATH) -> list[TrainingRecord]:
    """Load a previously generated training dataset from disk."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate ML training data for hub-open predictor.")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--n-scenarios", type=int, default=10)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--attack-mode", type=str, default="combo",
                   choices=["missile", "bomb", "combo"])
    p.add_argument("--n-pool", type=int, default=10)
    p.add_argument("--pool-gap", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1,
                   help="Parallel worker processes (default: 1 = sequential). "
                        "Each worker gets floor(cpu_count/n_workers) Gurobi threads. "
                        "Gurobi B&B parallelism saturates around 4-8 threads, so use "
                        "more workers with fewer threads each: e.g. --n-workers 8 on a "
                        "32-core machine (4 threads/worker). Cap at ~16 due to RAM "
                        "(each worker loads the full instance + Gurobi model ~1-2 GB).")
    p.add_argument("--quiet", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    generate_training_data(
        n_trials=args.n_trials,
        output_path=args.output,
        n_scenarios=args.n_scenarios,
        delta=args.delta,
        attack_mode=args.attack_mode,
        n_pool=args.n_pool,
        pool_gap=args.pool_gap,
        base_seed=args.seed,
        n_workers=args.n_workers,
        verbose=not args.quiet,
    )
