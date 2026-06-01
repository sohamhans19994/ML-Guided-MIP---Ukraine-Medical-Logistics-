"""
benchmark_solve_time.py
-----------------------
Benchmarks pure Gurobi solve time at |S| = 10, 20, 30 scenarios.
No ML involved. Answers the question: at what |S| does the MIP get
hard enough to make Predict+Search improvement clearly visible?

Run from the project root:
    uv run python benchmark_solve_time.py

Optional args:
    --trials N      solves per scenario count (default 3)
    --delta D       violation fraction (default 0.10)
    --time-limit T  seconds per solve before Gurobi cuts off (default 300)
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from attack_scenarios.config import load_scenario_parameters
from attack_scenarios.geometry import load_attack_geography
from attack_scenarios.io import load_base_bundle
from mip.data import load_instance
from mip.models.robust import build_robust_model
from mip.scenarios import DEFAULT_THRESHOLD_BY_K
from ml.training import _generate_scenario_batch, _sample_K_sequence

SCENARIO_COUNTS = [10, 20, 30]
BASE_SEED = 42_000      # well away from training seeds (which start at 0)


def benchmark(n_trials: int, delta: float, time_limit: int) -> None:
    print("Loading instance and attack oracle ...")
    instance        = load_instance()
    params_template = load_scenario_parameters()
    base_bundle     = load_base_bundle(params_template.bundle_path)
    geography       = load_attack_geography(base_bundle["config"])

    print(f"  nodes={len(instance.N)}  demand_nodes={len(instance.D)}")
    print(f"  delta={delta}  time_limit={time_limit}s  trials_per_S={n_trials}")
    print()

    summary = {}

    for S in SCENARIO_COUNTS:
        allowed = int(S * delta)
        print(f"{'=' * 55}")
        print(f"|S| = {S}   violations allowed = {allowed}   ({delta:.0%} of {S})")
        print(f"{'=' * 55}")

        times, objs, nodes_list, gaps = [], [], [], []

        for trial in range(n_trials):
            seed = BASE_SEED + trial * 1000 + S
            rng  = np.random.default_rng(seed)
            K_seq = _sample_K_sequence(S, rng)

            scenarios = _generate_scenario_batch(
                instance=instance,
                base_bundle=base_bundle,
                geography=geography,
                params_template=params_template,
                K_sequence=K_seq,
                attack_mode="combo",
                trial_id=trial,
                base_seed=seed,
                threshold_by_k=DEFAULT_THRESHOLD_BY_K,
            )

            model, _ = build_robust_model(
                instance, scenarios, delta=delta, verbose=False
            )
            model.Params.TimeLimit = time_limit

            t0 = time.perf_counter()
            model.optimize()
            elapsed = time.perf_counter() - t0

            sol_count = model.SolCount
            obj  = model.ObjVal   if sol_count > 0 else float("nan")
            gap  = model.MIPGap   if sol_count > 0 else float("nan")
            n_bb = int(model.NodeCount)

            status_map = {
                2: "OPTIMAL", 3: "INFEASIBLE", 5: "UNBOUNDED",
                9: "TIME_LIMIT", 11: "INTERRUPTED", 13: "SUBOPTIMAL",
            }
            status = status_map.get(model.Status, f"STATUS_{model.Status}")

            times.append(elapsed)
            objs.append(obj)
            nodes_list.append(n_bb)
            gaps.append(gap)

            print(
                f"  trial {trial + 1}/{n_trials}: "
                f"time={elapsed:6.1f}s  "
                f"obj={obj:8.2f}  "
                f"nodes={n_bb:>7,}  "
                f"gap={gap:6.2%}  "
                f"[{status}]"
            )

        avg_t   = float(np.nanmean(times))
        avg_g   = float(np.nanmean(gaps))
        avg_n   = int(np.nanmean(nodes_list))
        print(
            f"  ─── MEAN: time={avg_t:.1f}s  nodes={avg_n:,}  gap={avg_g:.2%}\n"
        )
        summary[S] = dict(avg_time=avg_t, avg_gap=avg_g, avg_nodes=avg_n)

    print("SUMMARY")
    print(f"{'|S|':>5}  {'avg time':>10}  {'avg nodes':>12}  {'avg gap':>9}")
    print("─" * 42)
    for S, r in summary.items():
        print(
            f"{S:>5}  {r['avg_time']:>9.1f}s  "
            f"{r['avg_nodes']:>12,}  "
            f"{r['avg_gap']:>8.2%}"
        )
    print()
    print("Pick the |S| where avg time is in the 30s–5min range.")
    print("That is the target scenario count for training data generation.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials",     type=int,   default=3)
    p.add_argument("--delta",      type=float, default=0.10)
    p.add_argument("--time-limit", type=int,   default=300)
    args = p.parse_args()
    benchmark(n_trials=args.trials, delta=args.delta, time_limit=args.time_limit)
