"""
ml/predict_search.py
--------------------
Experiment 1 / 2 / 3: Predict + Search evaluation.

For each fresh test instance the script:
  1. Generates a new scenario set (unseen during training).
  2. Runs pure Gurobi as the baseline.
  3. Runs the GNN, gets P(y_i=1) per hub.
  4. For each fixing fraction alpha, adds hard equality constraints
     for the top-alpha% most confident hub predictions, then re-solves.
  5. Prints a comparison table: time, objective, gap, nodes, feasibility.
  6. Records primal bound at user-specified time snapshots — useful for
     large |S| where Gurobi hits the time limit.

Usage
-----
    uv run python -m ml.predict_search
    uv run python -m ml.predict_search --model models/hub_gnn_bce.pt --n-test 10
    uv run python -m ml.predict_search --n-scenarios 100 --alphas 0.5 0.75 --time-limit 600
    uv run python -m ml.predict_search --n-scenarios 100 --snapshots 30 60 120 300 600

All test instances use seeds far from training seeds (base_seed=500000).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

from attack_scenarios.config import load_scenario_parameters
from attack_scenarios.geometry import load_attack_geography
from attack_scenarios.io import load_base_bundle
from mip.data import load_instance
from mip.models.robust import build_robust_model
from mip.scenarios import DEFAULT_THRESHOLD_BY_K
from ml.dataset import build_hub_features, build_instance_graph
from ml.model import HubGNN
from ml.training import _generate_scenario_batch, _sample_K_sequence

TEST_BASE_SEED = 500_000


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    label:      str
    time_s:     float
    obj:        float
    gap:        float
    nodes:      int
    feasible:   bool
    n_fixed:    int = 0
    snapshots:  list[tuple[float, float]] = field(default_factory=list)
    # snapshots: list of (snapshot_time_s, primal_bound) recorded during solve


# ---------------------------------------------------------------------------
# Primal-bound callback
# ---------------------------------------------------------------------------

def _make_callback(model: gp.Model, snap_times: list[float]):
    """
    Returns (callback_fn, records_list).

    callback_fn is passed to model.optimize(). It records the best primal
    bound each time a snapshot time is reached. records_list is mutated
    in-place and available after optimization completes.
    """
    records: list[tuple[float, float]] = []
    idx = [0]

    def cb(where: int) -> None:
        if where != GRB.Callback.MIP:
            return
        if idx[0] >= len(snap_times):
            return
        t   = model.cbGet(GRB.Callback.RUNTIME)
        obj = model.cbGet(GRB.Callback.MIP_OBJBST)
        if obj >= GRB.INFINITY:
            obj = float("nan")
        while idx[0] < len(snap_times) and t >= snap_times[idx[0]]:
            records.append((snap_times[idx[0]], obj))
            idx[0] += 1

    return cb, records


# ---------------------------------------------------------------------------
# Core solve helpers
# ---------------------------------------------------------------------------

def solve_baseline(
    instance,
    scenarios,
    delta:      float,
    time_limit: int,
    snap_times: list[float],
) -> SolveResult:
    """Pure Gurobi — no variable fixing."""
    model, _ = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    model.Params.TimeLimit = time_limit

    cb, records = _make_callback(model, snap_times)
    t0 = time.perf_counter()
    model.optimize(cb)
    elapsed = time.perf_counter() - t0

    feasible = model.SolCount > 0
    return SolveResult(
        label="Gurobi",
        time_s=elapsed,
        obj=model.ObjVal  if feasible else float("nan"),
        gap=model.MIPGap  if feasible else float("nan"),
        nodes=int(model.NodeCount),
        feasible=feasible,
        snapshots=records,
    )


def solve_predict_search(
    instance,
    scenarios,
    probs:      torch.Tensor,
    hubs:       list[int],
    alpha:      float,
    delta:      float,
    time_limit: int,
    snap_times: list[float],
) -> SolveResult:
    """
    Predict + Search: fix the top-alpha% most confident hub predictions,
    solve the reduced MIP.

    Confidence = |P(y_i=1) - 0.5| * 2  (0=uncertain, 1=certain).
    """
    n_hubs   = len(hubs)
    n_fix    = max(1, int(round(alpha * n_hubs)))
    probs_np = probs.detach().cpu().numpy()

    confidence  = np.abs(probs_np - 0.5) * 2.0
    fix_indices = np.argsort(confidence)[::-1][:n_fix]

    model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    model.Params.TimeLimit = time_limit
    y = variables["y"]

    for idx in fix_indices:
        hub_id  = hubs[idx]
        fix_val = 1 if probs_np[idx] >= 0.5 else 0
        model.addConstr(y[hub_id] == fix_val, name=f"fix_{hub_id}")
    model.update()

    cb, records = _make_callback(model, snap_times)
    t0 = time.perf_counter()
    model.optimize(cb)
    elapsed = time.perf_counter() - t0

    feasible = model.SolCount > 0
    return SolveResult(
        label=f"P+S a={alpha:.0%}",
        time_s=elapsed,
        obj=model.ObjVal  if feasible else float("nan"),
        gap=model.MIPGap  if feasible else float("nan"),
        nodes=int(model.NodeCount),
        feasible=feasible,
        n_fixed=n_fix,
        snapshots=records,
    )


# ---------------------------------------------------------------------------
# GNN inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[HubGNN, dict]:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args  = ckpt.get("args", {})
    model = HubGNN(
        hidden_dim=args.get("hidden_dim", 64),
        n_rounds=args.get("n_rounds",     2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_probs(
    model:     HubGNN,
    instance,
    scenarios,
    hub_feats: torch.Tensor,
    hubs:      list[int],
    hub_idx:   dict[int, int],
    device:    torch.device,
) -> torch.Tensor:
    graph = build_instance_graph(instance, scenarios, hub_feats, hubs, hub_idx)
    graph = graph.to(device)
    return model(graph).cpu()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _snapshot_str(snapshots: list[tuple[float, float]], solve_time: float) -> str:
    """Format snapshot records for display."""
    if not snapshots:
        return f"    (solved in {solve_time:.1f}s — before first snapshot)"
    parts = [f"@{int(t)}s={obj:.1f}" if not np.isnan(obj) else f"@{int(t)}s=n/a"
             for t, obj in snapshots]
    return "    Primal bound: " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model_path:   str,
    n_test:       int,
    n_scenarios:  int,
    alphas:       list[float],
    delta:        float,
    time_limit:   int,
    attack_mode:  str,
    seed:         int,
    snap_times:   list[float],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading instance and model ...")
    instance        = load_instance()
    params_template = load_scenario_parameters()
    base_bundle     = load_base_bundle(params_template.bundle_path)
    geography       = load_attack_geography(base_bundle["config"])

    model, ckpt = load_model(model_path, device)
    hub_feats, hubs, hub_idx = build_hub_features(instance)
    hub_feats = hub_feats.to(device)

    print(f"Model checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"val_acc={ckpt.get('val_acc', 0):.4f}  "
          f"val_auc={ckpt.get('val_auc', 0):.4f}")
    print(f"Test instances: {n_test}  |S|={n_scenarios}  delta={delta}  "
          f"alphas={[f'{a:.0%}' for a in alphas]}")
    if snap_times:
        print(f"Primal snapshots at: {[f'{int(t)}s' for t in snap_times]}")
    print()

    all_results: list[list[SolveResult]] = []

    for t in range(n_test):
        trial_seed = seed + t * n_scenarios
        rng        = np.random.default_rng(trial_seed)
        K_seq      = _sample_K_sequence(n_scenarios, rng)

        scenarios = _generate_scenario_batch(
            instance=instance,
            base_bundle=base_bundle,
            geography=geography,
            params_template=params_template,
            K_sequence=K_seq,
            attack_mode=attack_mode,
            trial_id=t,
            base_seed=trial_seed,
            threshold_by_k=DEFAULT_THRESHOLD_BY_K,
        )

        k_dist = dict(sorted({k: K_seq.count(k) for k in set(K_seq)}.items()))
        print(f"Test {t+1}/{n_test}  seed={trial_seed}  K={k_dist}")

        probs = predict_probs(model, instance, scenarios, hub_feats, hubs, hub_idx, device)

        instance_results: list[SolveResult] = []

        # baseline
        r = solve_baseline(instance, scenarios, delta, time_limit, snap_times)
        instance_results.append(r)
        status = "OPTIMAL" if r.gap == 0.0 else "TIME_LIMIT"
        print(f"  {'Gurobi':14s}  time={r.time_s:6.1f}s  obj={r.obj:7.2f}  "
              f"gap={r.gap:.2%}  nodes={r.nodes:,}  [{status}]")
        if snap_times:
            print(_snapshot_str(r.snapshots, r.time_s))

        # P+S at each alpha
        for alpha in alphas:
            r = solve_predict_search(
                instance, scenarios, probs, hubs, alpha, delta, time_limit, snap_times
            )
            feasible_str = "OK" if r.feasible else "INFEASIBLE"
            baseline     = instance_results[0]
            speedup      = baseline.time_s / r.time_s if r.feasible else float("nan")
            obj_gap      = ((r.obj - baseline.obj) / baseline.obj * 100
                           if (r.feasible and baseline.feasible) else float("nan"))
            instance_results.append(r)
            print(f"  {r.label:14s}  time={r.time_s:6.1f}s  obj={r.obj:7.2f}  "
                  f"gap={r.gap:.2%}  nodes={r.nodes:,}  "
                  f"speedup={speedup:.2f}x  obj_gap={obj_gap:+.2f}%  "
                  f"fixed={r.n_fixed}  [{feasible_str}]")
            if snap_times:
                print(_snapshot_str(r.snapshots, r.time_s))

        all_results.append(instance_results)
        print()

    # ---- summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    labels = ["Gurobi"] + [f"P+S a={a:.0%}" for a in alphas]

    # per-method stats
    print(f"{'':16s}  {'Avg time':>10}  {'Feasible':>9}  "
          f"{'Avg speedup':>12}  {'Avg obj gap':>12}")
    print("-" * 65)

    baseline_times = [res[0].time_s for res in all_results]

    for col, label in enumerate(labels):
        times     = [res[col].time_s for res in all_results if res[col].feasible]
        feasibles = sum(1 for res in all_results if res[col].feasible)
        speedups  = [baseline_times[i] / all_results[i][col].time_s
                     for i in range(n_test) if all_results[i][col].feasible]
        obj_gaps  = [((all_results[i][col].obj - all_results[i][0].obj) / all_results[i][0].obj * 100)
                     for i in range(n_test)
                     if all_results[i][col].feasible and all_results[i][0].feasible and col > 0]

        avg_time    = float(np.mean(times))    if times    else float("nan")
        avg_speedup = float(np.mean(speedups)) if speedups else float("nan")
        avg_obj_gap = float(np.mean(obj_gaps)) if obj_gaps else float("nan")

        speedup_str  = f"{avg_speedup:.2f}x" if col > 0 else "1.00x"
        obj_gap_str  = f"{avg_obj_gap:+.2f}%" if col > 0 else "0.00%"
        print(f"  {label:16s}  {avg_time:>9.1f}s  {feasibles:>7}/{n_test}  "
              f"{speedup_str:>12}  {obj_gap_str:>12}")

    # snapshot comparison table (only when snapshots were requested)
    if snap_times and all_results:
        print(f"\nPrimal bound at snapshot times (avg over feasible instances):")
        header = f"{'':16s}" + "".join(f"  @{int(t)}s" .rjust(10) for t in snap_times)
        print(header)
        print("-" * (16 + 10 * len(snap_times) + 2))

        for col, label in enumerate(labels):
            row = f"  {label:16s}"
            for snap_t in snap_times:
                objs_at_t = []
                for res in all_results:
                    r = res[col]
                    # find snapshot record for this time
                    match = [obj for t, obj in r.snapshots if t == snap_t]
                    if match and not np.isnan(match[0]):
                        objs_at_t.append(match[0])
                    elif r.feasible and r.time_s <= snap_t:
                        # solved before this snapshot — use final obj
                        objs_at_t.append(r.obj)
                avg = float(np.mean(objs_at_t)) if objs_at_t else float("nan")
                row += f"  {avg:>8.1f}" if not np.isnan(avg) else f"  {'n/a':>8}"
            print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="models/hub_gnn_bce.pt")
    p.add_argument("--n-test",      type=int,   default=10)
    p.add_argument("--n-scenarios", type=int,   default=20)
    p.add_argument("--alphas",      type=float, nargs="+", default=[0.25, 0.50, 0.75])
    p.add_argument("--delta",       type=float, default=0.10)
    p.add_argument("--time-limit",  type=int,   default=600)
    p.add_argument("--attack-mode", default="combo")
    p.add_argument("--seed",        type=int,   default=TEST_BASE_SEED)
    p.add_argument("--snapshots",   type=float, nargs="*", default=[],
                   help="Record primal bound at these times (seconds). "
                        "E.g. --snapshots 30 60 120 300 600")
    args = p.parse_args()

    run_experiment(
        model_path=args.model,
        n_test=args.n_test,
        n_scenarios=args.n_scenarios,
        alphas=args.alphas,
        delta=args.delta,
        time_limit=args.time_limit,
        attack_mode=args.attack_mode,
        seed=args.seed,
        snap_times=sorted(args.snapshots),
    )
