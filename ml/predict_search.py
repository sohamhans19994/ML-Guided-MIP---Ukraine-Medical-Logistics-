"""
ml/predict_search.py
--------------------
Comprehensive Predict + Search evaluation supporting multiple methods,
multiple scenario sizes, and primal-bound snapshots.

Methods
-------
  gurobi     : Pure Gurobi baseline (no ML)
  ps         : Predict+Search — hard-fix both open and closed predictions
  open-only  : Predict+Search — fix only predicted-open hubs, leave closed hubs free
  repair     : Predict+Search with alpha repair — reduce alpha until feasible
  lns        : LNS — GNN-guided destroy/repair starting from an initial P+S solution

Usage
-----
  # single scenario size, all methods
  uv run python -m ml.predict_search --scenario-sizes 20 --methods all

  # sweep scenario sizes, selected methods
  uv run python -m ml.predict_search \\
      --model models/hub_gnn_bce.pt \\
      --scenario-sizes 20 40 60 \\
      --methods gurobi ps open-only repair lns \\
      --alphas 0.50 0.75 \\
      --time-limit 600 \\
      --snapshots 30 60 120 300 600
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
ALL_METHODS    = ["gurobi", "ps", "open-only", "repair", "lns"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    label:          str
    time_s:         float
    obj:            float
    gap:            float
    nodes:          int
    feasible:       bool
    n_fixed:        int   = 0
    alpha_used:     float = 0.0   # final alpha (repair shows which alpha worked)
    lns_iters_done: int   = 0     # LNS: iterations completed
    lns_improved:   int   = 0     # LNS: iterations that improved the solution
    snapshots: list[tuple[float, float]] = field(default_factory=list)
    y_vals:    dict[int, int] | None     = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Primal-bound callback (version-safe)
# ---------------------------------------------------------------------------

def _make_callback(model: gp.Model, snap_times: list[float]):
    records: list[tuple[float, float]] = []
    idx = [0]

    def cb(*args) -> None:
        where = args[-1]   # (model, where) old API or (where,) new API
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
# Shared solve executor
# ---------------------------------------------------------------------------

def _run_model(
    model:      gp.Model,
    variables:  dict,
    hubs:       list[int],
    snap_times: list[float],
    time_limit: int,
    label:      str,
    n_fixed:    int    = 0,
    alpha_used: float  = 0.0,
) -> SolveResult:
    """Optimize a built model and return a SolveResult with y_vals extracted."""
    model.Params.TimeLimit = time_limit
    cb, records = _make_callback(model, snap_times)

    t0 = time.perf_counter()
    model.optimize(cb)
    elapsed = time.perf_counter() - t0

    feasible = model.SolCount > 0
    y = variables["y"]

    y_vals = None
    if feasible:
        y_vals = {h: int(round(float(y[h].X))) for h in hubs}

    return SolveResult(
        label=label,
        time_s=elapsed,
        obj=float(model.ObjVal)  if feasible else float("nan"),
        gap=float(model.MIPGap)  if feasible else float("nan"),
        nodes=int(model.NodeCount),
        feasible=feasible,
        n_fixed=n_fixed,
        alpha_used=alpha_used,
        snapshots=records,
        y_vals=y_vals,
    )


# ---------------------------------------------------------------------------
# Fixing helpers
# ---------------------------------------------------------------------------

def _apply_fixing_both(
    y_vars:   dict,
    hubs:     list[int],
    probs_np: np.ndarray,
    hub_idx:  dict[int, int],
    alpha:    float,
    model:    gp.Model,
) -> int:
    """Fix top-alpha% hubs both open and closed. Returns number fixed."""
    n_fix       = max(1, int(round(alpha * len(hubs))))
    confidence  = np.abs(probs_np - 0.5) * 2.0
    fix_indices = np.argsort(confidence)[::-1][:n_fix]

    for idx in fix_indices:
        hub_id  = hubs[idx]
        fix_val = 1 if probs_np[idx] >= 0.5 else 0
        model.addConstr(y_vars[hub_id] == fix_val, name=f"fix_{hub_id}")

    model.update()
    return n_fix


def _apply_fixing_open_only(
    y_vars:   dict,
    hubs:     list[int],
    probs_np: np.ndarray,
    hub_idx:  dict[int, int],
    alpha:    float,
    model:    gp.Model,
) -> int:
    """Fix the top-alpha% of predicted-open hubs (prob>=0.5) ranked by confidence.
    Predicted-closed hubs are always left free — eliminates infeasibility risk.

    Alpha applies only within the open-prediction pool, not across all hubs.
    E.g. if 8 hubs are predicted open and alpha=0.50, fix the 4 most confident."""
    open_indices = np.where(probs_np >= 0.5)[0]
    if len(open_indices) == 0:
        return 0

    # rank predicted-open hubs by confidence (distance from 0.5)
    open_confidence = np.abs(probs_np[open_indices] - 0.5) * 2.0
    n_fix           = max(1, int(round(alpha * len(open_indices))))
    top_open        = open_indices[np.argsort(open_confidence)[::-1][:n_fix]]

    for idx in top_open:
        model.addConstr(y_vars[hubs[idx]] == 1, name=f"fix_open_{hubs[idx]}")

    model.update()
    return len(top_open)


# ---------------------------------------------------------------------------
# Solve methods
# ---------------------------------------------------------------------------

def solve_baseline(
    instance, scenarios, delta: float, time_limit: int, snap_times: list[float],
) -> SolveResult:
    model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    return _run_model(model, variables, sorted(instance.N), snap_times, time_limit, "Gurobi")


def solve_predict_search(
    instance, scenarios, probs: torch.Tensor, hubs: list[int], hub_idx: dict[int, int],
    alpha: float, delta: float, time_limit: int, snap_times: list[float],
) -> SolveResult:
    probs_np = probs.detach().cpu().numpy()
    model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    n_fixed = _apply_fixing_both(variables["y"], hubs, probs_np, hub_idx, alpha, model)
    r = _run_model(model, variables, hubs, snap_times, time_limit,
                   f"PS a={alpha:.0%}", n_fixed=n_fixed, alpha_used=alpha)
    return r


def solve_open_only(
    instance, scenarios, probs: torch.Tensor, hubs: list[int], hub_idx: dict[int, int],
    alpha: float, delta: float, time_limit: int, snap_times: list[float],
) -> SolveResult:
    probs_np = probs.detach().cpu().numpy()
    model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    n_fixed = _apply_fixing_open_only(variables["y"], hubs, probs_np, hub_idx, alpha, model)
    r = _run_model(model, variables, hubs, snap_times, time_limit,
                   f"OpenOnly a={alpha:.0%}", n_fixed=n_fixed, alpha_used=alpha)
    return r


def solve_with_alpha_repair(
    instance, scenarios, probs: torch.Tensor, hubs: list[int], hub_idx: dict[int, int],
    alpha: float, delta: float, time_limit: int, snap_times: list[float],
    repair_step: float = 0.10,
) -> SolveResult:
    """Try P+S at alpha; if infeasible reduce alpha by repair_step and retry."""
    probs_np      = probs.detach().cpu().numpy()
    current_alpha = alpha
    t_start       = time.perf_counter()

    while current_alpha > 0:
        remaining = time_limit - (time.perf_counter() - t_start)
        if remaining <= 0:
            break

        model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
        n_fixed = _apply_fixing_both(variables["y"], hubs, probs_np, hub_idx, current_alpha, model)
        r = _run_model(model, variables, hubs, snap_times, int(remaining),
                       f"Repair a={current_alpha:.0%}", n_fixed=n_fixed, alpha_used=current_alpha)

        if r.feasible:
            r.label = f"Repair(final a={current_alpha:.0%})"
            return r

        current_alpha = round(current_alpha - repair_step, 10)

    # fallback to baseline
    remaining = max(1, int(time_limit - (time.perf_counter() - t_start)))
    r = solve_baseline(instance, scenarios, delta, remaining, snap_times)
    r.label = "Repair(fallback)"
    return r


def solve_lns(
    instance, scenarios, probs: torch.Tensor, hubs: list[int], hub_idx: dict[int, int],
    delta: float, total_time_limit: int, snap_times: list[float],
    n_iters: int, k_destroy: int, iter_time_limit: int,
    init_alpha: float = 0.50, repair_step: float = 0.10,
) -> SolveResult:
    """
    LNS with GNN-guided destroy operator.

    Phase 1 — Initialise:
      Try P+S at init_alpha; if infeasible, apply alpha repair; if still
      infeasible, run a short Gurobi solve (up to 1/3 of total budget).

    Phase 2 — Improve:
      For each iteration, destroy k_destroy hubs (prioritised by GNN
      uncertainty — least confident hubs destroyed first), fix all others
      at current solution values, re-solve within iter_time_limit.
      Accept if objective improves.
    """
    probs_np  = probs.detach().cpu().numpy()
    t_start   = time.perf_counter()

    # ---- Phase 1: get initial feasible solution ----
    init_budget = min(total_time_limit // 3, 120)

    # Try P+S with alpha repair for init
    init_r = solve_with_alpha_repair(
        instance, scenarios, probs, hubs, hub_idx,
        init_alpha, delta, init_budget, [], repair_step,
    )

    if not init_r.feasible:
        # Last resort: short Gurobi
        remaining_init = max(1, init_budget - int(time.perf_counter() - t_start))
        init_r = solve_baseline(instance, scenarios, delta, remaining_init, [])

    if not init_r.feasible:
        elapsed = time.perf_counter() - t_start
        return SolveResult(
            label="LNS", time_s=elapsed, obj=float("nan"),
            gap=float("nan"), nodes=0, feasible=False,
        )

    y_current  = dict(init_r.y_vals)        # hub_id -> 0/1
    obj_current = init_r.obj
    total_nodes = init_r.nodes

    # uncertainty: low value = model is uncertain about this hub
    uncertainty = np.abs(probs_np - 0.5)    # [n_hubs], 0=uncertain 0.5=certain

    iters_done  = 0
    iters_improved = 0
    best_snapshots: list[tuple[float, float]] = []

    # ---- Phase 2: destroy-repair iterations ----
    for it in range(n_iters):
        elapsed = time.perf_counter() - t_start
        if elapsed >= total_time_limit:
            break

        remaining = total_time_limit - elapsed
        it_limit  = min(iter_time_limit, int(remaining))
        if it_limit <= 5:
            break

        # Destroy: sample k_destroy hubs weighted by inverse uncertainty
        # (least confident hubs get highest probability of being freed)
        inv_conf = 1.0 / (uncertainty + 1e-6)
        weights  = inv_conf / inv_conf.sum()
        destroy_indices = np.random.choice(
            len(hubs), size=min(k_destroy, len(hubs)), replace=False, p=weights,
        )
        destroy_set = {hubs[i] for i in destroy_indices}

        # Repair: build new model, fix non-destroyed hubs at y_current
        model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
        y = variables["y"]
        for j in hubs:
            if j not in destroy_set:
                model.addConstr(y[j] == y_current[j], name=f"lns_fix_{j}")
        model.update()

        r = _run_model(model, variables, hubs, [], it_limit,
                       f"LNS_iter{it+1}", n_fixed=len(hubs)-len(destroy_set))
        total_nodes += r.nodes
        iters_done  += 1

        if r.feasible and r.obj < obj_current - 1e-4:
            obj_current  = r.obj
            y_current    = dict(r.y_vals)
            iters_improved += 1

    # Record primal snapshots relative to t_start for the best obj found
    elapsed_total = time.perf_counter() - t_start
    return SolveResult(
        label="LNS",
        time_s=elapsed_total,
        obj=obj_current,
        gap=float("nan"),   # true gap vs optimal is unknown; use obj comparison in summary
        nodes=total_nodes,
        feasible=True,
        lns_iters_done=iters_done,
        lns_improved=iters_improved,
        y_vals=y_current,
    )


# ---------------------------------------------------------------------------
# GNN helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[HubGNN, dict]:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args  = ckpt.get("args", {})
    model = HubGNN(
        hidden_dim=args.get("hidden_dim", 64),
        n_rounds=args.get("n_rounds", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_probs(model, instance, scenarios, hub_feats, hubs, hub_idx, device):
    graph = build_instance_graph(instance, scenarios, hub_feats, hubs, hub_idx)
    return model(graph.to(device)).cpu()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _snapshot_str(r: SolveResult, snap_times: list[float]) -> str:
    if not snap_times:
        return ""
    if not r.snapshots:
        return f"    (solved in {r.time_s:.1f}s — before first snapshot)"
    parts = [f"@{int(t)}s={obj:.1f}" if not np.isnan(obj) else f"@{int(t)}s=n/a"
             for t, obj in r.snapshots]
    return "    Primal: " + "  ".join(parts)


def _print_result(r: SolveResult, baseline_time: float, snap_times: list[float]) -> None:
    if not r.feasible:
        status = "INFEASIBLE"
    elif np.isnan(r.gap):
        status = "FEASIBLE"
    elif r.gap == 0.0:
        status = "OPTIMAL"
    else:
        status = "TIME_LIMIT"
    speedup  = baseline_time / r.time_s if (r.feasible and r.time_s > 0) else float("nan")
    gap_str  = f"{r.gap:.2%}" if not np.isnan(r.gap) else "nan"
    spd_str  = f"{speedup:.2f}x" if not np.isnan(speedup) else "nan"
    obj_str  = f"{r.obj:8.2f}" if not np.isnan(r.obj) else "     nan"
    extra = ""
    if r.lns_iters_done > 0:
        extra = f"  iters={r.lns_iters_done}(+{r.lns_improved})"

    print(f"  {r.label:22s}  time={r.time_s:6.1f}s  obj={obj_str}  "
          f"gap={gap_str:>7}  nodes={r.nodes:>8,}  "
          f"speedup={spd_str:>7}  fixed={r.n_fixed}{extra}  [{status}]")
    if snap_times:
        s = _snapshot_str(r, snap_times)
        if s:
            print(s)


# ---------------------------------------------------------------------------
# Per-size experiment runner
# ---------------------------------------------------------------------------

def run_size(
    n_scenarios:   int,
    methods:       list[str],
    alphas:        list[float],
    n_test:        int,
    delta:         float,
    time_limit:    int,
    attack_mode:   str,
    seed:          int,
    snap_times:    list[float],
    repair_step:   float,
    lns_iters:     int,
    lns_destroy:   int,
    lns_iter_time: int,
    instance,
    params_template,
    base_bundle,
    geography,
    model_gnn,
    hub_feats,
    hubs,
    hub_idx,
    device,
) -> list[list[SolveResult]]:
    """Run all methods on n_test fresh instances for a given scenario count."""

    print(f"\n{'='*70}")
    print(f"|S| = {n_scenarios}   delta={delta}   n_test={n_test}   time_limit={time_limit}s")
    print(f"{'='*70}")

    all_results: list[list[SolveResult]] = []

    for t in range(n_test):
        trial_seed = seed + t * n_scenarios
        rng        = np.random.default_rng(trial_seed)
        K_seq      = _sample_K_sequence(n_scenarios, rng)
        scenarios  = _generate_scenario_batch(
            instance=instance, base_bundle=base_bundle, geography=geography,
            params_template=params_template, K_sequence=K_seq,
            attack_mode=attack_mode, trial_id=t, base_seed=trial_seed,
            threshold_by_k=DEFAULT_THRESHOLD_BY_K,
        )

        k_dist = dict(sorted({k: K_seq.count(k) for k in set(K_seq)}.items()))
        print(f"\nTest {t+1}/{n_test}  seed={trial_seed}  K={k_dist}")

        probs = predict_probs(model_gnn, instance, scenarios, hub_feats, hubs, hub_idx, device)
        instance_results: list[SolveResult] = []

        # run each method in order
        for method in methods:
            if method == "gurobi":
                r = solve_baseline(instance, scenarios, delta, time_limit, snap_times)
                instance_results.append(r)
                baseline_t = r.time_s
                _print_result(r, r.time_s, snap_times)

            elif method == "ps":
                for alpha in alphas:
                    r = solve_predict_search(
                        instance, scenarios, probs, hubs, hub_idx,
                        alpha, delta, time_limit, snap_times,
                    )
                    baseline_t = instance_results[0].time_s if instance_results else r.time_s
                    instance_results.append(r)
                    _print_result(r, baseline_t, snap_times)

            elif method == "open-only":
                for alpha in alphas:
                    r = solve_open_only(
                        instance, scenarios, probs, hubs, hub_idx,
                        alpha, delta, time_limit, snap_times,
                    )
                    baseline_t = instance_results[0].time_s if instance_results else r.time_s
                    instance_results.append(r)
                    _print_result(r, baseline_t, snap_times)

            elif method == "repair":
                for alpha in alphas:
                    r = solve_with_alpha_repair(
                        instance, scenarios, probs, hubs, hub_idx,
                        alpha, delta, time_limit, snap_times, repair_step,
                    )
                    baseline_t = instance_results[0].time_s if instance_results else r.time_s
                    instance_results.append(r)
                    _print_result(r, baseline_t, snap_times)

            elif method == "lns":
                r = solve_lns(
                    instance, scenarios, probs, hubs, hub_idx,
                    delta, time_limit, snap_times,
                    n_iters=lns_iters, k_destroy=lns_destroy,
                    iter_time_limit=lns_iter_time,
                    init_alpha=alphas[0] if alphas else 0.50,
                    repair_step=repair_step,
                )
                baseline_t = instance_results[0].time_s if instance_results else r.time_s
                instance_results.append(r)
                _print_result(r, baseline_t, snap_times)

        all_results.append(instance_results)

    return all_results


def print_summary(all_results: list[list[SolveResult]], n_test: int, snap_times: list[float]) -> None:
    if not all_results:
        return

    n_methods = len(all_results[0])
    baseline_times = [res[0].time_s for res in all_results]

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':24s}  {'Avg time':>10}  {'Feasible':>9}  {'Avg speedup':>12}  {'Avg obj gap':>12}")
    print(f"  {'-'*66}")

    for col in range(n_methods):
        label     = all_results[0][col].label
        feasibles = [res[col] for res in all_results if res[col].feasible]
        times     = [r.time_s for r in feasibles]
        speedups  = [baseline_times[i] / all_results[i][col].time_s
                     for i in range(n_test) if all_results[i][col].feasible]
        obj_gaps  = [((all_results[i][col].obj - all_results[i][0].obj) / all_results[i][0].obj * 100)
                     for i in range(n_test)
                     if all_results[i][col].feasible and all_results[i][0].feasible and col > 0]

        avg_t   = float(np.mean(times))    if times    else float("nan")
        avg_spd = float(np.mean(speedups)) if speedups else float("nan")
        avg_gap = float(np.mean(obj_gaps)) if obj_gaps else float("nan")

        spd_str = f"{avg_spd:.2f}x" if col > 0 else "1.00x"
        gap_str = f"{avg_gap:+.2f}%" if col > 0 and not np.isnan(avg_gap) else "0.00%"
        t_str   = f"{avg_t:.1f}s"   if not np.isnan(avg_t) else "nan"
        print(f"  {label:24s}  {t_str:>10}  {len(feasibles):>7}/{n_test}  "
              f"{spd_str:>12}  {gap_str:>12}")

    if snap_times and all_results:
        print(f"\n  Primal bound at snapshot times (avg over feasible instances):")
        header = f"  {'':24s}" + "".join(f"  @{int(t)}s".rjust(9) for t in snap_times)
        print(header)
        print(f"  {'-'*62}")
        for col in range(n_methods):
            label = all_results[0][col].label
            row   = f"  {label:24s}"
            for snap_t in snap_times:
                objs = []
                for res in all_results:
                    r = res[col]
                    match = [obj for st, obj in r.snapshots if st == snap_t]
                    if match and not np.isnan(match[0]):
                        objs.append(match[0])
                    elif r.feasible and r.time_s <= snap_t:
                        objs.append(r.obj)
                avg = float(np.mean(objs)) if objs else float("nan")
                row += f"  {avg:>7.1f}" if not np.isnan(avg) else f"  {'n/a':>7}"
            print(row)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    model_path:     str,
    scenario_sizes: list[int],
    methods:        list[str],
    n_test:         int,
    alphas:         list[float],
    delta:          float,
    time_limit:     int,
    attack_mode:    str,
    seed:           int,
    snap_times:     list[float],
    repair_step:    float,
    lns_iters:      int,
    lns_destroy:    int,
    lns_iter_time:  int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading instance and model ...")
    instance        = load_instance()
    params_template = load_scenario_parameters()
    base_bundle     = load_base_bundle(params_template.bundle_path)
    geography       = load_attack_geography(base_bundle["config"])

    model_gnn, ckpt = load_model(model_path, device)
    hub_feats, hubs, hub_idx = build_hub_features(instance)
    hub_feats = hub_feats.to(device)

    print(f"Model checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"val_acc={ckpt.get('val_acc', 0):.4f}  val_auc={ckpt.get('val_auc', 0):.4f}")
    print(f"Methods: {methods}  Alphas: {[f'{a:.0%}' for a in alphas]}")
    print(f"Scenario sizes: {scenario_sizes}  n_test={n_test}  time_limit={time_limit}s")

    for n_scenarios in scenario_sizes:
        results = run_size(
            n_scenarios=n_scenarios, methods=methods, alphas=alphas,
            n_test=n_test, delta=delta, time_limit=time_limit,
            attack_mode=attack_mode, seed=seed, snap_times=snap_times,
            repair_step=repair_step, lns_iters=lns_iters,
            lns_destroy=lns_destroy, lns_iter_time=lns_iter_time,
            instance=instance, params_template=params_template,
            base_bundle=base_bundle, geography=geography,
            model_gnn=model_gnn, hub_feats=hub_feats,
            hubs=hubs, hub_idx=hub_idx, device=device,
        )
        print_summary(results, n_test, snap_times)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="models/hub_gnn_bce.pt")
    p.add_argument("--scenario-sizes", type=int,   nargs="+", default=[20])
    p.add_argument("--methods",        nargs="+",  default=["gurobi", "ps"],
                   help=f"Methods to run. Options: {ALL_METHODS} or 'all'")
    p.add_argument("--n-test",         type=int,   default=10)
    p.add_argument("--alphas",         type=float, nargs="+", default=[0.50, 0.75])
    p.add_argument("--delta",          type=float, default=0.10)
    p.add_argument("--time-limit",     type=int,   default=600)
    p.add_argument("--attack-mode",    default="combo")
    p.add_argument("--seed",           type=int,   default=TEST_BASE_SEED)
    p.add_argument("--snapshots",      type=float, nargs="*", default=[])
    p.add_argument("--repair-step",    type=float, default=0.10)
    p.add_argument("--lns-iters",      type=int,   default=5)
    p.add_argument("--lns-destroy",    type=int,   default=20)
    p.add_argument("--lns-iter-time",  type=int,   default=120)
    args = p.parse_args()

    methods = ALL_METHODS if "all" in args.methods else args.methods

    run_experiment(
        model_path=args.model,
        scenario_sizes=args.scenario_sizes,
        methods=methods,
        n_test=args.n_test,
        alphas=args.alphas,
        delta=args.delta,
        time_limit=args.time_limit,
        attack_mode=args.attack_mode,
        seed=args.seed,
        snap_times=sorted(args.snapshots),
        repair_step=args.repair_step,
        lns_iters=args.lns_iters,
        lns_destroy=args.lns_destroy,
        lns_iter_time=args.lns_iter_time,
    )
