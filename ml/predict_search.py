"""
ml/predict_search.py
--------------------
Experiment 1 (and 2, 3): Predict + Search evaluation.

For each fresh test instance the script:
  1. Generates a new scenario set (unseen during training).
  2. Runs pure Gurobi as the baseline.
  3. Runs the GNN, gets P(y_i=1) per hub.
  4. For each fixing fraction alpha, adds hard equality constraints
     for the top-alpha% most confident hub predictions, then re-solves.
  5. Prints a comparison table: time, objective, gap, nodes, feasibility.

Usage
-----
    uv run python -m ml.predict_search
    uv run python -m ml.predict_search --model models/hub_gnn_bce.pt --n-test 10
    uv run python -m ml.predict_search --n-scenarios 20 --alphas 0.25 0.5 0.75

All test instances use seeds far from training seeds (base_seed=500000).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from attack_scenarios.config import load_scenario_parameters
from attack_scenarios.geometry import load_attack_geography
from attack_scenarios.io import load_base_bundle
from mip.data import load_instance
from mip.models.robust import build_robust_model
from mip.scenarios import DEFAULT_THRESHOLD_BY_K
from ml.dataset import build_hub_features, build_instance_graph
from ml.model import HubGNN
from ml.training import _generate_scenario_batch, _sample_K_sequence

TEST_BASE_SEED = 500_000   # well away from training seeds (which start at 0)


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
    n_fixed:    int = 0   # hubs fixed (0 for baseline)


# ---------------------------------------------------------------------------
# Core solve helpers
# ---------------------------------------------------------------------------

def solve_baseline(instance, scenarios, delta: float, time_limit: int) -> SolveResult:
    """Pure Gurobi — no variable fixing."""
    model, _ = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    model.Params.TimeLimit = time_limit
    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    feasible = model.SolCount > 0
    return SolveResult(
        label="Gurobi",
        time_s=elapsed,
        obj=model.ObjVal if feasible else float("nan"),
        gap=model.MIPGap if feasible else float("nan"),
        nodes=int(model.NodeCount),
        feasible=feasible,
    )


def solve_predict_search(
    instance,
    scenarios,
    probs:      torch.Tensor,   # [n_hubs]  P(y_i=1) from GNN
    hubs:       list[int],      # canonical hub ordering
    alpha:      float,          # fixing fraction
    delta:      float,
    time_limit: int,
) -> SolveResult:
    """
    Predict + Search: fix the top-alpha% most confident hub predictions,
    solve the reduced MIP.

    Confidence = |P(y_i=1) - 0.5| * 2  (0=uncertain, 1=certain).
    Top-alpha% by confidence are fixed; the rest are left free.
    """
    n_hubs   = len(hubs)
    n_fix    = max(1, int(round(alpha * n_hubs)))
    probs_np = probs.detach().cpu().numpy()

    # rank by confidence (distance from decision boundary)
    confidence = np.abs(probs_np - 0.5) * 2.0
    fix_indices = np.argsort(confidence)[::-1][:n_fix]   # top-n_fix

    model, variables = build_robust_model(instance, scenarios, delta=delta, verbose=False)
    model.Params.TimeLimit = time_limit
    y = variables["y"]

    for idx in fix_indices:
        hub_id  = hubs[idx]
        fix_val = 1 if probs_np[idx] >= 0.5 else 0
        model.addConstr(y[hub_id] == fix_val, name=f"fix_{hub_id}")

    model.update()

    t0 = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - t0

    feasible = model.SolCount > 0
    return SolveResult(
        label=f"P+S a={alpha:.0%}",
        time_s=elapsed,
        obj=model.ObjVal if feasible else float("nan"),
        gap=model.MIPGap if feasible else float("nan"),
        nodes=int(model.NodeCount),
        feasible=feasible,
        n_fixed=n_fix,
    )


# ---------------------------------------------------------------------------
# GNN inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> tuple[HubGNN, dict]:
    ckpt  = torch.load(checkpoint_path, map_location=device)
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
    """Run GNN on a fresh scenario set, return P(y_i=1) per hub."""
    graph = build_instance_graph(instance, scenarios, hub_feats, hubs, hub_idx)
    graph = graph.to(device)
    return model(graph).cpu()   # [n_hubs]


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
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load assets ----
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
    print(f"Test instances: {n_test}  |S|={n_scenarios}  "
          f"delta={delta}  alphas={[f'{a:.0%}' for a in alphas]}\n")

    all_results: list[list[SolveResult]] = []   # one list per test instance

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

        print(f"Test {t+1}/{n_test}  seed={trial_seed}  K={dict(sorted({k: K_seq.count(k) for k in set(K_seq)}.items()))}")

        # GNN prediction
        probs = predict_probs(model, instance, scenarios, hub_feats, hubs, hub_idx, device)

        instance_results: list[SolveResult] = []

        # baseline
        r = solve_baseline(instance, scenarios, delta, time_limit)
        instance_results.append(r)
        print(f"  {'Gurobi':12s}  time={r.time_s:6.1f}s  obj={r.obj:7.2f}  gap={r.gap:.2%}  nodes={r.nodes:,}")

        # P+S at each alpha
        for alpha in alphas:
            r = solve_predict_search(instance, scenarios, probs, hubs, alpha, delta, time_limit)
            feasible_str = "OK" if r.feasible else "INFEASIBLE"
            speedup = instance_results[0].time_s / r.time_s if r.feasible else float("nan")
            obj_gap = ((r.obj - instance_results[0].obj) / instance_results[0].obj * 100
                       if (r.feasible and instance_results[0].feasible) else float("nan"))
            instance_results.append(r)
            print(f"  {r.label:12s}  time={r.time_s:6.1f}s  obj={r.obj:7.2f}  "
                  f"gap={r.gap:.2%}  nodes={r.nodes:,}  "
                  f"speedup={speedup:.2f}x  obj_gap={obj_gap:+.2f}%  "
                  f"fixed={r.n_fixed}  [{feasible_str}]")

        all_results.append(instance_results)
        print()

    # ---- summary table ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    labels = ["Gurobi"] + [f"P+S a={a:.0%}" for a in alphas]
    header = f"{'':14s}  {'Avg time':>10}  {'Feasible':>9}  {'Avg speedup':>12}  {'Avg obj gap':>12}"
    print(header)
    print("-" * 62)

    baseline_times = [res[0].time_s for res in all_results]

    for col, label in enumerate(labels):
        times     = [res[col].time_s  for res in all_results if res[col].feasible]
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
        print(f"  {label:14s}  {avg_time:>9.1f}s  {feasibles:>7}/{n_test}  "
              f"{speedup_str:>12}  {obj_gap_str:>12}")


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
    )
