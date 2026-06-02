"""
ml/train.py
-----------
Training loop for HubGNN.

Loss functions (--loss flag):
  bce     : Binary cross-entropy on optimal hub assignment (pool_solutions[0]).
  infonce : Log-likelihood ranking over the full solution pool.
            Encourages higher likelihood for the optimal configuration
            than near-optimal alternatives.

Usage
-----
    uv run python -m ml.train
    uv run python -m ml.train --loss infonce --epochs 150
    uv run python -m ml.train --records data/ml_training/records_s20.pkl

Outputs
-------
  models/hub_gnn_bce.pt   -- best checkpoint by val loss
  models/hub_gnn_infonce.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from mip.data import load_instance
from ml.dataset import load_splits
from ml.model import HubGNN

# These are derived from the data at runtime — see batch_loss()
# N_HUBS = number of hub candidates (112 for this instance)
# N_POOL = pool solutions per record (10 by default in ml/training.py)


# ---------------------------------------------------------------------------
# Loss functions — operate on plain tensors
# ---------------------------------------------------------------------------

def bce_loss(probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard BCE against the optimal (rounded) hub assignment."""
    return F.binary_cross_entropy(probs, y)


def infonce_loss(probs: torch.Tensor, pool_y: torch.Tensor) -> torch.Tensor:
    """
    Log-likelihood ranking loss over the solution pool.

    pool_y : [n_pool, n_hubs] — binary hub assignments, best solution at index 0.

    Computes the log-likelihood of every pool solution under the model's
    current probabilities, then applies InfoNCE:
        L = -LL(optimal) + log( sum_k exp(LL(solution_k)) )

    This pushes P(y_i=1) to assign highest likelihood to the optimal
    hub configuration versus near-optimal alternatives.
    """
    eps = 1e-7
    p = probs.clamp(eps, 1 - eps)           # [n_hubs]
    y = pool_y.round()                       # [n_pool, n_hubs] — treat as binary

    log_p   = torch.log(p)                   # [n_hubs]
    log_1mp = torch.log(1.0 - p)            # [n_hubs]

    # log-likelihood of each pool solution: [n_pool]
    ll = (y * log_p.unsqueeze(0) + (1.0 - y) * log_1mp.unsqueeze(0)).sum(dim=-1)

    # InfoNCE: positive = index 0 (lowest objective = optimal)
    return -ll[0] + torch.logsumexp(ll, dim=0)


# ---------------------------------------------------------------------------
# Batch loss — handles both BCE and InfoNCE over a batched HeteroData
# ---------------------------------------------------------------------------

def batch_loss(
    probs:     torch.Tensor,   # [B * N_HUBS]
    batch,                     # batched HeteroData
    loss_name: str,
    n_graphs:  int,
) -> torch.Tensor:
    if loss_name == "bce":
        return bce_loss(probs, batch["hub"].y)

    # InfoNCE: compute per-graph, average.
    # Derive dimensions from the batch itself — works for any |S| or pool size.
    n_hubs = probs.size(0) // n_graphs                      # hubs per graph
    n_pool = batch["hub"].pool_y.size(0) // n_graphs        # pool size per graph

    # PyG concatenates pool_y along dim 0: [B*n_pool, n_hubs]
    probs_list  = probs.split(n_hubs)
    pool_y_all  = batch["hub"].pool_y
    pool_y_list = pool_y_all.split(n_pool)      # list of [n_pool, n_hubs]

    total = torch.zeros(1, device=probs.device)
    for p, py in zip(probs_list, pool_y_list):
        total = total + infonce_loss(p, py)
    return total / n_graphs


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def evaluate(
    model:   HubGNN,
    loader:  DataLoader,
    device:  torch.device,
) -> dict:
    """Compute val loss (BCE), binary accuracy and AUC."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = model(batch)

            total_loss += bce_loss(probs, batch["hub"].y).item()
            all_probs.append(probs.cpu())
            all_labels.append(batch["hub"].y.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    accuracy = ((all_probs >= 0.5).float() == all_labels).float().mean().item()
    try:
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
    except ValueError:
        auc = float("nan")

    return {
        "loss":     total_loss / len(loader),
        "accuracy": accuracy,
        "auc":      auc,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    records_path: str,
    loss_name:    str   = "bce",
    epochs:       int   = 100,
    lr:           float = 1e-3,
    batch_size:   int   = 32,
    hidden_dim:   int   = 64,
    n_rounds:     int   = 2,
    dropout:      float = 0.0,
    save_dir:     str   = "models",
    seed:         int   = 42,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Loss: {loss_name}  Epochs: {epochs}  LR: {lr}")

    instance = load_instance()
    train_ds, val_ds, _ = load_splits(records_path, instance, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model     = HubGNN(hidden_dim=hidden_dim, n_rounds=n_rounds, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = Path(save_dir) / f"hub_gnn_{loss_name}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(f"\n{'Epoch':>6}  {'Train loss':>11}  {'Val loss':>10}  {'Val acc':>9}  {'Val AUC':>9}  {'Time':>6}")
    print("-" * 62)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            probs = model(batch)
            loss  = batch_loss(probs, batch, loss_name, batch.num_graphs)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss  = total_train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        elapsed     = time.perf_counter() - t0

        print(
            f"{epoch:>6}  {train_loss:>11.4f}  "
            f"{val_metrics['loss']:>10.4f}  "
            f"{val_metrics['accuracy']:>9.4f}  "
            f"{val_metrics['auc']:>9.4f}  "
            f"{elapsed:>5.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_metrics["loss"],
                "val_acc":     val_metrics["accuracy"],
                "val_auc":     val_metrics["auc"],
                "args": {
                    "loss":       loss_name,
                    "hidden_dim": hidden_dim,
                    "n_rounds":   n_rounds,
                    "dropout":    dropout,
                },
            }, save_path)

    print(f"\nBest val loss: {best_val_loss:.4f}  Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records",    default="data/ml_training/records_10s.pkl")
    p.add_argument("--loss",       default="bce", choices=["bce", "infonce"])
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--hidden-dim", type=int,   default=64)
    p.add_argument("--n-rounds",   type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.0)
    p.add_argument("--save-dir",   default="models")
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()
    train(
        records_path=args.records,
        loss_name=args.loss,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_rounds=args.n_rounds,
        dropout=args.dropout,
        save_dir=args.save_dir,
        seed=args.seed,
    )
