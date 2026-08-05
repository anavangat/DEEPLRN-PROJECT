"""Shared Stage 2 train/eval loops.

Used by the proposed twostage model and by every baseline, so that
the only difference between methods is the model and the input, not the
optimization code.
"""
import copy

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

LABELS = list(range(36))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns the full metric dict plus raw arrays for predictions.csv."""
    model.eval()
    loss_sum, total = 0.0, 0
    ys, ps, cs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        total += y.size(0)
        conf, pred = torch.softmax(out, dim=1).max(dim=1)
        ys.append(y.cpu()); ps.append(pred.cpu()); cs.append(conf.cpu())

    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy(); c = torch.cat(cs).numpy()
    return {
        "loss": loss_sum / total,
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(precision_score(y, p, average="macro",
                                                 labels=LABELS, zero_division=0)),
        "macro_recall": float(recall_score(y, p, average="macro",
                                           labels=LABELS, zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro",
                                   labels=LABELS, zero_division=0)),
        "per_class_f1": [float(v) for v in
                         f1_score(y, p, average=None, labels=LABELS, zero_division=0)],
        "y_true": y, "y_pred": p, "confidence": c,
    }


def fit(model, train_dl, val_dl, criterion, optimizer, scheduler=None,
        epochs=60, patience=15, device=None, trial=None, verbose=True):
    """Trains, selecting on VALIDATION MACRO-F1. Restores the best weights.

    Model selection is on macro-F1 rather than accuracy because the dataset is
    imbalanced 2.62:1 and accuracy flatters a model that neglects digits.
    """
    device = device or get_device()
    best_f1, best_epoch, best_state, bad = -1.0, -1, None, 0
    history = []

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        v = evaluate(model, val_dl, criterion, device)
        if scheduler is not None:
            scheduler.step(v["macro_f1"])       # ReduceLROnPlateau(mode="max")
        history.append([ep, tr_loss, v["loss"], tr_acc, v["accuracy"]])

        if verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(f"ep {ep:3d}  train {tr_loss:.4f}/{tr_acc:.4f}  "
                  f"val {v['loss']:.4f}/{v['accuracy']:.4f}  "
                  f"macroF1 {v['macro_f1']:.4f}  lr {lr:.2e}", flush=True)

        if trial is not None:
            trial.report(v["macro_f1"], ep)
            import optuna
            if trial.should_prune():
                raise optuna.TrialPruned()

        if v["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, bad = v["macro_f1"], ep, 0
            best_state = copy.deepcopy({k: t.detach().cpu().clone()
                                        for k, t in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"early stop at epoch {ep} (best {best_epoch}, "
                          f"macroF1 {best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history}