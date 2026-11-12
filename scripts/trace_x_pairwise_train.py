import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class PairwiseEvidenceNet(nn.Module):
    """Symmetric pair classifier: swapping evidence A/B leaves features unchanged."""
    def __init__(self, input_dim=387, hidden=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pairs(records, features):
    if len(records) != len(features):
        raise ValueError(
            f"Alignment error: {len(records)} JSON records but "
            f"{len(features)} feature rows."
        )

    groups = {}
    for idx, row in enumerate(records):
        groups.setdefault(str(row["incident_id"]), []).append(idx)

    pair_features, labels = [], []
    for incident_id, indices in groups.items():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                a = features[i].astype(np.float32)
                b = features[j].astype(np.float32)

                # Symmetric features: no dependence on pair ordering.
                diff = np.abs(a - b)
                product = a * b

                ta = float(records[i].get("claimed_timestamp", 0))
                tb = float(records[j].get("claimed_timestamp", 0))
                time_delta = np.array(
                    [min(abs(ta - tb) / 86400.0, 30.0) / 30.0],
                    dtype=np.float32
                )

                pair_features.append(np.concatenate([diff, product, time_delta]))

                # event_id is used ONLY to construct the target label.
                # It is never passed to the model.
                ei = records[i].get("event_id", -1)
                ej = records[j].get("event_id", -1)
                is_related = int(ei not in (-1, None, "-1") and ei == ej)
                labels.append(is_related)

    return np.asarray(pair_features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def balanced_training_indices(y, negative_ratio=4):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) == 0:
        raise ValueError("No positive training pairs found.")
    rng = np.random.default_rng(SEED)
    n_negative = min(len(negative), len(positive) * negative_ratio)
    chosen_negative = rng.choice(negative, size=n_negative, replace=False)
    selected = np.concatenate([positive, chosen_negative])
    rng.shuffle(selected)
    return selected


def evaluate(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds):
        f1_values = 2 * precision[:-1] * recall[:-1] / (
            precision[:-1] + recall[:-1] + 1e-12
        )
        best_idx = int(np.nanargmax(f1_values))
        threshold = float(thresholds[best_idx])
        best_f1 = float(f1_values[best_idx])
    else:
        threshold, best_f1 = 0.5, 0.0

    pred = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "best_curve_f1": best_f1,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y_true, pred, labels=[0, 1]
        ).ravel().tolist(),
        "predicted_positive": int(pred.sum()),
        "total_pairs": int(len(y_true)),
        "positive_pairs": int(np.sum(y_true == 1)),
        "negative_pairs": int(np.sum(y_true == 0)),
    }


def predict(model, x, device, batch_size=2048):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "synthetic"
    feature_path = root / "features" / "evidence_features.npz"
    model_dir = root / "models"
    results_dir = root / "results"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print("Loading cached feature vectors and split records...")
    cached = np.load(feature_path)
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records = load_jsonl(data_dir / "validation.jsonl")

    x_train, y_train = build_pairs(train_records, cached["train_X"])
    x_val, y_val = build_pairs(val_records, cached["validation_X"])

    print(f"Training pairs: {len(y_train)} | positive={int(y_train.sum())} "
          f"| negative={int((y_train == 0).sum())}")
    print(f"Validation pairs: {len(y_val)} | positive={int(y_val.sum())} "
          f"| negative={int((y_val == 0).sum())}")

    # Balance only the training sampler. Validation remains naturally imbalanced.
    selected = balanced_training_indices(y_train, negative_ratio=4)
    print(f"Balanced training subset: {len(selected)} pairs "
          f"(1 positive : at most 4 negatives)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairwiseEvidenceNet(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(x_train[selected]),
        torch.from_numpy(y_train[selected].astype(np.float32))
    )
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_ap = -1.0
    best_state = None
    best_metrics = None
    stale = 0

    print(f"Device: {device}")
    print("Training a separate pairwise classifier; encoders and graph checkpoints "
          "will not be modified.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_probs = predict(model, x_val, device)
        val_ap = float(average_precision_score(y_val, val_probs))
        val_auc = float(roc_auc_score(y_val, val_probs))
        metrics = evaluate(y_val, val_probs)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={np.mean(losses):.4f} val_AP={val_ap:.4f} "
            f"val_ROC_AUC={val_auc:.4f} val_F1={metrics['f1']:.4f} "
            f"threshold={metrics['threshold']:.4f}"
        )

        if val_ap > best_ap + 1e-6:
            best_ap = val_ap
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {args.patience} epochs without AP improvement.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path = model_dir / "trace_pairwise_edge.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "feature_definition": "abs(a-b), a*b, normalized absolute timestamp delta",
        "seed": SEED,
    }, checkpoint_path)

    report = {
        "model": "PairwiseEvidenceNet",
        "device": str(device),
        "training_pairs_total": int(len(y_train)),
        "training_pairs_sampled": int(len(selected)),
        "validation": best_metrics,
        "selection_metric": "validation average precision",
        "test_used_for_training_or_selection": False,
        "encoders_retrained": False,
        "original_graph_checkpoint_modified": False,
        "limitations": (
            "Synthetic benchmark only. event_id is used only for pair labels, "
            "not as a model input. Scores are not forensic proof."
        ),
    }
    report_path = results_dir / "trace_pairwise_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPAIRWISE CLASSIFIER TRAINING COMPLETE")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved validation report: {report_path}")
    print(f"Best validation AP: {best_metrics['average_precision']:.4f}")
    print(f"Validation ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Validation precision/recall/F1: "
          f"{best_metrics['precision']:.4f} / "
          f"{best_metrics['recall']:.4f} / {best_metrics['f1']:.4f}")
    print("Original encoders and trace_graph.pt were not changed.")
    print("Do not interpret synthetic benchmark scores as real-world forensic validity.")


if __name__ == "__main__":
    main()
