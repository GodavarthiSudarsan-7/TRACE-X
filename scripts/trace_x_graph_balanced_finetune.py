"""Fine-tune ONLY the TRACE-X graph network with class-weighted edge loss.

Uses cached train/validation features and the existing graph checkpoint.
Does not run the original training pipeline or modify the original checkpoint.
Does not load/use test data for training, checkpoint selection, or threshold selection.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

SEED = 42
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
EDGE_POSITIVE_WEIGHT = 10.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class TraceGraphNet(nn.Module):
    def __init__(self, in_dim=193, hidden=96):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.15))
        self.msg = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.auth = nn.Linear(hidden, 1)
        self.role = nn.Linear(hidden, 2)
        self.edge = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x, edge_index):
        h = self.node(x)
        auth = torch.sigmoid(self.auth(h)).squeeze(1)
        agg = torch.zeros_like(h)
        deg = torch.zeros((len(h), 1), device=h.device)
        src, dst = edge_index
        messages = self.msg(h[src]) * auth[src, None]
        agg.index_add_(0, dst, messages)
        deg.index_add_(0, dst, torch.ones((len(dst), 1), device=h.device))
        h2 = self.update(agg / (deg + 1e-6), h)
        role = self.role(h2)
        a, b = h2[src], h2[dst]
        edge_logits = self.edge(torch.cat([a, b, torch.abs(a-b), a*b], dim=1))
        return h2, auth, role, edge_logits

class GraphWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TraceGraphNet()
    def forward(self, *args):
        return self.net(*args)

def load_split(root, split):
    rows = [
        json.loads(line)
        for line in (root / "data" / "synthetic" / f"{split}.jsonl")
            .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with np.load(root / "features" / "evidence_features.npz") as z:
        X = z[f"{split}_X"].copy()
    if len(rows) != len(X):
        raise RuntimeError(f"{split}: {len(rows)} records but {len(X)} feature rows")
    groups = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[int(row["incident_id"])].append(idx)

    examples = []
    for incident, indices in groups.items():
        pairs = [(i, j) for i in range(len(indices)) for j in range(len(indices)) if i != j]
        edge_index = np.asarray(pairs, dtype=np.int64).T
        yrole = np.asarray(
            [int(rows[g]["ground_truth"]["relevant"]) for g in indices],
            dtype=np.int64
        )
        yauth = np.asarray(
            [1.0 - float(rows[g]["ground_truth"]["forged"]) for g in indices],
            dtype=np.float32
        )
        yedge = []
        for i, j in pairs:
            a, b = rows[indices[i]], rows[indices[j]]
            related = int(a["event_id"] >= 0 and a["event_id"] == b["event_id"])
            contradiction = int(
                related and a["claimed_timestamp"] // 1800 != b["claimed_timestamp"] // 1800
            )
            yedge.append(1 if contradiction else related)
        examples.append({
            "incident": incident,
            "x": X[indices],
            "edge": edge_index,
            "yrole": yrole,
            "yauth": yauth,
            "yedge": np.asarray(yedge, dtype=np.int64),
        })
    return examples

def collect_validation(model, examples, device):
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for ex in examples:
            x = torch.tensor(ex["x"], dtype=torch.float32, device=device)
            edge = torch.tensor(ex["edge"], dtype=torch.long, device=device)
            _, _, _, logits = model(x, edge)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            ys.extend(ex["yedge"].tolist())
    y = np.asarray(ys, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    ap = float(average_precision_score(y, p))
    auc = float(roc_auc_score(y, p))
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.80, 400):
        pred = (p >= threshold).astype(np.int64)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    return ap, auc, best_t, best_f1, y, p

def main():
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Loading cached train and validation features only...")
    train_examples = load_split(root, "train")
    val_examples = load_split(root, "validation")

    train_y = np.concatenate([ex["yedge"] for ex in train_examples])
    pos = int(train_y.sum())
    neg = int((train_y == 0).sum())
    pos_weight = min(EDGE_POSITIVE_WEIGHT, neg / max(pos, 1))
    print(f"Training edges: {len(train_y)} | positive={pos} | negative={neg}")
    print(f"Weighted edge loss: class weights [1.0, {pos_weight:.3f}]")

    model = GraphWrapper().to(device)
    original_path = root / "models" / "trace_graph.pt"
    state = torch.load(original_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    print("Loaded original graph checkpoint (strict). It will not be overwritten.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)

    best_ap = -1.0
    best_epoch = 0
    stale = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(train_examples)
        losses = []
        for ex in train_examples:
            x = torch.tensor(ex["x"], dtype=torch.float32, device=device)
            edge = torch.tensor(ex["edge"], dtype=torch.long, device=device)
            yrole = torch.tensor(ex["yrole"], dtype=torch.long, device=device)
            yauth = torch.tensor(ex["yauth"], dtype=torch.float32, device=device)
            yedge = torch.tensor(ex["yedge"], dtype=torch.long, device=device)

            _, auth, role_logits, edge_logits = model(x, edge)
            loss_role = F.cross_entropy(role_logits, yrole)
            loss_auth = F.binary_cross_entropy(auth, yauth)
            loss_edge = F.cross_entropy(edge_logits, yedge, weight=class_weights)
            loss = loss_role + 0.5 * loss_auth + loss_edge

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        ap, auc, threshold, val_f1, _, _ = collect_validation(model, val_examples, device)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"train_loss={np.mean(losses):.4f} "
            f"val_AP={ap:.4f} val_ROC_AUC={auc:.4f} "
            f"val_best_F1={val_f1:.4f} threshold={threshold:.4f}"
        )

        if ap > best_ap + 1e-5:
            best_ap = ap
            best_epoch = epoch
            stale = 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without AP improvement.")
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state, strict=True)
    ap, auc, threshold, val_f1, y, p = collect_validation(model, val_examples, device)
    pred = (p >= threshold).astype(np.int64)
    out_model = root / "models" / "trace_graph_balanced.pt"
    torch.save(best_state, out_model)

    report = {
        "model": "graph-only fine-tune with weighted edge cross entropy",
        "source_checkpoint": str(original_path),
        "saved_checkpoint": str(out_model),
        "best_epoch": best_epoch,
        "validation_incidents": len(val_examples),
        "validation_edges": int(len(y)),
        "validation_positive_edges": int(y.sum()),
        "validation_negative_edges": int((y == 0).sum()),
        "validation_average_precision": ap,
        "validation_roc_auc": auc,
        "validation_selected_threshold": threshold,
        "validation_precision": float(precision_score(y, pred, zero_division=0)),
        "validation_recall": float(recall_score(y, pred, zero_division=0)),
        "validation_f1": float(f1_score(y, pred, zero_division=0)),
        "validation_confusion_matrix_TN_FP_FN_TP": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
        "test_data_used": False,
        "other_encoder_checkpoints_modified": False,
    }
    report_path = root / "results" / "trace_graph_balanced_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nGRAPH-ONLY FINE-TUNING COMPLETE")
    print("Saved checkpoint:", out_model)
    print("Saved validation report:", report_path)
    print("Validation AP:", round(ap, 4))
    print("Validation ROC-AUC:", round(auc, 4))
    print("Validation threshold:", round(threshold, 4))
    print("Validation precision/recall/F1:",
          round(report["validation_precision"], 4),
          round(report["validation_recall"], 4),
          round(report["validation_f1"], 4))
    print("Original trace_graph.pt preserved. Test split was not used.")

if __name__ == "__main__":
    main()
