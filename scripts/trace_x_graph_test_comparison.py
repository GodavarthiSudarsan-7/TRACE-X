"""Final untouched test evaluation for original vs balanced TRACE-X graph checkpoints.

Thresholds are selected on validation data and read from saved validation reports.
This script does not train, tune thresholds on test data, or modify checkpoints.
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, roc_auc_score
)

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
        logits = self.edge(torch.cat([a, b, torch.abs(a-b), a*b], dim=1))
        return h2, auth, role, logits

class GraphWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TraceGraphNet()
    def forward(self, *args):
        return self.net(*args)

def load_test(root):
    rows = [json.loads(s) for s in
            (root / "data" / "synthetic" / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if s.strip()]
    with np.load(root / "features" / "evidence_features.npz") as z:
        X = z["test_X"].copy()
    if len(rows) != len(X):
        raise RuntimeError(f"Test row mismatch: {len(rows)} records vs {len(X)} features")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["incident_id"])].append(i)
    return rows, X, groups

def infer(model, rows, X, groups, device):
    ys, probs = [], []
    model.eval()
    with torch.no_grad():
        for incident, indices in groups.items():
            pairs = [(i, j) for i in range(len(indices)) for j in range(len(indices)) if i != j]
            edge_index = torch.tensor(pairs, dtype=torch.long, device=device).T
            x = torch.tensor(X[indices], dtype=torch.float32, device=device)
            _, _, _, logits = model(x, edge_index)
            p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for i, j in pairs:
                a, b = rows[indices[i]], rows[indices[j]]
                related = int(a["event_id"] >= 0 and a["event_id"] == b["event_id"])
                contradiction = int(related and
                                    a["claimed_timestamp"] // 1800 != b["claimed_timestamp"] // 1800)
                ys.append(int(1 if contradiction else related))
            probs.extend(p1.tolist())
    return np.asarray(ys, dtype=int), np.asarray(probs, dtype=float)

def load_threshold(path, key):
    report = json.loads(path.read_text(encoding="utf-8"))
    return float(report[key])

def summarize(name, y, p, threshold):
    pred = (p >= threshold).astype(int)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Validation-selected threshold: {threshold:.4f}")
    print(f"Accuracy : {accuracy_score(y, pred):.4f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y, pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, p):.4f}")
    print(f"Average precision: {average_precision_score(y, p):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:",
          confusion_matrix(y, pred, labels=[0,1]).tolist())
    print(f"Predicted positive edges: {int(pred.sum())}/{len(pred)}")

def main():
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, X, groups = load_test(root)
    y = None

    original = GraphWrapper().to(device)
    original.load_state_dict(torch.load(root / "models" / "trace_graph.pt",
                                        map_location=device, weights_only=True), strict=True)
    y, p_original = infer(original, rows, X, groups, device)

    balanced = GraphWrapper().to(device)
    balanced.load_state_dict(torch.load(root / "models" / "trace_graph_balanced.pt",
                                        map_location=device, weights_only=True), strict=True)
    y2, p_balanced = infer(balanced, rows, X, groups, device)
    if not np.array_equal(y, y2):
        raise RuntimeError("Ground-truth edge labels mismatch between model evaluations.")

    original_threshold = load_threshold(
        root / "results" / "trace_graph_original_validation.json",
        "validation_selected_threshold"
    ) if (root / "results" / "trace_graph_original_validation.json").exists() else 0.079
    balanced_threshold = load_threshold(
        root / "results" / "trace_graph_balanced_validation.json",
        "validation_selected_threshold"
    )

    print("TRACE-X FINAL TEST EVALUATION")
    print("=" * 54)
    print(f"Test incidents: {len(groups)}")
    print(f"Directed edges: {len(y)} | positives: {int(y.sum())} | negatives: {int((y==0).sum())}")
    print("No test-based threshold tuning. Both thresholds were selected on validation.")
    summarize("Original graph checkpoint", y, p_original, original_threshold)
    summarize("Balanced fine-tuned graph checkpoint", y, p_balanced, balanced_threshold)

    result = {
        "test_incidents": len(groups),
        "test_edges": len(y),
        "test_positive_edges": int(y.sum()),
        "test_negative_edges": int((y == 0).sum()),
        "thresholds_selected_on_validation": {
            "original": original_threshold,
            "balanced": balanced_threshold
        },
        "original": {
            "average_precision": float(average_precision_score(y, p_original)),
            "roc_auc": float(roc_auc_score(y, p_original)),
            "precision": float(precision_score(y, p_original >= original_threshold, zero_division=0)),
            "recall": float(recall_score(y, p_original >= original_threshold, zero_division=0)),
            "f1": float(f1_score(y, p_original >= original_threshold, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, p_original >= original_threshold, labels=[0,1]).tolist()
        },
        "balanced": {
            "average_precision": float(average_precision_score(y, p_balanced)),
            "roc_auc": float(roc_auc_score(y, p_balanced)),
            "precision": float(precision_score(y, p_balanced >= balanced_threshold, zero_division=0)),
            "recall": float(recall_score(y, p_balanced >= balanced_threshold, zero_division=0)),
            "f1": float(f1_score(y, p_balanced >= balanced_threshold, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, p_balanced >= balanced_threshold, labels=[0,1]).tolist()
        },
        "thresholds_tuned_on_test": False
    }
    out = root / "results" / "trace_graph_final_test_comparison.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSaved:", out)
    print("This evaluates synthetic benchmark performance, not real-world forensic validity.")

if __name__ == "__main__":
    main()
