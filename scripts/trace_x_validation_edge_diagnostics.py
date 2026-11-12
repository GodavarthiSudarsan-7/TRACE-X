"""Evaluate the existing TRACE-X graph checkpoint on validation edges. No training."""
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

def main():
    root = Path(__file__).resolve().parents[1]
    split = "validation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphWrapper().to(device)
    state = torch.load(root / "models" / "trace_graph.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    jsonl = root / "data" / "synthetic" / f"{split}.jsonl"
    rows = [json.loads(s) for s in jsonl.read_text(encoding="utf-8").splitlines() if s.strip()]
    with np.load(root / "features" / "evidence_features.npz") as z:
        X = z[f"{split}_X"].copy()
    if len(rows) != len(X):
        raise RuntimeError(f"Row mismatch: {len(rows)} records vs {len(X)} feature rows")

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["incident_id"])].append(i)

    ys, probs = [], []
    with torch.no_grad():
        for incident, indices in groups.items():
            pairs = [(i, j) for i in range(len(indices)) for j in range(len(indices)) if i != j]
            edge_index = torch.tensor(pairs, dtype=torch.long, device=device).T
            x = torch.tensor(X[indices], dtype=torch.float32, device=device)
            _, _, _, logits = model(x, edge_index)
            p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = []
            for i, j in pairs:
                a, b = rows[indices[i]], rows[indices[j]]
                related = int(a["event_id"] >= 0 and a["event_id"] == b["event_id"])
                contradiction = int(related and
                                    a["claimed_timestamp"] // 1800 != b["claimed_timestamp"] // 1800)
                labels.append(int(1 if contradiction else related))
            ys.extend(labels)
            probs.extend(p1.tolist())

    y = np.asarray(ys, dtype=int)
    p = np.asarray(probs, dtype=float)
    print("TRACE-X VALIDATION EDGE DIAGNOSTICS (existing checkpoint; no training)")
    print("=" * 68)
    print(f"Incidents: {len(groups)} | Directed edges: {len(y)}")
    print(f"Positive labels: {int(y.sum())} | Negative labels: {int((y == 0).sum())}")
    print(f"Mean P(class 1), positive: {p[y == 1].mean():.6f}")
    print(f"Mean P(class 1), negative: {p[y == 0].mean():.6f}")
    print(f"ROC-AUC: {roc_auc_score(y, p):.4f}")
    print(f"Average precision (PR-AUC summary): {average_precision_score(y, p):.4f}")
    print()
    for threshold in [0.50, 0.30, 0.20, 0.15, 0.10, 0.08, 0.05]:
        pred = (p >= threshold).astype(int)
        print(f"threshold={threshold:.2f} positives={int(pred.sum()):4d} "
              f"accuracy={accuracy_score(y,pred):.4f} "
              f"precision={precision_score(y,pred,zero_division=0):.4f} "
              f"recall={recall_score(y,pred,zero_division=0):.4f} "
              f"F1={f1_score(y,pred,zero_division=0):.4f}")
    best_t, best_f1 = None, -1.0
    for threshold in np.linspace(0.001, 0.5, 500):
        pred = (p >= threshold).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(threshold), float(score)
    pred = (p >= best_t).astype(int)
    print()
    print(f"Best validation F1 threshold: {best_t:.4f}")
    print(f"Validation F1 at selected threshold: {best_f1:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:",
          confusion_matrix(y, pred, labels=[0,1]).tolist())
    print("Use this threshold only as a validation-selected candidate; "
          "the final untouched test evaluation must be reported separately.")

if __name__ == "__main__":
    main()
