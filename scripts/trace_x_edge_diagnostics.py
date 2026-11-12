"""Evaluate TRACE-X graph edge predictions on a split without training."""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def main():
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphWrapper().to(device)
    state = torch.load(root / "models" / "trace_graph.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = [json.loads(line) for line in
            (root / "data" / "synthetic" / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    with np.load(root / "features" / "evidence_features.npz") as z:
        X = z["test_X"].copy()
    if len(rows) != len(X):
        raise RuntimeError(f"Row mismatch: {len(rows)} records, {len(X)} feature rows")

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[int(row["incident_id"])].append(i)

    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for incident, indices in groups.items():
            # Match the original training label rule exactly.
            pairs = [(i, j) for i in range(len(indices)) for j in range(len(indices)) if i != j]
            edge_index = torch.tensor(pairs, dtype=torch.long, device=device).T
            x = torch.tensor(X[indices], dtype=torch.float32, device=device)
            _, _, _, logits = model(x, edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            labels = []
            for i, j in pairs:
                a, b = rows[indices[i]], rows[indices[j]]
                related = int(a["event_id"] >= 0 and a["event_id"] == b["event_id"])
                contradiction = int(related and
                                    a["claimed_timestamp"] // 1800 != b["claimed_timestamp"] // 1800)
                labels.append(1 if contradiction else related)

            all_y.extend(labels)
            all_prob.extend(probs.tolist())
            all_p.extend((probs >= 0.5).astype(int).tolist())

    y = np.asarray(all_y, dtype=int)
    p = np.asarray(all_p, dtype=int)
    prob = np.asarray(all_prob, dtype=float)

    print("TRACE-X TEST EDGE DIAGNOSTICS (no training)")
    print("=" * 48)
    print(f"Incidents: {len(groups)}")
    print(f"Directed edges: {len(y)}")
    print(f"Positive ground-truth edges: {int(y.sum())}")
    print(f"Negative ground-truth edges: {int((y == 0).sum())}")
    print(f"Positive prediction rate at 0.50: {p.mean():.4f}")
    print(f"Mean P(class 1) on positive edges: {prob[y == 1].mean():.6f}")
    print(f"Mean P(class 1) on negative edges: {prob[y == 0].mean():.6f}")
    print()
    print("At argmax / threshold 0.50")
    print(f"Accuracy : {accuracy_score(y, p):.4f}")
    print(f"Precision: {precision_score(y, p, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, p, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y, p, zero_division=0):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, p, labels=[0, 1]).tolist())
    print()
    print("Threshold sweep (diagnostic only; do not select threshold on test data):")
    for t in [0.50, 0.40, 0.30, 0.20, 0.10, 0.05]:
        pred = (prob >= t).astype(int)
        print(f"threshold={t:.2f} predicted_positive={int(pred.sum()):4d} "
              f"precision={precision_score(y, pred, zero_division=0):.4f} "
              f"recall={recall_score(y, pred, zero_division=0):.4f} "
              f"F1={f1_score(y, pred, zero_division=0):.4f}")
    print()
    print("Interpretation: this is synthetic test-set diagnosis, not real-world validation.")

if __name__ == "__main__":
    main()
