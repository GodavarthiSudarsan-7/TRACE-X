"""
TRACE-X standalone inference/demo runner.
Loads existing checkpoints and cached test features. Does NOT train or modify models.
Run from project root:
    python scripts/trace_x_infer.py --incident 9 --split test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


class TextNet(nn.Module):
    def __init__(self, vocab_size, dim=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, 64)
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        z = self.emb(x)
        _, h = self.gru(z)
        feat = torch.tanh(self.proj(torch.cat([h[-2], h[-1]], dim=1)))
        return feat, self.head(feat)


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(64, 64)
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        z = self.conv(x).flatten(1)
        feat = torch.tanh(self.proj(z))
        return feat, self.head(feat)


class ActivityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 32, batch_first=True)
        self.proj = nn.Linear(32, 64)
        self.decoder = nn.Linear(32, 6)
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        seq, h = self.gru(x)
        feat = torch.tanh(self.proj(h[-1]))
        recon = self.decoder(seq)
        return feat, recon, self.head(feat)


class TraceGraphNet(nn.Module):
    def __init__(self, in_dim=193, hidden=96):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Dropout(0.15))
        self.msg = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.auth = nn.Linear(hidden, 1)
        self.role = nn.Linear(hidden, 2)
        self.edge = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--incident", type=int, required=True,
                    help="Incident ID, e.g. 9")
    ap.add_argument("--split", choices=["train", "validation", "test"],
                    default="test")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = root / "models"
    cfg = json.loads((root / "configs.json").read_text(encoding="utf-8"))
    vocab_size = len(cfg["vocab"])

    # Strictly load all four existing checkpoints; no training is performed.
    models = {
        "text_encoder": TextNet(vocab_size),
        "image_encoder": ImageNet(),
        "activity_encoder": ActivityNet(),
        "trace_graph": GraphWrapper(),
    }
    for name, model in models.items():
        state = torch.load(model_dir / f"{name}.pt", map_location=device,
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        print(f"PASS checkpoint: {name}.pt (strict load)")

    data_path = root / "data" / "synthetic" / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in data_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    features_path = root / "features" / "evidence_features.npz"
    with np.load(features_path) as archive:
        features = archive[f"{args.split}_X"].copy()

    if len(rows) != features.shape[0]:
        raise RuntimeError(
            f"Row mismatch: {len(rows)} JSONL records vs {features.shape[0]} feature rows")
    if features.ndim != 2 or features.shape[1] != 193:
        raise RuntimeError(f"Expected feature shape [N,193], got {features.shape}")

    # Original pipeline preserves record order when creating split JSONL and features.
    incident_indices = [i for i, r in enumerate(rows)
                        if int(r["incident_id"]) == args.incident]
    if not incident_indices:
        raise SystemExit(
            f"Incident {args.incident} not found in split '{args.split}'. "
            "Choose an incident ID present in that split.")
    subset = [rows[i] for i in incident_indices]
    x = torch.tensor(features[incident_indices], dtype=torch.float32,
                     device=device)
    n = len(subset)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long, device=device).T

    graph = models["trace_graph"]
    with torch.no_grad():
        _, auth, role_logits, edge_logits = graph(x, edge_index)
        role_probs = torch.softmax(role_logits, dim=1)
        edge_probs = torch.softmax(edge_logits, dim=1)
        auth_np = auth.cpu().numpy()
        role_np = role_probs.cpu().numpy()
        edge_np = edge_probs.cpu().numpy()
        edge_pred = edge_logits.argmax(dim=1).cpu().numpy()

    evidence = []
    for i, row in enumerate(subset):
        evidence.append({
            "evidence_id": row["evidence_id"],
            "claimed_timestamp": int(row["claimed_timestamp"]),
            "authenticity_estimate": round(float(auth_np[i]), 4),
            "relevance_prediction": (
                "relevant" if int(role_np[i].argmax()) == 1 else "distractor"),
            "relevance_confidence": round(float(role_np[i].max()), 4),
            "review_flag": bool(auth_np[i] < 0.5),
            # Ground truth is printed separately for transparent synthetic evaluation.
            "synthetic_ground_truth": {
                "forged": int(row["ground_truth"]["forged"]),
                "relevant": int(row["ground_truth"]["relevant"]),
                "event_order": int(row["ground_truth"]["event_order"]),
            },
        })
    timeline = sorted(evidence, key=lambda e: e["claimed_timestamp"])

    edges = []
    for k, (i, j) in enumerate(pairs):
        if int(edge_pred[k]) == 1:
            edges.append({
                "source": subset[i]["evidence_id"],
                "target": subset[j]["evidence_id"],
                "predicted_relation": "related_or_contradictory_candidate",
                "confidence": round(float(edge_np[k, 1]), 4),
            })

    # Leave-one-evidence-out sensitivity of mean relevance probability.
    def mean_relevance(arr):
        m = len(arr)
        if m < 2:
            return 0.0
        pp = [(i, j) for i in range(m) for j in range(m) if i != j]
        ei = torch.tensor(pp, dtype=torch.long, device=device).T
        with torch.no_grad():
            _, _, logits, _ = graph(torch.tensor(arr, dtype=torch.float32,
                                                 device=device), ei)
            return float(torch.softmax(logits, dim=1)[:, 1].mean().cpu())

    base = mean_relevance(features[incident_indices])
    explanations = []
    for k, row in enumerate(subset):
        reduced = np.delete(features[incident_indices], k, axis=0)
        change = base - mean_relevance(reduced)
        explanations.append({
            "removed_evidence": row["evidence_id"],
            "confidence_change": round(float(change), 6),
        })
    explanations.sort(key=lambda z: abs(z["confidence_change"]), reverse=True)

    report = {
        "project": "TRACE-X",
        "split": args.split,
        "incident_id": args.incident,
        "evidence_count": n,
        "timeline_basis": "claimed timestamps (not learned causal order)",
        "timeline": timeline,
        "predicted_edges": edges,
        "leave_one_out_sensitivity": explanations,
        "limitations": [
            "Synthetic research prototype; not validated for real investigations.",
            "Authenticity estimates are model scores, not proof of authenticity or forgery.",
            "Edge class 1 combines related and contradictory candidates; it does not distinguish them.",
            "Timeline is sorted by claimed timestamps and is not causal reconstruction.",
            "Leave-one-out changes are sensitivity measures, not causal explanations.",
            "Synthetic ground truth is included only for transparent test-set evaluation.",
        ],
    }

    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"incident_{args.incident}_{args.split}_demo.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nTRACE-X INCIDENT DEMO: {args.incident}")
    print(f"Split: {args.split} | Evidence records: {n}")
    print("\nTIMELINE (claimed timestamp order)")
    for item in timeline:
        print(
            f"{item['evidence_id']:18s} "
            f"auth={item['authenticity_estimate']:.3f} "
            f"role={item['relevance_prediction']:10s} "
            f"review={item['review_flag']}"
        )
    print(f"\nPredicted related/contradictory candidate directed edges: {len(edges)}")
    print("\nTop leave-one-out sensitivity items")
    for item in explanations[:5]:
        print(f"{item['removed_evidence']:18s} change={item['confidence_change']:+.6f}")
    print(f"\nSaved report: {out_path}")
    print("\nWARNING: Scores and relationships are synthetic-prototype outputs, not forensic proof.")


if __name__ == "__main__":
    main()
