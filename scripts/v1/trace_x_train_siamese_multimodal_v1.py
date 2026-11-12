import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve
)

SEED = 20261111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_bundle(npz, split):
    return {
        "text": torch.from_numpy(npz[f"{split}_text_ids"].astype(np.int64)),
        "image": torch.from_numpy(npz[f"{split}_images"].astype(np.float32)),
        "activity": torch.from_numpy(npz[f"{split}_activities"].astype(np.float32)),
        "incident": npz[f"{split}_incident_ids"].astype(np.int64),
        "event": npz[f"{split}_event_ids"].astype(np.int64),
    }


def make_pairs(bundle):
    groups = {}
    for idx, incident_id in enumerate(bundle["incident"].tolist()):
        groups.setdefault(int(incident_id), []).append(idx)

    pairs = []
    labels = []
    for indices in groups.values():
        for p, i in enumerate(indices):
            for j in indices[p + 1:]:
                ei, ej = int(bundle["event"][i]), int(bundle["event"][j])
                label = int(ei != -1 and ei == ej)
                pairs.append((i, j))
                labels.append(label)
    return np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=np.float32)


class PairDataset(Dataset):
    def __init__(self, bundle, pairs, labels):
        self.bundle = bundle
        self.pairs = torch.from_numpy(pairs)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx].tolist()
        b = self.bundle
        return (
            b["text"][i], b["image"][i], b["activity"][i],
            b["text"][j], b["image"][j], b["activity"][j],
            self.labels[idx]
        )


class ModalityEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.text_gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.text_projection = nn.Sequential(nn.Linear(64, 48), nn.ReLU())

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(48, 48), nn.ReLU()
        )

        self.activity_gru = nn.GRU(6, 32, batch_first=True)
        self.activity_projection = nn.Sequential(nn.Linear(32, 48), nn.ReLU())

    def forward(self, text, image, activity):
        _, ht = self.text_gru(self.text_embedding(text))
        text_z = self.text_projection(torch.cat([ht[-2], ht[-1]], dim=1))
        image_z = self.image_encoder(image)
        _, ha = self.activity_gru(activity)
        activity_z = self.activity_projection(ha[-1])
        return text_z, image_z, activity_z


class SiameseTraceNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ModalityEncoder(vocab_size)
        # Each modality contributes |a-b| and a*b (48+48 dimensions).
        pair_dim = 3 * 48 * 2
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, 192), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 1)
        )

    def forward(self, ta, ia, aa, tb, ib, ab):
        za = self.encoder(ta, ia, aa)
        zb = self.encoder(tb, ib, ab)
        pair_parts = []
        for a, b in zip(za, zb):
            pair_parts.extend([torch.abs(a - b), a * b])
        return self.classifier(torch.cat(pair_parts, dim=1)).squeeze(1)


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate(y, probs):
    threshold = choose_threshold(y, probs)
    pred = (probs >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel().tolist(),
        "predicted_positive": int(pred.sum()),
        "total_pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()),
    }


def predict(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            ta, ia, aa, tb, ib, ab, y = batch
            logits = model(
                ta.to(device), ia.to(device), aa.to(device),
                tb.to(device), ib.to(device), ab.to(device)
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_probs)


def main():
    root = Path(__file__).resolve().parents[2]
    feature_dir = root / "features" / "v1"
    model_dir = root / "models" / "v1"
    result_dir = root / "results" / "v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((feature_dir / "vocab.json").read_text(encoding="utf-8"))
    npz = np.load(feature_dir / "trace_bench_v1_tensors.npz")

    train_bundle = load_bundle(npz, "train")
    val_bundle = load_bundle(npz, "validation")
    train_pairs, train_y = make_pairs(train_bundle)
    val_pairs, val_y = make_pairs(val_bundle)

    pos = np.flatnonzero(train_y == 1)
    neg = np.flatnonzero(train_y == 0)
    rng = np.random.default_rng(SEED)
    sampled_neg = rng.choice(neg, size=min(len(neg), 4 * len(pos)), replace=False)
    sampled = np.concatenate([pos, sampled_neg])
    rng.shuffle(sampled)

    train_ds = PairDataset(train_bundle, train_pairs[sampled], train_y[sampled])
    val_ds = PairDataset(val_bundle, val_pairs, val_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseTraceNet(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_ap = -1.0
    best_state = None
    best_metrics = None
    patience, stale, max_epochs = 7, 0, 35

    print("TRACE_Bench_v1 SIAMESE MULTIMODAL RELATIONSHIP TRAINING")
    print("=" * 66)
    print(f"Train pairs: {len(train_y)} | positives={int(train_y.sum())} | negatives={int((train_y == 0).sum())}")
    print(f"Balanced training sample: {len(sampled)} pairs (positive:negative <= 1:4)")
    print(f"Validation pairs: {len(val_y)} | positives={int(val_y.sum())} | negatives={int((val_y == 0).sum())}")
    print(f"Device: {device}")
    print("Timestamp is deliberately excluded from model inputs.")
    print("Test split is not loaded or used.")

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            ta, ia, aa, tb, ib, ab, y = batch
            ta, ia, aa = ta.to(device), ia.to(device), aa.to(device)
            tb, ib, ab = tb.to(device), ib.to(device), ab.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(ta, ia, aa, tb, ib, ab)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        y_true, val_probs = predict(model, val_loader, device)
        ap = float(average_precision_score(y_true, val_probs))
        auc = float(roc_auc_score(y_true, val_probs))
        metrics = evaluate(y_true, val_probs)
        print(
            f"Epoch {epoch:02d}/{max_epochs} loss={np.mean(losses):.4f} "
            f"val_AP={ap:.4f} val_ROC_AUC={auc:.4f} "
            f"val_F1={metrics['f1']:.4f} threshold={metrics['threshold']:.4f}"
        )

        if ap > best_ap + 1e-6:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping after {patience} epochs without AP improvement.")
                break

    if best_state is None:
        raise RuntimeError("No best checkpoint was produced.")

    model.load_state_dict(best_state)
    checkpoint = model_dir / "trace_siamese_multimodal_v1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": len(vocab),
        "seed": SEED,
        "modalities": ["text", "image", "activity"],
        "timestamp_used": False,
        "pair_features": "per modality: absolute embedding difference and elementwise product",
    }, checkpoint)

    report = {
        "benchmark": "TRACE_Bench_v1",
        "model": "SiameseTraceNet",
        "validation": best_metrics,
        "selection_metric": "validation average precision",
        "timestamp_used_as_input": False,
        "test_used_for_training_or_selection": False,
        "original_models_modified": False,
        "limitations": (
            "Synthetic benchmark only. event_id is used only to form pair labels, "
            "not as model input. Scores do not establish real-world forensic validity."
        ),
    }
    report_path = result_dir / "trace_siamese_multimodal_v1_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTRAINING COMPLETE")
    print(f"Checkpoint: {checkpoint}")
    print(f"Validation report: {report_path}")
    print(f"Best validation AP: {best_metrics['average_precision']:.4f}")
    print(f"Validation ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"Validation precision/recall/F1: {best_metrics['precision']:.4f} / "
          f"{best_metrics['recall']:.4f} / {best_metrics['f1']:.4f}")
    print("Original checkpoints and v0 dataset were not modified.")


if __name__ == "__main__":
    main()
