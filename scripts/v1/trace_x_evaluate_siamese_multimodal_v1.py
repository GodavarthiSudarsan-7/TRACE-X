import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve,
)

# Independent V1 test evaluation.
# Does not train, alter, or overwrite any model checkpoint.
# The test split is used only for final evaluation.

def load_bundle(npz, split):
    return {
        "text": torch.from_numpy(npz[f"{split}_text_ids"].astype(np.int64)),
        "image": torch.from_numpy(npz[f"{split}_images"].astype(np.float32)),
        "activity": torch.from_numpy(npz[f"{split}_activities"].astype(np.float32)),
        "incident": npz[f"{split}_incident_ids"].astype(np.int64),
        "event": npz[f"{split}_event_ids"].astype(np.int64),
        "timestamps": npz[f"{split}_timestamps"].astype(np.float64),
    }


def make_pairs(bundle):
    groups = {}
    for idx, incident_id in enumerate(bundle["incident"].tolist()):
        groups.setdefault(int(incident_id), []).append(idx)

    pairs, labels, time_scores = [], [], []
    for indices in groups.values():
        for p, i in enumerate(indices):
            for j in indices[p + 1:]:
                ei, ej = int(bundle["event"][i]), int(bundle["event"][j])
                labels.append(int(ei != -1 and ei == ej))
                pairs.append((i, j))
                # Higher score means more likely to be same-event.
                time_scores.append(-abs(
                    float(bundle["timestamps"][i]) -
                    float(bundle["timestamps"][j])
                ))
    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(time_scores, dtype=np.float64),
    )


class PairDataset(Dataset):
    def __init__(self, bundle, pairs):
        self.bundle = bundle
        self.pairs = torch.from_numpy(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx].tolist()
        b = self.bundle
        return (
            b["text"][i], b["image"][i], b["activity"][i],
            b["text"][j], b["image"][j], b["activity"][j],
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
            nn.Linear(48, 48), nn.ReLU(),
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
        pair_dim = 3 * 48 * 2
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, 192), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, ta, ia, aa, tb, ib, ab):
        za = self.encoder(ta, ia, aa)
        zb = self.encoder(tb, ib, ab)
        pair_parts = []
        for a, b in zip(za, zb):
            pair_parts.extend([torch.abs(a - b), a * b])
        return self.classifier(torch.cat(pair_parts, dim=1)).squeeze(1)


def predict(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            ta, ia, aa, tb, ib, ab = batch
            logits = model(
                ta.to(device), ia.to(device), aa.to(device),
                tb.to(device), ib.to(device), ab.to(device),
            )
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


def choose_f1_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def metrics_at_threshold(y, scores, threshold):
    pred = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel().tolist(),
        "predicted_positive": int(pred.sum()),
        "total_pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    feature_dir = root / "features" / "v1"
    model_path = root / "models" / "v1" / "trace_siamese_multimodal_v1.pt"
    validation_path = root / "results" / "v1" / "trace_siamese_multimodal_v1_validation.json"
    output_path = root / "results" / "v1" / "trace_siamese_multimodal_v1_test.json"

    for path in (feature_dir / "vocab.json",
                 feature_dir / "trace_bench_v1_tensors.npz",
                 model_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    vocab = json.loads((feature_dir / "vocab.json").read_text(encoding="utf-8"))
    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    multimodal_threshold = float(validation_report["validation"]["threshold"])

    # Load only validation labels/timestamps for baseline threshold selection,
    # and test tensors for final evaluation. No training split is loaded.
    with np.load(feature_dir / "trace_bench_v1_tensors.npz") as npz:
        val_bundle = load_bundle(npz, "validation")
        test_bundle = load_bundle(npz, "test")

    val_pairs, val_y, val_time_scores = make_pairs(val_bundle)
    test_pairs, test_y, test_time_scores = make_pairs(test_bundle)

    # Timestamp-only baseline threshold is selected on validation, then frozen.
    timestamp_threshold = choose_f1_threshold(val_y, val_time_scores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SiameseTraceNet(vocab_size=int(checkpoint["vocab_size"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    test_ds = PairDataset(test_bundle, test_pairs)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    test_probs = predict(model, test_loader, device)

    report = {
        "benchmark": "TRACE_Bench_v1",
        "evaluation": "held-out test split; no retraining or threshold tuning on test",
        "test_incidents": int(len(np.unique(test_bundle["incident"]))),
        "test_pairs": int(len(test_y)),
        "positive_pairs": int(test_y.sum()),
        "negative_pairs": int((test_y == 0).sum()),
        "siamese_multimodal": metrics_at_threshold(
            test_y, test_probs, multimodal_threshold
        ),
        "timestamp_only_baseline": {
            "threshold_selected_on_validation": timestamp_threshold,
            **metrics_at_threshold(test_y, test_time_scores, timestamp_threshold),
        },
        "threshold_selection": {
            "siamese_threshold_source": "saved validation report",
            "timestamp_threshold_source": "validation split only",
            "test_used_for_threshold_selection": False,
        },
        "model_checkpoint_modified": False,
        "limitations": (
            "Synthetic benchmark only. event_id is used only to form evaluation "
            "pair labels, not as a model input. Results do not establish "
            "real-world forensic validity."
        ),
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("TRACE_Bench_v1 HELD-OUT TEST EVALUATION")
    print("=" * 52)
    print(f"Device: {device}")
    print(f"Test incidents: {report['test_incidents']}")
    print(f"Test pairs: {len(test_y)} | positives={int(test_y.sum())} | negatives={int((test_y == 0).sum())}")
    print("\nSIAMESE MULTIMODAL (validation-selected threshold)")
    for key, value in report["siamese_multimodal"].items():
        print(f"{key}: {value}")
    print("\nTIMESTAMP-ONLY BASELINE (threshold selected on validation)")
    for key, value in report["timestamp_only_baseline"].items():
        print(f"{key}: {value}")
    print(f"\nSaved report: {output_path}")
    print("No training performed. Checkpoint and source datasets were not modified.")


if __name__ == "__main__":
    main()
