import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


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
        self.classifier = nn.Sequential(
            nn.Linear(288, 192), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, ta, ia, aa, tb, ib, ab):
        za = self.encoder(ta, ia, aa)
        zb = self.encoder(tb, ib, ab)
        parts = []
        for a, b in zip(za, zb):
            parts.extend([torch.abs(a - b), a * b])
        return self.classifier(torch.cat(parts, dim=1)).squeeze(1)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="TRACE-X V1 demonstration: score evidence-pair relationships."
    )
    parser.add_argument("--incident", type=int, default=None,
                        help="Test incident ID. Default: first incident in test split.")
    parser.add_argument("--top", type=int, default=8,
                        help="Number of highest-scoring pairs to display.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "synthetic_v1"
    feature_dir = root / "features" / "v1"
    model_path = root / "models" / "v1" / "trace_siamese_multimodal_v1.pt"
    validation_path = root / "results" / "v1" / "trace_siamese_multimodal_v1_validation.json"

    for p in (data_dir / "test.jsonl", feature_dir / "trace_bench_v1_tensors.npz",
              feature_dir / "vocab.json", model_path, validation_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    rows = load_jsonl(data_dir / "test.jsonl")
    val_report = json.loads(validation_path.read_text(encoding="utf-8"))
    threshold = float(val_report["validation"]["threshold"])
    vocab = json.loads((feature_dir / "vocab.json").read_text(encoding="utf-8"))

    selected = [r for r in rows if args.incident is None or int(r["incident_id"]) == args.incident]
    if not selected:
        raise ValueError("Incident ID not found in the test split. Choose an ID present in data/synthetic_v1/test.jsonl.")
    incident_id = int(selected[0]["incident_id"])
    selected = [r for r in selected if int(r["incident_id"]) == incident_id]

    # Match the V1 preprocessing: lowercase whitespace tokenization, train vocabulary,
    # 48-token truncation, 32x32 grayscale image, and 8x6 activity sequence.
    text_ids = []
    for row in selected:
        tokens = str(row["text"]).lower().strip().split()[:48]
        ids = [vocab.get(token, 1) for token in tokens]
        text_ids.append(ids + [0] * (48 - len(ids)))

    text = torch.tensor(np.asarray(text_ids), dtype=torch.long)
    images = torch.tensor(np.asarray([r["image"] for r in selected], dtype=np.float32))
    activities = torch.tensor(np.asarray([r["activity"] for r in selected], dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SiameseTraceNet(int(checkpoint["vocab_size"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    candidates = []
    with torch.no_grad():
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                logit = model(
                    text[i:i+1].to(device), images[i:i+1].to(device),
                    activities[i:i+1].to(device),
                    text[j:j+1].to(device), images[j:j+1].to(device),
                    activities[j:j+1].to(device),
                )
                probability = float(torch.sigmoid(logit).item())
                candidates.append((probability, i, j))

    candidates.sort(reverse=True)
    print("\nTRACE-X | SYNTHETIC INCIDENT INVESTIGATION DEMO")
    print("=" * 62)
    print(f"Test incident: {incident_id}")
    print(f"Evidence records: {len(selected)}")
    print(f"Candidate pairs scored: {len(candidates)}")
    print(f"Decision threshold (selected on validation): {threshold:.6f}")
    print("Model inputs: text + image + activity; timestamp is not an input.")
    print("\nTOP EVIDENCE RELATIONSHIP CANDIDATES")
    print("-" * 62)

    for rank, (prob, i, j) in enumerate(candidates[:max(1, args.top)], 1):
        label = "RELATED CANDIDATE" if prob >= threshold else "NOT LINKED"
        a, b = selected[i], selected[j]
        print(f"\n{rank}. {label} | model score={prob:.4f}")
        print(f"   {a['evidence_id']}: {a['text']}")
        print(f"   {b['evidence_id']}: {b['text']}")

    print("\nINTERPRETATION")
    print("- Scores rank candidate relationships; they are not proof.")
    print("- This is a synthetic research demonstration, not a forensic tool.")
    print("- Human review and independent real-world validation are required.")
    print("- Ground-truth event IDs were not supplied to the model for prediction.")
    print("\nNo training performed. Checkpoint and dataset were not modified.")


if __name__ == "__main__":
    main()
