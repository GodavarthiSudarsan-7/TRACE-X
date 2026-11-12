import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine_similarity_rows(a, b, eps=1e-8):
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, eps)


def main():
    root = Path(__file__).resolve().parents[1]
    records = load_jsonl(root / "data" / "synthetic" / "validation.jsonl")
    cached = np.load(root / "features" / "evidence_features.npz")
    features = cached["validation_X"]

    if len(records) != len(features):
        raise ValueError(
            f"Alignment mismatch: {len(records)} records vs {len(features)} features"
        )

    groups = {}
    for idx, row in enumerate(records):
        groups.setdefault(str(row["incident_id"]), []).append(idx)

    # Feature layout from TRACE-X: text[0:64], image[64:128],
    # activity[128:192], anomaly scalar[192].
    slices = {
        "text": slice(0, 64),
        "image": slice(64, 128),
        "activity": slice(128, 192),
        "combined_192d": slice(0, 192),
    }

    similarities = {name: [] for name in slices}
    labels = []
    time_deltas = []

    for incident_id, indices in groups.items():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                ri, rj = records[i], records[j]
                ei, ej = ri.get("event_id", -1), rj.get("event_id", -1)
                label = int(ei not in (-1, None, "-1") and ei == ej)
                labels.append(label)

                for name, slc in slices.items():
                    a = features[i, slc].astype(np.float64)
                    b = features[j, slc].astype(np.float64)
                    similarities[name].append(
                        float(cosine_similarity_rows(a[None, :], b[None, :])[0])
                    )

                ti = float(ri.get("claimed_timestamp", 0))
                tj = float(rj.get("claimed_timestamp", 0))
                time_deltas.append(abs(ti - tj))

    y = np.asarray(labels, dtype=np.int64)
    print("TRACE-X VALIDATION REPRESENTATION DIAGNOSTICS")
    print("=" * 58)
    print(f"Incidents: {len(groups)} | pairs: {len(y)}")
    print(f"Positive same-event pairs: {int(y.sum())}")
    print(f"Negative pairs: {int((y == 0).sum())}")
    print("\nCosine similarity as a simple same-event score")
    print("-" * 58)

    for name, values in similarities.items():
        scores = np.asarray(values, dtype=np.float64)
        pos_mean = float(scores[y == 1].mean())
        neg_mean = float(scores[y == 0].mean())
        auc = float(roc_auc_score(y, scores))
        ap = float(average_precision_score(y, scores))
        print(
            f"{name:16s} AUC={auc:.4f} AP={ap:.4f} "
            f"mean_same={pos_mean:.4f} mean_different={neg_mean:.4f} "
            f"gap={pos_mean - neg_mean:+.4f}"
        )

    td = np.asarray(time_deltas, dtype=np.float64)
    print("\nAbsolute timestamp difference (seconds)")
    print("-" * 58)
    print(f"Same-event mean:      {td[y == 1].mean():.2f}")
    print(f"Different-event mean: {td[y == 0].mean():.2f}")
    print("Note: timestamp difference is descriptive, not proof of event identity.")

    report = {
        "split": "validation",
        "pairs": int(len(y)),
        "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()),
        "cosine_similarity": {},
        "timestamp_delta_seconds": {
            "same_event_mean": float(td[y == 1].mean()),
            "different_event_mean": float(td[y == 0].mean()),
        },
        "test_split_used": False,
    }
    for name, values in similarities.items():
        scores = np.asarray(values, dtype=np.float64)
        report["cosine_similarity"][name] = {
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "same_event_mean": float(scores[y == 1].mean()),
            "different_event_mean": float(scores[y == 0].mean()),
        }

    output = root / "results" / "trace_representation_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")
    print("No model was trained or modified. Test split was not used.")


if __name__ == "__main__":
    main()
