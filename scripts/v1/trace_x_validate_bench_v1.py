import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_split(name, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["incident_id"]].append(row)

    errors = []
    for incident_id, items in groups.items():
        if len(items) != 11:
            errors.append(f"{name}: incident {incident_id} has {len(items)} records, expected 11")

        event_groups = defaultdict(list)
        distractors = []
        for row in items:
            if row["event_id"] == -1:
                distractors.append(row)
            else:
                event_groups[row["event_id"]].append(row)

            image = np.asarray(row["image"], dtype=np.float32)
            activity = np.asarray(row["activity"], dtype=np.float32)
            if image.shape != (1, 32, 32):
                errors.append(f"{name}: {row['evidence_id']} image shape {image.shape}")
            if activity.shape != (8, 6):
                errors.append(f"{name}: {row['evidence_id']} activity shape {activity.shape}")

        if len(event_groups) != 5:
            errors.append(f"{name}: incident {incident_id} has {len(event_groups)} events, expected 5")
        if len(distractors) != 1:
            errors.append(f"{name}: incident {incident_id} has {len(distractors)} distractors, expected 1")

        for event_id, pair in event_groups.items():
            if len(pair) != 2:
                errors.append(
                    f"{name}: incident {incident_id}, event {event_id} "
                    f"has {len(pair)} evidence records, expected 2"
                )

    return len(groups), errors


def pair_arrays(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["incident_id"]].append(i)

    labels, neg_abs_time_delta = [], []
    for indices in groups.values():
        for p, i in enumerate(indices):
            for j in indices[p + 1:]:
                a, b = rows[i], rows[j]
                same_event = int(
                    a["event_id"] != -1 and a["event_id"] == b["event_id"]
                )
                labels.append(same_event)
                delta = abs(
                    float(a["claimed_timestamp"]) - float(b["claimed_timestamp"])
                )
                neg_abs_time_delta.append(-delta)

    return np.asarray(labels), np.asarray(neg_abs_time_delta)


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "synthetic_v1"

    split_rows = {}
    all_incident_ids = {}
    all_errors = []

    print("TRACE_Bench_v1 STRUCTURAL VALIDATION")
    print("=" * 58)

    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = load_jsonl(path)
        split_rows[split] = rows
        incident_count, errors = validate_split(split, rows)
        all_errors.extend(errors)
        all_incident_ids[split] = set(r["incident_id"] for r in rows)

        print(
            f"{split:10s}: incidents={incident_count:3d}, "
            f"records={len(rows):4d}, structural_errors={len(errors)}"
        )

    # Incident-level split leakage check.
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = all_incident_ids[a] & all_incident_ids[b]
        if overlap:
            all_errors.append(f"Incident leakage between {a} and {b}: {sorted(overlap)[:10]}")

    # Validate manifest against actual files.
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        all_errors.append("manifest.json is missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("benchmark") != "TRACE_Bench_v1":
            all_errors.append("Unexpected benchmark name in manifest")

    # Validation-only timestamp baseline diagnostics.
    val_y, val_score = pair_arrays(split_rows["validation"])
    if len(np.unique(val_y)) == 2:
        auc = float(roc_auc_score(val_y, val_score))
        ap = float(average_precision_score(val_y, val_score))
    else:
        auc, ap = float("nan"), float("nan")

    print("\nVALIDATION TIMESTAMP DIAGNOSTIC (not a model)")
    print("-" * 58)
    print(f"Pairs: {len(val_y)} | same-event: {int(val_y.sum())} | different-event: {int((val_y == 0).sum())}")
    print(f"Timestamp-only ROC-AUC: {auc:.4f}")
    print(f"Timestamp-only Average Precision: {ap:.4f}")

    # Confirm there is meaningful timestamp overlap across pair classes.
    abs_delta = -val_score
    same = abs_delta[val_y == 1]
    different = abs_delta[val_y == 0]
    print(f"Same-event |delta| range:      {same.min():.0f} to {same.max():.0f} seconds")
    print(f"Different-event |delta| range: {different.min():.0f} to {different.max():.0f} seconds")
    ranges_overlap = max(same.min(), different.min()) <= min(same.max(), different.max())
    print(f"Timestamp ranges overlap: {ranges_overlap}")

    print("\nVALIDATION SUMMARY")
    print("-" * 58)
    if all_errors:
        print(f"FAIL: {len(all_errors)} issue(s) found")
        for error in all_errors[:30]:
            print(" -", error)
    else:
        print("PASS: structure, counts, tensor shapes and split separation checks passed.")

    print("\nNo files were modified. No models were trained.")
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
