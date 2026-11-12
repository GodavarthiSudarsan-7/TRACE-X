import json
from pathlib import Path
import random

import numpy as np

SEED = 20261111
N_INCIDENTS = 500
EVENTS_PER_INCIDENT = 5
EVIDENCE_PER_EVENT = 2
IMAGE_SIZE = 32

# Stable vocabularies make event content interpretable and reproducible.
EVENT_TYPES = [
    ("delivery", "package", "loading bay"),
    ("meeting", "document", "conference room"),
    ("access", "door", "north entrance"),
    ("vehicle", "car", "parking area"),
    ("system", "server", "control room"),
    ("camera", "recording", "security desk"),
    ("call", "phone", "reception"),
    ("inspection", "equipment", "storage area"),
]
ACTIONS = [
    "arrived", "departed", "was recorded", "was transferred",
    "was accessed", "was inspected", "was reported", "was updated"
]
SOURCES = [
    "entry log", "camera record", "operator note",
    "system event", "activity record", "dispatch note"
]
TEXT_TEMPLATES = [
    "{source} reports that the {object_name} {action} at the {location}.",
    "At the {location}, the {object_name} {action}; source: {source}.",
    "Record from {source}: {action} involving {object_name} near the {location}.",
    "The {location} record notes the {object_name} {action}, according to {source}.",
]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_image(event_rng, forged=False):
    """A shared event-level visual signature plus evidence-specific variation."""
    img = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]

    cx, cy, radius, rect_w, rect_h = event_rng
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[0][circle] = 0.72

    x0 = max(0, min(IMAGE_SIZE - rect_w, cx - rect_w // 2))
    y0 = max(0, min(IMAGE_SIZE - rect_h, cy + radius // 2))
    img[0, y0:y0 + rect_h, x0:x0 + rect_w] = 0.38

    # A small shared marker helps same-event evidence retain identity.
    marker_x = max(1, min(IMAGE_SIZE - 3, cx + 3))
    marker_y = max(1, min(IMAGE_SIZE - 3, cy - 3))
    img[0, marker_y:marker_y + 2, marker_x:marker_x + 2] = 1.0

    img += np.random.normal(0, 0.035, img.shape).astype(np.float32)

    if forged:
        # Controlled alteration: local occlusion plus a small spatial shift.
        img[0, 10:15, 10:17] *= 0.15
        img[0] = np.roll(img[0], shift=2, axis=1)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_activity(profile, rng, forged=False):
    """Event-level activity signature with per-record measurement noise."""
    sequence = profile[None, :] + rng.normal(0, 0.16, size=(8, 6))
    sequence[:, 5] += np.linspace(-0.08, 0.08, 8)
    if forged:
        sequence[:, 1] += rng.normal(0.9, 0.12, size=8)
        sequence[3:6, 4] -= 0.8
    return sequence.astype(np.float32)


def build_dataset():
    py_rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    # Split by incident, not by evidence, to prevent incident-level leakage.
    incident_ids = np_rng.permutation(N_INCIDENTS).tolist()
    train_ids = set(incident_ids[:350])
    val_ids = set(incident_ids[350:425])
    test_ids = set(incident_ids[425:])

    splits = {"train": [], "validation": [], "test": []}

    for incident_id in range(N_INCIDENTS):
        if incident_id in train_ids:
            split = "train"
        elif incident_id in val_ids:
            split = "validation"
        else:
            split = "test"

        incident_rng = np.random.default_rng(SEED + incident_id * 1009)
        incident_rows = []

        base_time = 1_800_000_000 + incident_id * 100_000
        for event_id in range(EVENTS_PER_INCIDENT):
            category, object_name, location = py_rng.choice(EVENT_TYPES)
            action = py_rng.choice(ACTIONS)
            source = py_rng.choice(SOURCES)

            # Shared event-level visual and activity characteristics.
            cx = int(incident_rng.integers(7, 25))
            cy = int(incident_rng.integers(7, 25))
            radius = int(incident_rng.integers(3, 7))
            rect_w = int(incident_rng.integers(5, 11))
            rect_h = int(incident_rng.integers(3, 8))
            visual_signature = (cx, cy, radius, rect_w, rect_h)
            activity_profile = incident_rng.normal(0, 1, size=6).astype(np.float32)
            activity_profile[0] += event_id * 0.45

            event_time = base_time + event_id * 300
            event_is_forged = bool(incident_rng.random() < 0.25)
            forged_variant = int(incident_rng.integers(0, EVIDENCE_PER_EVENT))

            for variant in range(EVIDENCE_PER_EVENT):
                forged = bool(event_is_forged and variant == forged_variant)

                # Same-event evidence can be far apart in time. Different events
                # are only five minutes apart, so their time ranges can overlap.
                timestamp_jitter = int(incident_rng.integers(-1200, 1201))
                claimed_timestamp = int(event_time + timestamp_jitter)

                template = py_rng.choice(TEXT_TEMPLATES)
                text = template.format(
                    source=source,
                    object_name=object_name,
                    action=action,
                    location=location,
                )
                if variant == 1:
                    text += " Secondary observation confirms the same event context."
                if forged:
                    text = text.replace(object_name, py_rng.choice(
                        ["unknown item", "unverified object", "unidentified record"]
                    ))
                    text += " Alteration indicator: inconsistent detail."

                # Each record gets its own RNG for reproducibility.
                record_rng = np.random.default_rng(
                    SEED + incident_id * 100_000 + event_id * 100 + variant
                )
                image = make_image(visual_signature, forged=forged)
                activity = make_activity(activity_profile, record_rng, forged=forged)

                incident_rows.append({
                    "evidence_id": f"I{incident_id:04d}_E{event_id}_V{variant}",
                    "incident_id": incident_id,
                    "event_id": event_id,
                    "variant": variant,
                    "text": text,
                    "claimed_timestamp": claimed_timestamp,
                    "image": image.tolist(),
                    "activity": activity.tolist(),
                    "ground_truth": {
                        "forged": int(forged),
                        "relevant": 1,
                        "event_order": event_id,
                        "risk": int(forged),
                    },
                })

        # Distractor shares the incident but has no event identity.
        distractor_rng = np.random.default_rng(SEED + incident_id * 77_777 + 9)
        distractor_activity = distractor_rng.normal(0, 1, size=(8, 6)).astype(np.float32)
        distractor_image = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        distractor_image[0, 2:7, 24:30] = 0.65
        distractor_image += distractor_rng.normal(
            0, 0.05, distractor_image.shape
        ).astype(np.float32)

        incident_rows.append({
            "evidence_id": f"I{incident_id:04d}_DIST",
            "incident_id": incident_id,
            "event_id": -1,
            "variant": 0,
            "text": "Routine unrelated archive maintenance notification.",
            "claimed_timestamp": int(base_time + 900),
            "image": np.clip(distractor_image, 0, 1).tolist(),
            "activity": distractor_activity.tolist(),
            "ground_truth": {
                "forged": 0,
                "relevant": 0,
                "event_order": -1,
                "risk": 0,
            },
        })

        splits[split].extend(incident_rows)

    return splits


def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data" / "synthetic_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = build_dataset()
    manifest = {
        "benchmark": "TRACE_Bench_v1",
        "seed": SEED,
        "incidents_total": N_INCIDENTS,
        "events_per_incident": EVENTS_PER_INCIDENT,
        "evidence_per_event": EVIDENCE_PER_EVENT,
        "records_per_incident": EVENTS_PER_INCIDENT * EVIDENCE_PER_EVENT + 1,
        "split_incidents": {"train": 350, "validation": 75, "test": 75},
        "split_records": {},
        "design": {
            "same_event_text": "Shared event semantics across wording variants",
            "same_event_image": "Shared event-level geometry with per-record noise",
            "same_event_activity": "Shared event-level activity profile with per-record noise",
            "timestamps": "Same-event jitter up to 20 minutes; event centers 5 minutes apart",
            "forgery": "Controlled text, image, and activity modifications",
            "distractors": "One unrelated evidence item per incident",
            "split_policy": "Incident-disjoint train/validation/test",
        },
        "caveat": (
            "Synthetic benchmark for method development only; not a real-world "
            "forensic dataset or evidence of forensic validity."
        ),
    }

    for split_name, rows in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        write_jsonl(path, rows)
        incident_count = len({row["incident_id"] for row in rows})
        manifest["split_records"][split_name] = {
            "incidents": incident_count,
            "records": len(rows),
            "forged_records": sum(r["ground_truth"]["forged"] for r in rows),
            "distractors": sum(r["event_id"] == -1 for r in rows),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("TRACE_Bench_v1 DATASET GENERATED")
    print("=" * 52)
    for split_name, details in manifest["split_records"].items():
        print(
            f"{split_name:10s}: incidents={details['incidents']:3d}, "
            f"records={details['records']:4d}, "
            f"forged={details['forged_records']:3d}, "
            f"distractors={details['distractors']:3d}"
        )
    print(f"\nOutput directory: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print("Original data, features, models and results were not modified.")
    print("No model was trained.")
    print("Review the generated benchmark before feature extraction or training.")


if __name__ == "__main__":
    main()
