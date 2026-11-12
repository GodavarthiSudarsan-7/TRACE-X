import json
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 20261111
MAX_TEXT_LEN = 48
MAX_VOCAB = 5000


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text):
    return str(text).lower().strip().split()


def build_vocab(train_rows):
    counts = Counter()
    for row in train_rows:
        counts.update(tokenize(row["text"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counts.most_common(MAX_VOCAB - 2):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_rows(rows, vocab):
    text_ids = np.zeros((len(rows), MAX_TEXT_LEN), dtype=np.int64)
    images = np.zeros((len(rows), 1, 32, 32), dtype=np.float32)
    activities = np.zeros((len(rows), 8, 6), dtype=np.float32)
    forged = np.zeros(len(rows), dtype=np.int64)
    relevant = np.zeros(len(rows), dtype=np.int64)
    event_order = np.full(len(rows), -1, dtype=np.int64)
    timestamps = np.zeros(len(rows), dtype=np.int64)
    incident_ids = np.zeros(len(rows), dtype=np.int64)
    event_ids = np.full(len(rows), -1, dtype=np.int64)
    evidence_ids = []

    for i, row in enumerate(rows):
        tokens = tokenize(row["text"])[:MAX_TEXT_LEN]
        ids = [vocab.get(token, 1) for token in tokens]
        text_ids[i, :len(ids)] = ids

        images[i] = np.asarray(row["image"], dtype=np.float32)
        activities[i] = np.asarray(row["activity"], dtype=np.float32)
        forged[i] = int(row["ground_truth"]["forged"])
        relevant[i] = int(row["ground_truth"]["relevant"])
        event_order[i] = int(row["ground_truth"]["event_order"])
        timestamps[i] = int(row["claimed_timestamp"])
        incident_ids[i] = int(row["incident_id"])
        event_ids[i] = int(row["event_id"])
        evidence_ids.append(str(row["evidence_id"]))

    return {
        "text_ids": text_ids,
        "images": images,
        "activities": activities,
        "forged": forged,
        "relevant": relevant,
        "event_order": event_order,
        "timestamps": timestamps,
        "incident_ids": incident_ids,
        "event_ids": event_ids,
        "evidence_ids": np.asarray(evidence_ids, dtype=str),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "synthetic_v1"
    out_dir = root / "features" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "validation.jsonl")
    test_rows = load_jsonl(data_dir / "test.jsonl")

    # Vocabulary is fitted on training text only.
    vocab = build_vocab(train_rows)
    encoded = {
        "train": encode_rows(train_rows, vocab),
        "validation": encode_rows(val_rows, vocab),
        "test": encode_rows(test_rows, vocab),
    }

    output_path = out_dir / "trace_bench_v1_tensors.npz"
    np.savez_compressed(
        output_path,
        **{
            f"{split}_{key}": value
            for split, bundle in encoded.items()
            for key, value in bundle.items()
        },
    )

    vocab_path = out_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    summary = {
        "benchmark": "TRACE_Bench_v1",
        "seed": SEED,
        "vocabulary_size": len(vocab),
        "max_text_length": MAX_TEXT_LEN,
        "vocab_fit_on": "train only",
        "splits": {},
        "feature_shapes": {},
        "test_used_for_fitting": False,
        "models_trained": False,
        "original_files_modified": False,
    }
    for split, bundle in encoded.items():
        summary["splits"][split] = {
            "records": int(len(bundle["text_ids"])),
            "forged": int(bundle["forged"].sum()),
            "relevant": int(bundle["relevant"].sum()),
            "distractors": int((bundle["event_ids"] == -1).sum()),
            "unique_incidents": int(len(np.unique(bundle["incident_ids"]))),
        }
        summary["feature_shapes"][split] = {
            key: list(value.shape)
            for key, value in bundle.items()
            if key not in ("evidence_ids",)
        }

    summary_path = out_dir / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("TRACE_Bench_v1 PREPROCESSING COMPLETE")
    print("=" * 56)
    print(f"Vocabulary size (train only): {len(vocab)}")
    for split, info in summary["splits"].items():
        print(
            f"{split:10s}: records={info['records']:4d}, "
            f"forged={info['forged']:3d}, distractors={info['distractors']:3d}, "
            f"incidents={info['unique_incidents']:3d}"
        )
        print(f"  text={encoded[split]['text_ids'].shape}, "
              f"images={encoded[split]['images'].shape}, "
              f"activities={encoded[split]['activities'].shape}")
    print(f"\nSaved tensors: {output_path}")
    print(f"Saved vocabulary: {vocab_path}")
    print(f"Saved summary: {summary_path}")
    print("No model was trained. No existing model or dataset was modified.")
    print("event_ids are retained only as labels/metadata for relationship training.")


if __name__ == "__main__":
    main()
