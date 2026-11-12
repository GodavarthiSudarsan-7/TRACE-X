import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_pairs(records):
    groups = {}
    for i, row in enumerate(records):
        groups.setdefault(str(row["incident_id"]), []).append(i)

    scores, labels = [], []
    for indices in groups.values():
        for pos, i in enumerate(indices):
            for j in indices[pos + 1:]:
                ri, rj = records[i], records[j]
                ei, ej = ri.get("event_id", -1), rj.get("event_id", -1)
                label = int(ei not in (-1, None, "-1") and ei == ej)

                ti = float(ri.get("claimed_timestamp", 0))
                tj = float(rj.get("claimed_timestamp", 0))
                delta = abs(ti - tj)

                # Higher score means more likely to be same-event.
                scores.append(-delta)
                labels.append(label)

    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return float(np.median(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def report(split, y, scores, threshold):
    pred = (scores >= threshold).astype(np.int64)
    return {
        "split": split,
        "threshold_on_negative_absolute_timestamp_delta": float(threshold),
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
    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "synthetic"

    val_scores, val_y = make_pairs(load_jsonl(data / "validation.jsonl"))
    test_scores, test_y = make_pairs(load_jsonl(data / "test.jsonl"))

    threshold = choose_threshold(val_y, val_scores)
    val_report = report("validation", val_y, val_scores, threshold)
    test_report = report("test", test_y, test_scores, threshold)

    output = {
        "baseline": "timestamp-only same-event ranking",
        "score_definition": "negative absolute claimed-timestamp difference",
        "threshold_selection": "validation F1 only; fixed before test evaluation",
        "validation": val_report,
        "test": test_report,
        "limitations": (
            "Synthetic benchmark only. event_id is used only to construct labels. "
            "Timestamp proximity is not proof of event identity or causation."
        ),
    }

    out_path = root / "results" / "trace_timestamp_baseline.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("TRACE-X TIMESTAMP-ONLY BASELINE")
    print("=" * 54)
    print(f"Validation pairs: {len(val_y)} | positives: {int(val_y.sum())}")
    print(f"Test pairs:       {len(test_y)} | positives: {int(test_y.sum())}")
    print(f"Threshold selected on validation: {threshold:.2f} (negative seconds)")
    for item in (val_report, test_report):
        print(f"\n{item['split'].upper()}")
        for key in (
            "accuracy", "precision", "recall", "f1",
            "roc_auc", "average_precision"
        ):
            print(f"{key:18s}: {item[key]:.4f}")
        print(
            "Confusion matrix [TN, FP, FN, TP]: "
            f"{item['confusion_matrix_tn_fp_fn_tp']}"
        )
    print(f"\nSaved: {out_path}")
    print("No model was trained or modified.")
    print("Validation selected the threshold; test was evaluation only.")


if __name__ == "__main__":
    main()
