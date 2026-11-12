import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

def dt(value):
    try:
        return datetime.fromisoformat(str(value).strip())
    except Exception:
        return None

def load_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def correlate(rows, report):
    # Preserve the report's flagged row numbers: they refer to CSV data rows (header excluded).
    flagged_by_row = {int(e["row"]): e for e in report.get("flagged_events", [])}
    events = []
    for row_no, row in enumerate(rows, start=1):
        if row_no not in flagged_by_row:
            continue
        ev = dict(flagged_by_row[row_no])
        ev["parsed_time"] = dt(ev.get("timestamp"))
        ev["links"] = []
        events.append(ev)

    events.sort(key=lambda e: (e["parsed_time"] or datetime.max, e["row"]))

    # Correlate only with explicit shared entities and compatible time windows.
    for i, left in enumerate(events):
        for right in events[i + 1:]:
            if not left["parsed_time"] or not right["parsed_time"]:
                continue
            delta = (right["parsed_time"] - left["parsed_time"]).total_seconds()
            if delta < 0 or delta > 900:
                continue

            shared = []
            for field in ("user", "source_ip", "host"):
                a = str(left.get(field, "")).strip()
                b = str(right.get(field, "")).strip()
                if a and a != "-" and a == b:
                    shared.append(field)

            if not shared:
                continue

            # A link is a candidate association, not proof of causation.
            left["links"].append({
                "to_row": right["row"],
                "shared_entities": shared,
                "time_gap_seconds": int(delta),
                "relationship": "candidate_temporal_entity_correlation"
            })

    # Build timeline entries with source-backed evidence and links.
    timeline = []
    for index, ev in enumerate(events, start=1):
        timeline.append({
            "sequence": index,
            "row": ev["row"],
            "timestamp": ev["timestamp"],
            "event": ev["event_type"],
            "severity": ev["severity"],
            "user": ev.get("user"),
            "source_ip": ev.get("source_ip"),
            "host": ev.get("host"),
            "message": ev.get("message"),
            "detection_reasons": ev.get("reasons", []),
            "candidate_links": ev["links"]
        })

    # Summarize only what the event data supports.
    edge_count = sum(len(e["links"]) for e in events)
    shared_entity_edges = defaultdict(int)
    for ev in events:
        for link in ev["links"]:
            for entity in link["shared_entities"]:
                shared_entity_edges[entity] += 1

    return {
        "tool": "TRACE-X Temporal Evidence Correlator",
        "records_in_original_log": len(rows),
        "flagged_events_correlated": len(events),
        "candidate_relationships": edge_count,
        "correlation_window_seconds": 900,
        "correlation_basis": ["shared user", "shared source IP", "shared host", "within 15 minutes"],
        "timeline": timeline,
        "relationship_summary": dict(shared_entity_edges),
        "interpretation": (
            "Relationships are candidate correlations based on shared entities and time proximity. "
            "They do not establish causation or prove an attack. Events without shared entities "
            "are not linked merely because they occurred close together."
        )
    }

def main():
    p = argparse.ArgumentParser(description="Correlate flagged TRACE-X security events into a timeline.")
    p.add_argument("--logs", default="data/cyberattack/cyber_incident_logs.csv")
    p.add_argument("--report", default="results/cyberattack/investigation_report.json")
    p.add_argument("--out", default="results/cyberattack/temporal_correlation.json")
    args = p.parse_args()

    rows = load_csv(args.logs)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = correlate(rows, report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("TRACE-X TEMPORAL CORRELATION COMPLETE")
    print(f"Original log records: {result['records_in_original_log']}")
    print(f"Flagged events: {result['flagged_events_correlated']}")
    print(f"Candidate relationships: {result['candidate_relationships']}")
    print(f"Output: {out}")
    print("\nTIMELINE")
    for item in result["timeline"]:
        print(f"{item['sequence']:02d}. {item['timestamp']} | {item['event']} | row={item['row']}")
        if item["candidate_links"]:
            for link in item["candidate_links"]:
                print(f"    -> row {link['to_row']} via {', '.join(link['shared_entities'])} "
                      f"({link['time_gap_seconds']} sec)")
        else:
            print("    -> no entity-backed link found")
    print("\nCaution: candidate correlation is not proof of causation or compromise.")

if __name__ == "__main__":
    main()
