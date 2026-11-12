import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

def parse_time(value):
    try:
        return datetime.fromisoformat(value.strip())
    except Exception:
        return None

def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise ValueError("Supported input formats: .csv and .jsonl")

def analyze(rows):
    events = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["_index"] = index
        item["_time"] = parse_time(str(row.get("timestamp", "")))
        item["_flags"] = []
        item["_score"] = 0
        events.append(item)

    # Rule 1: repeated failed logins grouped by account and source IP.
    failures = defaultdict(list)
    for e in events:
        if str(e.get("event_type", "")).lower() == "login_failed":
            failures[(e.get("user", "-"), e.get("source_ip", "-"))].append(e)

    for (user, ip), group in failures.items():
        valid = [e for e in group if e["_time"]]
        if len(valid) >= 3:
            for e in valid:
                e["_flags"].append("Repeated failed logins for same account/source")
                e["_score"] += 3

    # Rule 2: successful login shortly after a burst of failures for same account/IP.
    for e in events:
        if str(e.get("event_type", "")).lower() != "login_success" or not e["_time"]:
            continue
        key = (e.get("user", "-"), e.get("source_ip", "-"))
        prior = [
            f for f in failures.get(key, [])
            if f["_time"] and 0 <= (e["_time"] - f["_time"]).total_seconds() <= 600
        ]
        if len(prior) >= 2:
            e["_flags"].append("Successful login within 10 minutes of repeated failures")
            e["_score"] += 4

    # Rule 3: privilege changes and suspicious endpoint events.
    for e in events:
        event_type = str(e.get("event_type", "")).lower()
        message = str(e.get("message", "")).lower()
        if event_type in {"privilege_change", "account_created", "permission_change"}:
            e["_flags"].append("Account privilege or permission change")
            e["_score"] += 4
        if "unsigned" in message or event_type in {"process_anomaly", "malware_alert"}:
            e["_flags"].append("Potentially unusual endpoint process activity")
            e["_score"] += 3

    # Rule 4: network egress and transfer anomalies.
    for e in events:
        event_type = str(e.get("event_type", "")).lower()
        if event_type in {"connection_anomaly", "transfer_anomaly", "unusual_outbound"}:
            e["_flags"].append("Unusual outbound network or data-transfer activity")
            e["_score"] += 4

    # Rule 5: alert-level records are worth review, but not automatically malicious.
    for e in events:
        if str(e.get("severity", "")).upper() == "ALERT" and not e["_flags"]:
            e["_flags"].append("Alert severity requires analyst review")
            e["_score"] += 1

    suspicious = [e for e in events if e["_score"] > 0]
    suspicious.sort(key=lambda x: (x["_time"] or datetime.max, -x["_score"]))

    # Build an evidence-grounded, heuristic narrative from observed flags.
    findings = []
    if any("Repeated failed logins" in f for e in events for f in e["_flags"]):
        findings.append("Repeated authentication failures were observed for an account/source combination.")
    if any("Successful login within" in f for e in events for f in e["_flags"]):
        findings.append("A successful login followed repeated failures within the configured 10-minute window.")
    if any("privilege" in f.lower() for e in events for f in e["_flags"]):
        findings.append("An account privilege or permission change was recorded.")
    if any("endpoint process" in f.lower() for e in events for f in e["_flags"]):
        findings.append("Potentially unusual endpoint process activity was recorded.")
    if any("outbound network" in f.lower() for e in events for f in e["_flags"]):
        findings.append("Unusual outbound network or data-transfer activity was recorded.")

    # Correlate by account and source IP for analyst pivots.
    pivots = defaultdict(list)
    for e in suspicious:
        for field in ("user", "source_ip", "host"):
            value = str(e.get(field, "")).strip()
            if value and value != "-":
                pivots[f"{field}:{value}"].append(e["_index"])

    report_events = []
    for e in suspicious:
        report_events.append({
            "row": e["_index"],
            "timestamp": e.get("timestamp"),
            "source": e.get("source"),
            "severity": e.get("severity"),
            "event_type": e.get("event_type"),
            "user": e.get("user"),
            "source_ip": e.get("source_ip"),
            "host": e.get("host"),
            "message": e.get("message"),
            "heuristic_score": e["_score"],
            "reasons": e["_flags"]
        })

    return {
        "tool": "TRACE-X Log Investigator (rule-based prototype)",
        "records_analyzed": len(events),
        "flagged_records": len(suspicious),
        "findings": findings,
        "flagged_events": report_events,
        "entity_pivots": {k: v for k, v in pivots.items() if len(v) > 1},
        "interpretation": (
            "Heuristic indicators only. A flagged event is not proof of compromise, "
            "malware, command-and-control, or data exfiltration. Validate with source "
            "systems and an analyst. No machine-learning model is used by this script."
        )
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze TRACE-X synthetic or authorized security logs.")
    parser.add_argument("input", help="Path to .csv or .jsonl log file")
    parser.add_argument("--out", default="results/cyberattack", help="Output directory")
    args = parser.parse_args()

    rows = load_rows(args.input)
    result = analyze(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "investigation_report.json"
    txt_path = out_dir / "investigation_report.txt"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "TRACE-X | CYBERATTACK LOG INVESTIGATION",
        "=" * 48,
        f"Records analyzed: {result['records_analyzed']}",
        f"Flagged records:  {result['flagged_records']}",
        "",
        "KEY FINDINGS"
    ]
    lines += [f"- {x}" for x in result["findings"]] or ["- No configured rule patterns matched."]
    lines += ["", "FLAGGED EVENTS"]
    for e in result["flagged_events"]:
        lines.append(
            f"\nRow {e['row']} | {e['timestamp']} | {e['severity']} | "
            f"{e['event_type']} | score={e['heuristic_score']}"
        )
        lines.append(f"  {e['message']}")
        for reason in e["reasons"]:
            lines.append(f"  Reason: {reason}")
    lines += ["", "CAUTION", result["interpretation"]]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("TRACE-X LOG INVESTIGATION COMPLETE")
    print(f"Records analyzed: {result['records_analyzed']}")
    print(f"Flagged records: {result['flagged_records']}")
    print(f"JSON report: {json_path}")
    print(f"Text report: {txt_path}")
    print("\nKey findings:")
    for finding in result["findings"]:
        print("-", finding)
    if not result["findings"]:
        print("- No configured rule patterns matched.")

if __name__ == "__main__":
    main()
