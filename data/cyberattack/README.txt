TRACE-X Synthetic Cyber Incident Dataset V1

Files:
- cyber_incident_logs.csv: 150 timestamp-sorted synthetic security events.
- cyber_incident_logs.jsonl: same events in JSON Lines format.
- ground_truth.json: scenario, label definition, and caveats.

Columns:
timestamp, source, severity, event_type, user, source_ip, host, message,
is_suspicious, attack_stage

Scenario:
A simulated account-compromise investigation with repeated failed logins,
a subsequent successful login from an unfamiliar documentation IP, an
unexpected privilege change, and unusual endpoint/network indicators.

All events are fictional. IPs use reserved documentation ranges.
Labels are scenario ground truth, not proof of maliciousness.
Human review is required; do not claim real-world detection accuracy.
