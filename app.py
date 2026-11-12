import csv
import io
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from flask import (
    Flask,
    jsonify,
    render_template,
    render_template_string,
    request,
    redirect,
    send_from_directory
)

from torch import nn


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "synthetic_v1"
FEATURE_DIR = ROOT / "features" / "v1"

MODEL_PATH = ROOT / "models" / "v1" / "trace_siamese_multimodal_v1.pt"

VALIDATION_PATH = (
    ROOT / "results" / "v1" /
    "trace_siamese_multimodal_v1_validation.json"
)

TEST_PATH = DATA_DIR / "test.jsonl"

TEST_REPORT_PATH = (
    ROOT / "results" / "v1" /
    "trace_siamese_multimodal_v1_test.json"
)

CYBER_DIR = ROOT / "results" / "cyberattack"

CYBER_REPORT_PATH = CYBER_DIR / "investigation_report.json"
CYBER_TIMELINE_PATH = CYBER_DIR / "temporal_correlation.json"
CYBER_GRAPH_PATH = CYBER_DIR / "investigation_graph.html"


app = Flask(__name__)

# Maximum uploaded file size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Latest uploaded investigation (single-user local demo state).
LATEST_CYBER_RESULT = None
LATEST_CYBER_FILENAME = None


@app.after_request
def add_investigation_dashboard_link(response):
    """Add a visible investigation entry point to the existing dashboard."""
    if request.path == "/" and response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:
            body = response.get_data(as_text=True)
            cta = """
            <a href="/cyberattack" id="tracex-investigation-entry"
               style="position:fixed;right:24px;bottom:24px;z-index:99999;
               display:inline-flex;align-items:center;gap:9px;
               padding:14px 19px;border-radius:12px;
               background:#16a9bd;color:#03121b;text-decoration:none;
               font:700 14px Segoe UI,Arial,sans-serif;
               box-shadow:0 8px 28px rgba(0,0,0,.35);">
               ↗ Open Log Investigation
            </a>
            """
            if "</body>" in body.lower():
                pos = body.lower().rfind("</body>")
                body = body[:pos] + cta + body[pos:]
            else:
                body += cta
            response.set_data(body)
    return response


# ============================================================
# 2. EXISTING SIAMESE MULTIMODAL MODEL
# ============================================================

class ModalityEncoder(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.text_embedding = nn.Embedding(
            vocab_size, 32, padding_idx=0
        )

        self.text_gru = nn.GRU(
            32, 32,
            batch_first=True,
            bidirectional=True
        )

        self.text_projection = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU()
        )

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(48, 48),
            nn.ReLU()
        )

        self.activity_gru = nn.GRU(
            6, 32,
            batch_first=True
        )

        self.activity_projection = nn.Sequential(
            nn.Linear(32, 48),
            nn.ReLU()
        )

    def forward(self, text, image, activity):

        _, ht = self.text_gru(
            self.text_embedding(text)
        )

        text_features = self.text_projection(
            torch.cat([ht[-2], ht[-1]], dim=1)
        )

        image_features = self.image_encoder(image)

        _, ha = self.activity_gru(activity)

        activity_features = self.activity_projection(
            ha[-1]
        )

        return (
            text_features,
            image_features,
            activity_features
        )


class SiameseTraceNet(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.encoder = ModalityEncoder(vocab_size)

        self.classifier = nn.Sequential(
            nn.Linear(288, 192),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(96, 1)
        )

    def forward(self, ta, ia, aa, tb, ib, ab):

        za = self.encoder(ta, ia, aa)
        zb = self.encoder(tb, ib, ab)

        parts = []

        for a, b in zip(za, zb):
            parts.extend([
                torch.abs(a - b),
                a * b
            ])

        return self.classifier(
            torch.cat(parts, dim=1)
        ).squeeze(1)


# ============================================================
# 3. LOAD EXISTING MODEL
# ============================================================

def load_runtime():

    required = [
        MODEL_PATH,
        FEATURE_DIR / "vocab.json",
        VALIDATION_PATH,
        TEST_PATH
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing TRACE-X files:\n" +
            "\n".join(missing)
        )

    vocab = json.loads(
        (FEATURE_DIR / "vocab.json").read_text(
            encoding="utf-8"
        )
    )

    validation = json.loads(
        VALIDATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    threshold = float(
        validation["validation"]["threshold"]
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = SiameseTraceNet(
        int(checkpoint["vocab_size"])
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    return vocab, threshold, device, model


try:

    VOCAB, THRESHOLD, DEVICE, MODEL = load_runtime()

    RUNTIME_ERROR = None

except Exception as exc:

    VOCAB = {}
    THRESHOLD = 0.393115
    DEVICE = torch.device("cpu")
    MODEL = None

    RUNTIME_ERROR = str(exc)


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

def load_test_rows():

    if not TEST_PATH.exists():
        return []

    with TEST_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


try:

    ALL_ROWS = load_test_rows()
    DATA_LOAD_ERROR = None

except Exception as exc:

    ALL_ROWS = []
    DATA_LOAD_ERROR = str(exc)


INCIDENTS = {}

for row in ALL_ROWS:

    incident_id = int(row["incident_id"])

    INCIDENTS.setdefault(
        incident_id, []
    ).append(row)


# ============================================================
# 5. SIAMESE MODEL INFERENCE
# ============================================================

def encode_record(row):

    tokens = (
        str(row["text"])
        .lower()
        .strip()
        .split()[:48]
    )

    ids = [
        VOCAB.get(token, 1)
        for token in tokens
    ]

    ids += [0] * (48 - len(ids))

    text = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    image = torch.tensor(
        np.asarray(
            row["image"],
            dtype=np.float32
        )[None, ...],
        device=DEVICE
    )

    activity = torch.tensor(
        np.asarray(
            row["activity"],
            dtype=np.float32
        )[None, ...],
        device=DEVICE
    )

    return text, image, activity


def score_incident(incident_id):

    if MODEL is None:

        raise RuntimeError(
            RUNTIME_ERROR or
            "Model could not be loaded."
        )

    rows = INCIDENTS.get(incident_id)

    if not rows:

        raise ValueError(
            "Incident not found in held-out test split."
        )

    encoded = [
        encode_record(row)
        for row in rows
    ]

    edges = []

    with torch.no_grad():

        for i in range(len(rows)):

            for j in range(i + 1, len(rows)):

                a = encoded[i]
                b = encoded[j]

                logit = MODEL(
                    a[0], a[1], a[2],
                    b[0], b[1], b[2]
                )

                score = float(
                    torch.sigmoid(logit).item()
                )

                edges.append({
                    "source": i,
                    "target": j,
                    "score": score,
                    "related": score >= THRESHOLD,
                    "source_id": rows[i]["evidence_id"],
                    "target_id": rows[j]["evidence_id"]
                })

    edges.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return rows, edges


# ============================================================
# 6. CYBERATTACK LOG PARSING
# ============================================================

def parse_timestamp(value):

    try:

        return datetime.fromisoformat(
            str(value).strip().replace("Z", "+00:00")
        )

    except Exception:

        return None


def normalize_record(row, row_number):

    lower = {
        str(k).strip().lower(): v
        for k, v in row.items()
        if k is not None
    }

    def pick(*names, default="-"):

        for name in names:

            value = lower.get(name)

            if value not in (None, ""):
                return str(value).strip()

        return default

    message = pick(
        "message", "msg", "description",
        "details", "event_message", "raw"
    )

    event_type = pick(
        "event_type", "event", "action",
        "eventid", "event_id",
        default="unknown"
    ).lower()

    severity = pick(
        "severity", "level", "log_level",
        "priority",
        default="INFO"
    ).upper()

    source = pick(
        "source", "log_source",
        "service", "provider",
        default="UNKNOWN"
    )

    user = pick(
        "user", "username",
        "account", "user_name"
    )

    ip = pick(
        "source_ip", "src_ip",
        "ip", "src", "remote_ip",
        "client_ip"
    )

    host = pick(
        "host", "hostname",
        "computer", "device", "endpoint"
    )

    timestamp = pick(
        "timestamp", "time",
        "datetime", "date",
        "event_time",
        default=""
    )

    msglow = message.lower()

    # Infer event type when not explicitly provided.

    if event_type == "unknown":

        if (
            "failed" in msglow
            and (
                "login" in msglow
                or "logon" in msglow
                or "auth" in msglow
            )
        ):

            event_type = "login_failed"

        elif (
            ("successful" in msglow or "success" in msglow)
            and (
                "login" in msglow
                or "logon" in msglow
            )
        ):

            event_type = "login_success"

        elif (
            "privilege" in msglow
            or "permission changed" in msglow
        ):

            event_type = "privilege_change"

        elif (
            "transfer" in msglow
            or "exfil" in msglow
        ):

            event_type = "transfer_anomaly"

        elif (
            "outbound" in msglow
            or "connection anomaly" in msglow
        ):

            event_type = "connection_anomaly"

        elif (
            "unsigned" in msglow
            or "suspicious process" in msglow
        ):

            event_type = "process_anomaly"

        else:

            event_type = "unknown"

    return {
        "row": row_number,
        "timestamp": timestamp,
        "source": source,
        "severity": severity,
        "event_type": event_type,
        "user": user,
        "source_ip": ip,
        "host": host,
        "message": message
    }


# ============================================================
# 7. UPLOAD FILE READER
# ============================================================

def parse_uploaded_file(file_storage):

    filename = file_storage.filename or "uploaded_logs"

    suffix = Path(filename).suffix.lower()

    raw = file_storage.read()

    if not raw:

        raise ValueError(
            "Uploaded file is empty."
        )

    if len(raw) > 10 * 1024 * 1024:

        raise ValueError(
            "File exceeds the 10 MB limit."
        )

    text = raw.decode(
        "utf-8-sig",
        errors="replace"
    )

    if suffix == ".csv":

        source_rows = list(
            csv.DictReader(
                io.StringIO(text)
            )
        )

    elif suffix == ".json":

        obj = json.loads(text)

        if isinstance(obj, list):

            source_rows = obj

        elif isinstance(obj, dict):

            source_rows = obj.get(
                "logs",
                obj.get(
                    "events",
                    [obj]
                )
            )

        else:

            raise ValueError(
                "JSON must contain an object or array."
            )

    elif suffix == ".jsonl":

        source_rows = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    elif suffix in {".log", ".txt"}:

        source_rows = []

        pattern = re.compile(
            r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]"
            r"\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+"
            r"(?:(?P<level>INFO|WARN|WARNING|ERROR|"
            r"CRITICAL|ALERT|DEBUG)\s+)?"
            r"(?P<msg>.*)$",
            re.IGNORECASE
        )

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            match = pattern.match(line)

            if match:

                source_rows.append({
                    "timestamp": match.group("ts"),
                    "severity": (
                        match.group("level") or "INFO"
                    ).upper(),
                    "message": match.group("msg")
                })

            else:

                source_rows.append({
                    "message": line
                })

    else:

        raise ValueError(
            "Supported formats: CSV, JSON, JSONL, LOG and TXT."
        )

    if not source_rows:

        raise ValueError(
            "No log records found."
        )

    records = [
        normalize_record(row, i)
        for i, row in enumerate(source_rows, start=1)
    ]

    return filename, records


# ============================================================
# 8. CYBERATTACK DETECTION ENGINE
# ============================================================

def investigate_records(records):

    events = []

    for record in records:

        event = dict(record)

        event["_time"] = parse_timestamp(
            event.get("timestamp", "")
        )

        event["_score"] = 0
        event["_reasons"] = []

        events.append(event)

    # Repeated failed logins

    failures = defaultdict(list)

    for event in events:

        if event["event_type"] == "login_failed":

            failures[
                (
                    event["user"],
                    event["source_ip"]
                )
            ].append(event)

    for key, group in failures.items():

        timed = [
            e for e in group
            if e["_time"]
        ]

        if len(timed) >= 3:

            for event in timed:

                event["_score"] += 3

                event["_reasons"].append(
                    "Three or more failed logins for the same account/source."
                )

    # Successful login following repeated failures

    for event in events:

        if (
            event["event_type"] == "login_success"
            and event["_time"]
        ):

            previous = [
                failure
                for failure in failures[
                    (
                        event["user"],
                        event["source_ip"]
                    )
                ]
                if (
                    failure["_time"]
                    and 0 <= (
                        event["_time"] -
                        failure["_time"]
                    ).total_seconds() <= 600
                )
            ]

            if len(previous) >= 2:

                event["_score"] += 4

                event["_reasons"].append(
                    "Successful login within 10 minutes of repeated failures."
                )

    # Other security indicators

    for event in events:

        event_type = event["event_type"]
        message = event["message"].lower()

        if event_type in {
            "privilege_change",
            "account_created",
            "permission_change"
        }:

            event["_score"] += 4

            event["_reasons"].append(
                "Account privilege or permission change."
            )

        if (
            event_type in {
                "process_anomaly",
                "malware_alert"
            }
            or "unsigned" in message
        ):

            event["_score"] += 3

            event["_reasons"].append(
                "Potentially unusual endpoint process activity."
            )

        if event_type in {
            "connection_anomaly",
            "transfer_anomaly",
            "unusual_outbound"
        }:

            event["_score"] += 4

            event["_reasons"].append(
                "Unusual outbound network or data-transfer activity."
            )

        if (
            event["severity"] == "ALERT"
            and not event["_reasons"]
        ):

            event["_score"] += 1

            event["_reasons"].append(
                "Alert severity requires analyst review."
            )

    # Keep flagged events

    flagged = [
        event
        for event in events
        if event["_score"] > 0
    ]

    flagged.sort(
        key=lambda e: (
            e["_time"] or datetime.max,
            e["row"]
        )
    )

    # Entity-based temporal correlation

    for event in flagged:
        event["_links"] = []

    for i, left in enumerate(flagged):

        for right in flagged[i + 1:]:

            if not left["_time"] or not right["_time"]:
                continue

            gap = (
                right["_time"] -
                left["_time"]
            ).total_seconds()

            if gap < 0 or gap > 900:
                continue

            shared = [
                field
                for field in (
                    "user",
                    "source_ip",
                    "host"
                )
                if (
                    left.get(field) not in (
                        None, "", "-"
                    )
                    and left.get(field) == right.get(field)
                )
            ]

            if shared:

                left["_links"].append({
                    "to_row": right["row"],
                    "shared_entities": shared,
                    "time_gap_seconds": int(gap)
                })

    # Prepare output

    timeline = []

    for event in flagged:

        timeline.append({
            "row": event["row"],
            "timestamp": event["timestamp"],
            "source": event["source"],
            "severity": event["severity"],
            "event_type": event["event_type"],
            "user": event["user"],
            "source_ip": event["source_ip"],
            "host": event["host"],
            "message": event["message"],
            "heuristic_score": event["_score"],
            "reasons": event["_reasons"],
            "candidate_links": event["_links"]
        })

    findings = []

    categories = [
        (
            "login_failed",
            "Repeated authentication failures were detected."
        ),
        (
            "login_success",
            "A successful login followed repeated failures."
        ),
        (
            "privilege_change",
            "An account privilege change was detected."
        ),
        (
            "process_anomaly",
            "Potentially unusual endpoint process activity was detected."
        ),
        (
            "connection_anomaly",
            "An unusual outbound connection was detected."
        ),
        (
            "transfer_anomaly",
            "An unusual data-transfer event was detected."
        )
    ]

    for event_type, description in categories:

        if any(
            e["event_type"] == event_type
            and e["_score"] > 0
            for e in events
        ):

            findings.append(description)

    return {
        "records_analyzed": len(records),
        "flagged_records": len(flagged),
        "candidate_relationships": sum(
            len(e["_links"])
            for e in flagged
        ),
        "findings": findings,
        "timeline": timeline,
        "interpretation": (
            "This prototype uses heuristic rules, not the trained "
            "Siamese model. Flags and links are leads for analyst "
            "review, not proof of compromise, causation, malware, "
            "command-and-control, or data exfiltration."
        )
    }


# ============================================================
# 9. CYBERATTACK DASHBOARD HTML
# ============================================================

CYBER_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>TRACE-X Cyberattack Investigator</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #080d18;
    color: #e9effa;
    font-family: Segoe UI, Arial, sans-serif;
}

header {
    padding: 22px 4%;
    border-bottom: 1px solid #26364f;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

h1 {
    margin: 0;
    font-size: 23px;
}

header p,
.muted {
    color: #9aabc2;
}

.tag {
    color: #42d9aa;
    border: 1px solid #216b5b;
    padding: 8px 12px;
    border-radius: 30px;
}

main {
    max-width: 1300px;
    margin: 28px auto;
    padding: 0 22px;
}

.hero {
    margin-bottom: 22px;
}

.hero h2 {
    font-size: 30px;
}

.panel {
    background: #111b2d;
    border: 1px solid #26364f;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
}

.upload {
    border: 1px dashed #3e6684;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    background: #0d1727;
}

input[type=file] {
    display: block;
    margin: 18px auto;
    max-width: 100%;
}

button {
    background: #16a9bd;
    color: #03121b;
    border: 0;
    border-radius: 8px;
    padding: 12px 22px;
    font-weight: bold;
    cursor: pointer;
}

.cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.card {
    background: #0d1727;
    border: 1px solid #26364f;
    border-radius: 10px;
    padding: 16px;
}

.card b {
    display: block;
    font-size: 26px;
    margin-top: 5px;
}

.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}

.event {
    border-left: 3px solid #f0b65d;
    background: #0d1727;
    padding: 13px;
    margin: 9px 0;
    border-radius: 7px;
}

.event strong {
    color: #72ddeb;
}

.reason {
    color: #b9c9dd;
    margin-top: 5px;
}

.link {
    color: #68d9ad;
    font-size: 12px;
}

.alert {
    padding: 13px;
    background: #302719;
    border: 1px solid #795c2b;
    border-radius: 8px;
    color: #f4d59d;
    margin: 15px 0;
}

.error {
    background: #351e29;
    color: #ffc0cb;
    padding: 13px;
    border-radius: 8px;
}

a {
    color: #67d9e9;
}

table {
    width: 100%;
    border-collapse: collapse;
}

td, th {
    text-align: left;
    padding: 9px;
    border-bottom: 1px solid #26364f;
    vertical-align: top;
}

@media(max-width:800px) {

    .cards,
    .grid {
        grid-template-columns: 1fr 1fr;
    }

}

@media(max-width:560px) {

    .cards,
    .grid {
        grid-template-columns: 1fr;
    }

    header {
        flex-direction: column;
        align-items: flex-start;
    }

}

</style>

</head>

<body>

<header>

<div>

<h1>
TRACE-X / CYBERATTACK INVESTIGATOR
</h1>

<p>
Security log analysis · Event correlation · Evidence investigation
</p>

</div>

<span class="tag">
LOCAL RESEARCH DEMO
</span>

</header>


<main>

<section class="hero">

<h2>Investigate security logs</h2>

<p class="muted">

Upload an authorized security log file.
TRACE-X will parse records, identify configured suspicious
patterns, correlate related events and build a timeline.

</p>

</section>


<section class="panel">

<form
method="POST"
action="/cyberattack/analyze"
enctype="multipart/form-data"
>

<div class="upload">

<h3>Upload Security Logs</h3>

<p class="muted">
CSV, JSON, JSONL, LOG or TXT · Maximum 10 MB
</p>

<input
type="file"
name="logfile"
accept=".csv,.json,.jsonl,.log,.txt"
required
>

<button type="submit">
ANALYZE LOGS
</button>

</div>

</form>


{% if error %}

<div class="error">
{{ error }}
</div>

{% endif %}


{% if result %}

<div class="alert">

<strong>Interpretation:</strong>

{{ result.interpretation }}

</div>


<h2>Investigation Results</h2>

<p class="muted">
File: {{ filename }}
</p>


<div class="cards">

<div class="card">
Records analyzed
<b>{{ result.records_analyzed }}</b>
</div>

<div class="card">
Flagged records
<b>{{ result.flagged_records }}</b>
</div>

<div class="card">
Candidate links
<b>{{ result.candidate_relationships }}</b>
</div>

<div class="card">
Finding categories
<b>{{ result.findings|length }}</b>
</div>

</div>


<div class="grid" style="margin-top:18px">

<section class="panel">

<h3>Key Findings</h3>

{% if result.findings %}

<ul>

{% for finding in result.findings %}

<li>{{ finding }}</li>

{% endfor %}

</ul>

{% else %}

<p class="muted">
No configured detection rules matched this file.
</p>

{% endif %}

</section>


<section class="panel">

<h3>Investigation Timeline</h3>

<p class="muted">
Events are ordered by timestamp.
</p>

{% if result.timeline %}

{% for event in result.timeline %}

<div class="event">

<strong>
{{ event.timestamp or 'Timestamp unavailable' }}
·
{{ event.event_type }}
</strong>

<div>
{{ event.message }}
</div>

<div class="reason">

Row {{ event.row }}
·
{{ event.severity }}
·
Score {{ event.heuristic_score }}

</div>

{% for reason in event.reasons %}

<div class="reason">
• {{ reason }}
</div>

{% endfor %}

{% if event.candidate_links %}

<div class="link">

Candidate links:

{% for link in event.candidate_links %}

Row {{ link.to_row }}
(
{{ link.shared_entities|join(', ') }}
,
{{ link.time_gap_seconds }} seconds
)

{% endfor %}

</div>

{% endif %}

</div>

{% endfor %}

{% else %}

<p class="muted">
No flagged events.
</p>

{% endif %}

</section>

</div>


<section class="panel">

<h3>Flagged Evidence Details</h3>

{% if result.timeline %}

<table>

<thead>

<tr>
<th>Time</th>
<th>Event</th>
<th>User / IP / Host</th>
<th>Detection Reason</th>
</tr>

</thead>

<tbody>

{% for event in result.timeline %}

<tr>

<td>
{{ event.timestamp }}
</td>

<td>
{{ event.event_type }}
<br>
{{ event.severity }}
</td>

<td>
{{ event.user }}
<br>
{{ event.source_ip }}
<br>
{{ event.host }}
</td>

<td>
{{ event.reasons|join('; ') }}
</td>

</tr>

{% endfor %}

</tbody>

</table>

{% else %}

<p class="muted">
No flagged evidence.
</p>

{% endif %}

</section>

{% endif %}

</section>


<p>
<a href="/">
← Existing TRACE-X dashboard
</a>
</p>

</main>

</body>

</html>
"""



# ============================================================
# INTEGRATED UPLOAD-TO-INVESTIGATION GRAPH VIEW
# ============================================================

INVESTIGATION_VIEW = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TRACE-X | Investigation Workspace</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#080d18;color:#e9effa;font-family:Segoe UI,Arial,sans-serif}
header{padding:18px 4%;border-bottom:1px solid #26364f;display:flex;justify-content:space-between;align-items:center;gap:15px}
h1{font-size:22px;margin:0}h2{margin:0 0 12px}p,.muted{color:#9aabc2}
a{color:#67d9e9}.wrap{max-width:1450px;margin:25px auto;padding:0 22px}
.panel{background:#111b2d;border:1px solid #26364f;border-radius:14px;padding:20px;margin-bottom:17px}
.topline{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
button,.btn{border:0;border-radius:8px;padding:11px 16px;background:#16a9bd;color:#03121b;font-weight:700;cursor:pointer;text-decoration:none;display:inline-block}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
.tab{background:#18263b;color:#dce8f7;border:1px solid #344a66}.tab.active{background:#16a9bd;color:#03121b}
.view{display:none}.view.active{display:block}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:#0d1727;border:1px solid #26364f;border-radius:10px;padding:15px}
.card b{display:block;font-size:25px;margin-top:5px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #26364f;vertical-align:top}
.event{padding:13px;margin:9px 0;background:#0d1727;border-left:3px solid #e8ae55;border-radius:7px}
.reason{color:#b9c9dd;font-size:13px;margin-top:5px}.warning{background:#302719;border:1px solid #795c2b;padding:13px;border-radius:8px;color:#f4d59d;margin:14px 0}
.graphbox{overflow:auto;background:#0b1423;border:1px solid #26364f;border-radius:12px;padding:14px;min-height:420px}
svg{width:100%;min-width:850px;height:auto;min-height:520px;display:block}.lane{fill:#101d30;stroke:#26364f;stroke-width:1}.lane-title{fill:#9db4ce;font-size:15px;font-weight:700}.edge{stroke:#6c86a4;stroke-width:2.2;opacity:.42;fill:none}.edge.is-active{stroke:#35c9d8;stroke-width:3.5;opacity:1}.edge.is-dim{opacity:.07}.edge-hit{stroke:transparent;stroke-width:15;fill:none;cursor:pointer}.edge-label{fill:#9aabc2;font-size:11px;opacity:0;pointer-events:none}.node{cursor:pointer;transition:opacity .15s}.node circle{stroke:#8bddea;stroke-width:2.5;fill:#16334a}.node.auth circle{fill:#62451f;stroke:#f3bd67}.node.endpoint circle{fill:#183f59;stroke:#62c8e6}.node.network circle{fill:#254b3c;stroke:#6bd6a5}.node text{fill:#e9effa;font-size:12px;text-anchor:middle;pointer-events:none}.node .event-label{font-size:11px;fill:#b8c9dc}.node.is-dim{opacity:.2}.node.is-active circle{stroke:#fff;stroke-width:4}.edge.is-active+.edge-label{opacity:1}.graph-legend{display:flex;gap:18px;flex-wrap:wrap;color:#aebed1;font-size:13px;margin:12px 2px}.legend-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.details{background:#0d1727;padding:14px;border-radius:9px;margin-top:12px;min-height:65px}
.uploadbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.uploadbar input{max-width:100%}
@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}}@media(max-width:560px){.cards{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<header><div><h1>TRACE-X / INVESTIGATION WORKSPACE</h1><p>Evidence extraction · Behaviour reconstruction · Relationship graph</p></div>
<a href="/">← Dashboard</a></header>
<main class="wrap">
<section class="panel">
<div class="topline"><div><h2>Investigation generated</h2><p class="muted">Uploaded file: {{ filename }}</p></div>
<form class="uploadbar" method="POST" action="/cyberattack/analyze" enctype="multipart/form-data">
<input type="file" name="logfile" accept=".csv,.json,.jsonl,.log,.txt" required>
<button type="submit">Analyze another file</button></form></div>
<div class="cards">
<div class="card">Records analyzed<b>{{ result.records_analyzed }}</b></div>
<div class="card">Flagged records<b>{{ result.flagged_records }}</b></div>
<div class="card">Candidate relationships<b>{{ result.candidate_relationships }}</b></div>
<div class="card">Finding categories<b>{{ result.findings|length }}</b></div>
</div>
<div class="warning"><strong>Analyst note:</strong> {{ result.interpretation }}</div>
</section>
<nav class="tabs">
<button class="tab active" data-tab="evidence">1 · Evidence Extraction</button>
<button class="tab" data-tab="behaviour">2 · Behaviour Reconstruction</button>
<button class="tab" data-tab="graph">3 · Investigation Graph</button>
</nav>
<section id="evidence" class="view active panel">
<h2>Extracted suspicious evidence</h2>
<p class="muted">Flagged records and the rule-based reasons that caused them to be selected.</p>
{% if result.timeline %}
<div style="overflow-x:auto"><table><thead><tr><th>Row / Time</th><th>Event</th><th>User</th><th>Source IP</th><th>Host</th><th>Score / Reason</th></tr></thead><tbody>
{% for e in result.timeline %}<tr><td>{{ e.row }}<br>{{ e.timestamp or "Time unavailable" }}</td><td>{{ e.event_type }}<br>{{ e.severity }}</td><td>{{ e.user }}</td><td>{{ e.source_ip }}</td><td>{{ e.host }}</td><td>{{ e.heuristic_score }}<br>{{ e.reasons|join("; ") }}</td></tr>{% endfor %}
</tbody></table></div>{% else %}<p>No records matched the configured rules.</p>{% endif %}
</section>
<section id="behaviour" class="view panel">
<h2>Behaviour reconstruction</h2><p class="muted">Chronological sequence of flagged events. This is an investigation hypothesis, not proof of an attack.</p>
<div class="topline"><button id="play">▶ Play sequence</button><button id="pause" class="tab">Pause</button><span id="stepLabel" class="muted">Ready</span></div>
<div id="sequence">{% for e in result.timeline %}<div class="event sequence-event"><strong>{{ e.timestamp or "Timestamp unavailable" }} · {{ e.event_type }}</strong><div>{{ e.message }}</div><div class="reason">Row {{ e.row }} · {{ e.user }} · {{ e.source_ip }} · {{ e.host }}</div><div class="reason">{{ e.reasons|join("; ") }}</div></div>{% endfor %}</div>
</section>
<section id="graph" class="view panel">
<h2>Evidence relationship graph</h2><p class="muted">Nodes represent flagged log records. Lines represent candidate links based on a shared user, source IP or host within 15 minutes.</p>
<div class="graph-legend"><span>🟠 Authentication</span><span>🔵 Endpoint activity</span><span>🟢 Network activity</span><span>Lines = candidate links; select a node to isolate its relationships.</span></div>
<div class="graphbox"><svg id="relationshipGraph" viewBox="0 0 1100 560" role="img" aria-label="Investigation relationship graph"></svg></div>
<div id="nodeDetails" class="details">Select a node to inspect its evidence and the entities shared with connected records.</div>
</section>
<section class="panel"><strong>Key findings</strong><ul>{% for finding in result.findings %}<li>{{ finding }}</li>{% else %}<li>No configured finding categories matched.</li>{% endfor %}</ul>
<a class="btn" href="/cyberattack">← Upload page</a> <a class="btn" href="/">Dashboard</a></section>
</main>
<script>
const events = {{ result.timeline|tojson }};
const svg = document.getElementById('relationshipGraph');
const NS = 'http://www.w3.org/2000/svg';
const W = 1100, H = 560;
const positions = {};
const byRow = new Map(events.map(e => [Number(e.row), e]));
function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  Object.entries(attrs || {}).forEach(([k,v]) => n.setAttribute(k, v));
  parent.appendChild(n);
  return n;
}
function laneOf(e) {
  const t = String(e.event_type || '').toLowerCase();
  if (t.includes('login') || t.includes('auth')) return 'auth';
  if (t.includes('privilege') || t.includes('permission') ||
      t.includes('process') || t.includes('malware') || t.includes('account')) return 'endpoint';
  return 'network';
}
const lanes = [
  {id:'auth', title:'01  AUTHENTICATION', x:20, w:340},
  {id:'endpoint', title:'02  ENDPOINT / PRIVILEGE', x:380, w:340},
  {id:'network', title:'03  NETWORK / TRANSFER', x:740, w:340}
];
lanes.forEach(l => {
  el('rect',{x:l.x,y:28,width:l.w,height:500,rx:14,class:'lane'},svg);
  const title=el('text',{x:l.x+18,y:55,class:'lane-title'},svg);
  title.textContent=l.title;
});
const grouped = {auth:[], endpoint:[], network:[]};
events.forEach(e => grouped[laneOf(e)].push(e));
lanes.forEach(l => {
  const arr=grouped[l.id];
  arr.forEach((e,i) => {
    const gap=440/Math.max(arr.length,1);
    positions[Number(e.row)]={x:l.x+l.w/2,y:92+gap*(i+.5)};
  });
});
const edgeItems=[];
events.forEach(e => (e.candidate_links || []).forEach(link => {
  const a=positions[Number(e.row)], b=positions[Number(link.to_row)];
  if (!a || !b) return;
  const shared=(link.shared_entities||[]).join(', ');
  const dx=b.x-a.x, dy=b.y-a.y;
  const curve=Math.max(-55,Math.min(55,dx*.12));
  const d=`M ${a.x} ${a.y} Q ${(a.x+b.x)/2+curve} ${(a.y+b.y)/2-18} ${b.x} ${b.y}`;
  const path=el('path',{d,class:'edge'},svg);
  const title=el('title',{},path);
  title.textContent=`Shared ${shared || 'entity'} · ${link.time_gap_seconds}s`;
  const hit=el('path',{d,class:'edge-hit'},svg);
  hit.addEventListener('mouseenter',()=>{path.classList.add('is-active')});
  hit.addEventListener('mouseleave',()=>{if(!path.classList.contains('selected'))path.classList.remove('is-active')});
  edgeItems.push({from:Number(e.row),to:Number(link.to_row),path,hit,shared,gap:link.time_gap_seconds});
}));
const nodeItems=new Map();
events.forEach(e => {
  const p=positions[Number(e.row)], g=el('g',{class:`node flagged ${laneOf(e)}`,tabindex:'0',role:'button'},svg);
  el('circle',{cx:p.x,cy:p.y,r:25},g);
  const id=el('text',{x:p.x,y:p.y+4},g); id.textContent='R'+e.row;
  const label=el('text',{x:p.x,y:p.y+45,class:'event-label'},g);
  label.textContent=String(e.event_type||'event').replaceAll('_',' ').slice(0,24);
  const connected=edgeItems.filter(x=>x.from===Number(e.row)||x.to===Number(e.row));
  const inspect=()=> {
    nodeItems.forEach((node,row)=>node.classList.toggle('is-dim',row!==Number(e.row)&&!connected.some(x=>x.from===row||x.to===row)));
    edgeItems.forEach(x=>{
      const active=x.from===Number(e.row)||x.to===Number(e.row);
      x.path.classList.toggle('is-active',active);
      x.path.classList.toggle('is-dim',!active);
    });
    const relations=connected.length
      ? connected.map(x=>`R${x.from} ↔ R${x.to}: shared ${x.shared||'entity'} (${x.gap}s)`).join('<br>')
      : 'No candidate relationships for this event.';
    document.getElementById('nodeDetails').innerHTML =
      `<strong>Row ${e.row} · ${String(e.event_type)}</strong><br>`+
      `Time: ${String(e.timestamp||'Unavailable')} · User: ${String(e.user)} · IP: ${String(e.source_ip)} · Host: ${String(e.host)}<br>`+
      `Message: ${String(e.message)}<br>Reason: ${(e.reasons||[]).join('; ')}<hr>`+
      `<strong>Candidate relationships</strong><br>${relations}`;
  };
  g.addEventListener('click',inspect);
  g.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();inspect()}});
  nodeItems.set(Number(e.row),g);
});
document.querySelectorAll('.tab[data-tab]').forEach(btn=>btn.addEventListener('click',()=>{
  document.querySelectorAll('.tab[data-tab]').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(btn.dataset.tab).classList.add('active');
  if(btn.dataset.tab==='graph') setTimeout(()=>svg.setAttribute('data-ready','true'),0);
}));
let timer=null,index=0;
const sequence=[...document.querySelectorAll('.sequence-event')];
function showStep(){
  sequence.forEach((n,i)=>n.style.outline=i===index?'2px solid #16a9bd':'none');
  document.getElementById('stepLabel').textContent=sequence.length?`Event ${Math.min(index+1,sequence.length)} of ${sequence.length}`:'No flagged events';
  if(index<sequence.length-1)index++;else{clearInterval(timer);timer=null}
}
document.getElementById('play').onclick=()=>{if(!sequence.length)return;index=0;if(timer)clearInterval(timer);showStep();timer=setInterval(showStep,1100)};
document.getElementById('pause').onclick=()=>{if(timer)clearInterval(timer);timer=null};
</script>
</body></html>
"""

# ============================================================
# 10. MAIN DASHBOARD
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# 11. CYBERATTACK UPLOAD PAGE
# ============================================================

@app.route("/cyberattack")
def cyberattack_home():

    return render_template_string(
        CYBER_PAGE
    )


# ============================================================
# 12. CYBERATTACK FILE ANALYSIS
# ============================================================

@app.route(
    "/cyberattack/analyze",
    methods=["POST"]
)
def cyberattack_analyze():

    uploaded = request.files.get("logfile")

    if not uploaded or not uploaded.filename:

        return render_template_string(
            CYBER_PAGE,
            error="Please select a log file."
        ), 400

    try:

        filename, records = parse_uploaded_file(
            uploaded
        )

        result = investigate_records(records)

        # Keep the latest uploaded analysis available to the graph workspace.
        global LATEST_CYBER_RESULT, LATEST_CYBER_FILENAME
        LATEST_CYBER_RESULT = result
        LATEST_CYBER_FILENAME = filename

        # One upload now leads directly to all three investigation views.
        return redirect("/investigation-graph")

    except Exception as exc:

        return render_template_string(
            CYBER_PAGE,
            error=f"Could not analyze file: {exc}"
        ), 400


# ============================================================
# 13. EXISTING MODEL STATUS API
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({

        "model_loaded": MODEL is not None,

        "runtime_error": RUNTIME_ERROR,

        "data_load_error": DATA_LOAD_ERROR,

        "device": str(DEVICE),

        "threshold": THRESHOLD,

        "test_incidents": len(INCIDENTS),

        "records_per_incident": 11,

        "model_task": (
            "Pairwise same-event relationship candidate scoring"
        ),

        "data_type": "Synthetic held-out test data"

    })


# ============================================================
# 14. EXISTING INCIDENT API
# ============================================================

@app.route("/api/incidents")
def incidents():

    return jsonify([

        {
            "id": incident_id,
            "records": len(rows)
        }

        for incident_id, rows in sorted(
            INCIDENTS.items()
        )

    ])


# ============================================================
# 15. EXISTING INCIDENT SCORING API
# ============================================================

@app.route("/api/incident/<int:incident_id>")
def incident(incident_id):

    try:

        rows, edges = score_incident(
            incident_id
        )

        safe_rows = []

        for index, row in enumerate(rows):

            safe_rows.append({

                "index": index,

                "id": row["evidence_id"],

                "text": row["text"],

                "timestamp": row["claimed_timestamp"],

                "type": (
                    "Distractor"
                    if int(row["event_id"]) == -1
                    else "Evidence"
                ),

                "image": row["image"]

            })

        return jsonify({

            "incident_id": incident_id,

            "threshold": THRESHOLD,

            "records": safe_rows,

            "edges": edges,

            "candidate_links": sum(
                bool(edge["related"])
                for edge in edges
            ),

            "pairs_scored": len(edges),

            "note": (
                "Synthetic model scores are not proof "
                "or authenticity judgments."
            )

        })

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 400


# ============================================================
# 16. EXISTING MODEL METRICS API
# ============================================================

@app.route("/api/metrics")
def metrics():

    if not TEST_REPORT_PATH.exists():

        return jsonify({
            "available": False
        })

    try:

        data = json.loads(
            TEST_REPORT_PATH.read_text(
                encoding="utf-8"
            )
        )

        return jsonify({

            "available": True,

            "metrics": data.get(
                "siamese_multimodal",
                {}
            )

        })

    except Exception as exc:

        return jsonify({

            "available": False,

            "error": str(exc)

        }), 500


# ============================================================
# 17. INVESTIGATION GRAPH ROUTE
# ============================================================

@app.route("/investigation-graph")
def investigation_graph():

    # Prefer the graph generated from the most recently uploaded file.
    if LATEST_CYBER_RESULT is not None:
        return render_template_string(
            INVESTIGATION_VIEW,
            result=LATEST_CYBER_RESULT,
            filename=LATEST_CYBER_FILENAME or "uploaded logs"
        )

    # Preserve compatibility with the previously generated static graph.
    if CYBER_GRAPH_PATH.exists():
        return send_from_directory(
            str(CYBER_DIR),
            CYBER_GRAPH_PATH.name
        )

    return redirect("/cyberattack")


# ============================================================
# 18. CYBERATTACK REPORT API
# ============================================================

@app.route("/api/cyberattack/report")
def cyberattack_report():

    if not CYBER_REPORT_PATH.exists():

        return jsonify({

            "available": False,

            "error": "Investigation report not found."

        }), 404

    return jsonify({

        "available": True,

        "report": json.loads(
            CYBER_REPORT_PATH.read_text(
                encoding="utf-8"
            )
        )

    })


# ============================================================
# 19. CYBERATTACK TIMELINE API
# ============================================================

@app.route("/api/cyberattack/timeline")
def cyberattack_timeline():

    if not CYBER_TIMELINE_PATH.exists():

        return jsonify({

            "available": False,

            "error": "Temporal correlation file not found."

        }), 404

    return jsonify({

        "available": True,

        "timeline": json.loads(
            CYBER_TIMELINE_PATH.read_text(
                encoding="utf-8"
            )
        )

    })


# ============================================================
# 20. CYBERATTACK STATUS API
# ============================================================

@app.route("/api/cyberattack/status")
def cyberattack_status():

    return jsonify({

        "report_available": CYBER_REPORT_PATH.exists(),

        "timeline_available": CYBER_TIMELINE_PATH.exists(),

        "graph_available": CYBER_GRAPH_PATH.exists(),

        "upload_analysis": True,

        "upload_formats": [
            ".csv",
            ".json",
            ".jsonl",
            ".log",
            ".txt"
        ],

        "max_upload_mb": 10,

        "detection_method": (
            "Heuristic rules and entity-based temporal correlation"
        ),

        "warning": (
            "Flags and links are investigation leads, "
            "not proof of compromise or causation."
        )

    })


# ============================================================
# 21. FILE SIZE ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def too_large(error):

    return jsonify({

        "error": "Uploaded file exceeds the 10 MB limit."

    }), 413


# ============================================================
# 22. START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 55)

    print("TRACE-X MULTIMODAL EVIDENCE INVESTIGATION")

    print("=" * 55)

    print(
        "Main dashboard: "
        "http://127.0.0.1:5000/"
    )

    print(
        "Cyberattack investigator: "
        "http://127.0.0.1:5000/cyberattack"
    )

    print(
        "Investigation graph: "
        "http://127.0.0.1:5000/investigation-graph"
    )

    print("=" * 55)

    print(
        "Existing Siamese checkpoint is loaded if available."
    )

    print(
        "No training or checkpoint modification is performed."
    )

    print(
        "Cyberattack upload analysis currently uses heuristic rules."
    )

    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )