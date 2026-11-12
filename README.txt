# TRACE-X local dashboard

This is a local demonstration UI for the existing TRACE-X V1 Siamese checkpoint.
It does not train or modify any model.

## 1. Copy into your existing project

Copy these items into the root of your existing `TRACE-X` folder:

- `app.py`
- `templates/index.html`

The dashboard expects the existing folders and files:

- `data/synthetic_v1/test.jsonl`
- `features/v1/vocab.json`
- `models/v1/trace_siamese_multimodal_v1.pt`
- `results/v1/trace_siamese_multimodal_v1_validation.json`

Optional held-out test metrics:
- `results/v1/trace_siamese_multimodal_v1_test.json`

## 2. Install Flask if needed

```powershell
python -m pip install flask
```

NumPy, PyTorch and scikit-learn are already part of the TRACE-X environment used for model training. If the environment is different, install them too:

```powershell
python -m pip install numpy torch scikit-learn
```

## 3. Start dashboard

From the project root:

```powershell
python app.py
```

Open:

http://127.0.0.1:5000

## What the model is actually solving

The current V1 model scores pairs of evidence records and predicts whether they belong to the same synthetic event. It compares text, image and activity embeddings using a Siamese neural network.

It does NOT currently:
- verify evidence authenticity;
- reverse engineer binaries or malware;
- prove causation or wrongdoing;
- reconstruct a verified real-world timeline;
- establish real-world forensic accuracy.

Important V1 limitation: the synthetic activity generator encodes event order into activity feature 0. The modality audit found activity-only AP/AUC of 1.0, so the near-perfect combined score is shortcut-sensitive. Disclose this during evaluation and describe V1 as a synthetic proof of concept.

## Demo controls

- Select an incident from the held-out synthetic test split.
- Click Analyze incident.
- Review evidence records.
- Inspect the graph: green edges meet the saved validation threshold.
- Review ranked pair scores.

The app binds to 127.0.0.1 only and is intended for local classroom demonstration.
