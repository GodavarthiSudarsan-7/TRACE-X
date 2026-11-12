import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

class PairDataset(Dataset):
    def __init__(self, bundle, pairs, labels):
        self.b, self.p, self.y = bundle, torch.from_numpy(pairs), labels
    def __len__(self):
        return len(self.p)
    def __getitem__(self, k):
        i, j = self.p[k].tolist()
        b = self.b
        return (b["text"][i], b["image"][i], b["activity"][i],
                b["text"][j], b["image"][j], b["activity"][j])

class ModalityEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.text_gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.text_projection = nn.Sequential(nn.Linear(64, 48), nn.ReLU())
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,48,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48,48), nn.ReLU())
        self.activity_gru = nn.GRU(6,32,batch_first=True)
        self.activity_projection = nn.Sequential(nn.Linear(32,48), nn.ReLU())
    def forward(self, text, image, activity):
        _, ht = self.text_gru(self.text_embedding(text))
        tz = self.text_projection(torch.cat([ht[-2],ht[-1]],dim=1))
        iz = self.image_encoder(image)
        _, ha = self.activity_gru(activity)
        az = self.activity_projection(ha[-1])
        return tz, iz, az

class SiameseTraceNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = ModalityEncoder(vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(288,192), nn.ReLU(), nn.Dropout(.25),
            nn.Linear(192,96), nn.ReLU(), nn.Dropout(.15), nn.Linear(96,1))
    def forward(self, ta,ia,aa,tb,ib,ab, mode="all"):
        za = list(self.encoder(ta,ia,aa))
        zb = list(self.encoder(tb,ib,ab))
        if mode == "text": za[1]=torch.zeros_like(za[1]); za[2]=torch.zeros_like(za[2]); zb[1]=torch.zeros_like(zb[1]); zb[2]=torch.zeros_like(zb[2])
        elif mode == "image": za[0]=torch.zeros_like(za[0]); za[2]=torch.zeros_like(za[2]); zb[0]=torch.zeros_like(zb[0]); zb[2]=torch.zeros_like(zb[2])
        elif mode == "activity": za[0]=torch.zeros_like(za[0]); za[1]=torch.zeros_like(za[1]); zb[0]=torch.zeros_like(zb[0]); zb[1]=torch.zeros_like(zb[1])
        elif mode == "no_activity": za[2]=torch.zeros_like(za[2]); zb[2]=torch.zeros_like(zb[2])
        elif mode == "no_text": za[0]=torch.zeros_like(za[0]); zb[0]=torch.zeros_like(zb[0])
        elif mode == "no_image": za[1]=torch.zeros_like(za[1]); zb[1]=torch.zeros_like(zb[1])
        parts=[]
        for a,b in zip(za,zb): parts.extend([torch.abs(a-b),a*b])
        return self.classifier(torch.cat(parts,dim=1)).squeeze(1)

def main():
    root=Path(__file__).resolve().parents[2]
    fd=root/"features"/"v1"
    ckpt_path=root/"models"/"v1"/"trace_siamese_multimodal_v1.pt"
    report_path=root/"results"/"v1"/"trace_siamese_multimodal_v1_validation.json"
    out=root/"results"/"v1"/"modality_masking_audit.json"
    vocab=json.loads((fd/"vocab.json").read_text(encoding="utf-8"))
    val=json.loads(report_path.read_text(encoding="utf-8"))
    threshold=float(val["validation"]["threshold"])
    with np.load(fd/"trace_bench_v1_tensors.npz") as z:
        text=torch.from_numpy(z["test_text_ids"].astype(np.int64))
        image=torch.from_numpy(z["test_images"].astype(np.float32))
        activity=torch.from_numpy(z["test_activities"].astype(np.float32))
        inc=z["test_incident_ids"].astype(np.int64)
        ev=z["test_event_ids"].astype(np.int64)
    groups={}
    for i,x in enumerate(inc.tolist()): groups.setdefault(int(x),[]).append(i)
    pairs=[]; ys=[]
    for ix in groups.values():
        for p,i in enumerate(ix):
            for j in ix[p+1:]:
                pairs.append((i,j)); ys.append(int(ev[i]!=-1 and ev[i]==ev[j]))
    pairs=np.asarray(pairs,dtype=np.int64); y=np.asarray(ys,dtype=np.int64)
    ds=PairDataset({"text":text,"image":image,"activity":activity},pairs,y)
    loader=DataLoader(ds,batch_size=128,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(ckpt_path,map_location=device,weights_only=False)
    model=SiameseTraceNet(int(ck["vocab_size"])).to(device)
    model.load_state_dict(ck["model_state_dict"],strict=True)
    model.eval()
    results={}
    with torch.no_grad():
        for mode in ["all","text","image","activity","no_text","no_image","no_activity"]:
            probs=[]
            for batch in loader:
                ta,ia,aa,tb,ib,ab=[x.to(device) for x in batch]
                probs.extend(torch.sigmoid(model(ta,ia,aa,tb,ib,ab,mode)).cpu().numpy().tolist())
            p=np.asarray(probs)
            pred=(p>=threshold).astype(np.int64)
            results[mode]={
                "roc_auc":float(roc_auc_score(y,p)),
                "average_precision":float(average_precision_score(y,p)),
                "f1_at_validation_threshold":float(f1_score(y,pred,zero_division=0)),
                "note":"Exploratory modality-masking diagnostic; not a separately trained ablation model."
            }
    report={
        "benchmark":"TRACE_Bench_v1",
        "split":"held-out test",
        "threshold_source":"validation report",
        "test_pairs":int(len(y)),
        "positive_pairs":int(y.sum()),
        "negative_pairs":int((y==0).sum()),
        "modality_masking":results,
        "caveat":"Masking modalities at inference is a diagnostic, not a fair substitute for retrained modality ablations. Synthetic shortcut signals may remain."
    }
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("TRACE-X V1 QUICK MODALITY-MASKING AUDIT")
    for name,m in results.items():
        print(f"{name:12s} AP={m['average_precision']:.4f} AUC={m['roc_auc']:.4f} F1={m['f1_at_validation_threshold']:.4f}")
    print(f"Saved: {out}")
    print("No training performed; checkpoint and datasets unchanged.")

if __name__=="__main__": main()
