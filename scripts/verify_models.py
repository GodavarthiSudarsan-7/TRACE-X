import torch
import torch.nn as nn
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"

class TextNet(nn.Module):
    def __init__(self, vocab_size, dim=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hidden, batch_first=True,
                          bidirectional=True)
        self.proj = nn.Linear(hidden * 2, 64)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        z = self.emb(x)
        _, h = self.gru(z)
        feat = torch.tanh(self.proj(torch.cat([h[-2], h[-1]], dim=1)))
        return feat, self.head(feat)


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, 64)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        z = self.conv(x).flatten(1)
        feat = torch.tanh(self.proj(z))
        return feat, self.head(feat)


class ActivityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 32, batch_first=True)
        self.proj = nn.Linear(32, 64)
        self.decoder = nn.Linear(32, 6)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        seq, h = self.gru(x)
        feat = torch.tanh(self.proj(h[-1]))
        return feat, self.decoder(seq), self.head(feat)


class TraceGraphNet(nn.Module):
    def __init__(self, in_dim=193, hidden=96):
        super().__init__()
        self.node = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.15)
        )
        self.msg = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.auth = nn.Linear(hidden, 1)
        self.role = nn.Linear(hidden, 2)
        self.edge = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, edge_index):
        h = self.node(x)
        auth = torch.sigmoid(self.auth(h)).squeeze(1)
        src, dst = edge_index
        agg = torch.zeros_like(h)
        deg = torch.zeros((len(h), 1), device=h.device)
        messages = self.msg(h[src]) * auth[src, None]
        agg.index_add_(0, dst, messages)
        deg.index_add_(0, dst, torch.ones((len(dst), 1), device=h.device))
        h2 = self.update(agg / (deg + 1e-6), h)
        role = self.role(h2)
        a, b = h2[src], h2[dst]
        edge_logits = self.edge(
            torch.cat([a, b, torch.abs(a-b), a*b], dim=1)
        )
        return h2, auth, role, edge_logits


class GraphWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = TraceGraphNet()


models = {
    "text_encoder.pt": TextNet(35),
    "image_encoder.pt": ImageNet(),
    "activity_encoder.pt": ActivityNet(),
    "trace_graph.pt": GraphWrapper()
}

print("TRACE-X STRICT CHECKPOINT LOADING")
print("=" * 45)

all_passed = True

for filename, model in models.items():
    try:
        checkpoint = torch.load(
            MODELS / filename,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        print(f"PASS: {filename}")

    except Exception as error:
        all_passed = False
        print(f"FAIL: {filename}")
        print(error)

print("=" * 45)
print("ALL FOUR MODELS LOADED SUCCESSFULLY"
      if all_passed else "CHECKPOINT COMPATIBILITY ISSUE FOUND")