import torch
import torch.nn as nn
import numpy as np

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AnomalyDetector:
    def __init__(self, model_path, threshold=0.01):
        self.model = AutoEncoder()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.threshold = threshold

    def predict(self, features):
        x = torch.tensor(
            [[
                features["count"],
                features["density"],
                features["avg_distance"],
                features["motion"],
                features["pressure"]
            ]],
            dtype=torch.float32
        )

        with torch.no_grad():
            recon = self.model(x)
            loss = torch.mean((x - recon) ** 2).item()

        return loss, loss > self.threshold
