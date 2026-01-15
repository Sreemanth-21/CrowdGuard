import torch
import torch.nn as nn
import numpy as np
from anomaly_detector import AutoEncoder

data = np.load("data/processed/normal_features.npy")

X = torch.tensor(data, dtype=torch.float32)

model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for _ in range(100):
    optimizer.zero_grad()
    recon = model(X)
    loss = criterion(recon, X)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/edge/anomaly_model.pt")
