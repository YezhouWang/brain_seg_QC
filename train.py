import torch
from torch.utils.data import DataLoader
from models.unet3d import UNet3D
from dataset import BrainSegQCDataset
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

# Paths
T1_DIR = "data/t1"
INIT_SEG_DIR = "data/seg_init"
QC_SEG_DIR = "data/seg_qc"

# Dataset and loader
train_ds = BrainSegQCDataset(T1_DIR, INIT_SEG_DIR, QC_SEG_DIR)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

# Model
model = UNet3D(in_channels=2, out_channels=3).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for x, y in tqdm(train_dl):
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {epoch_loss / len(train_dl):.4f}")
  
