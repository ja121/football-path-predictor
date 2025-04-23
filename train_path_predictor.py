import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ==== CONFIG ====
DATASET_FILE = "path_dataset.csv"
EPOCHS = 50
BATCH_SIZE = 32
HISTORY_LENGTH = 5  # 5 positions = 10 features (x0–x4, y0–y4)
LEARNING_RATE = 0.001

# ==== LOAD DATA ====
df = pd.read_csv(DATASET_FILE)

# Make sure we're only using 10 input features (5 positions × 2)
X = df.iloc[:, :HISTORY_LENGTH * 2].values  # x0, x1, ..., y4 (10 values)
y = df.iloc[:, -2:].values  # target_x, target_y

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== DATASET WRAPPER ====
class PathDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = PathDataset(X_train, y_train)
val_ds = PathDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE)

# ==== MODEL ====
model = nn.Sequential(
    nn.Linear(HISTORY_LENGTH * 2, 128),  # input size = 10
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2)  # output = (target_x, target_y)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== TRAIN LOOP ====
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            val_loss += criterion(output, y_batch).item() * X_batch.size(0)

    print(f"Epoch {epoch:02d} | Train Loss: {total_loss / len(train_ds):.4f} | Val Loss: {val_loss / len(val_ds):.4f}")

# ==== SAVE MODEL ====
torch.save(model.state_dict(), "path_predictor_model.pth")
print("✅ Model saved to path_predictor_model.pth")
