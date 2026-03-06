import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

print("Loading sequences...")

X = np.load("X_norm.npy")
y = np.load("y_norm.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Dataset
# ==========================

class DrowsyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = DrowsyDataset(X_train, y_train)
test_dataset = DrowsyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# ==========================
# LSTM Model
# ==========================

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=6,
            hidden_size=64,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(0.4)

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.dropout2 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x = x[:, -1, :]

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x.squeeze()


model = LSTMModel()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print("Device:", device)

# ==========================
# Training Setup
# ==========================

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5   # L2 regularization
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5
)

EPOCHS = 50
PATIENCE = 7

train_losses = []
val_losses = []

best_loss = float("inf")
early_stop_counter = 0

# ==========================
# Training Loop
# ==========================

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # ======================
    # Validation
    # ======================

    model.eval()

    total_val_loss = 0
    preds = []
    true = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item()

            preds.extend(outputs.cpu().numpy())
            true.extend(y_batch.cpu().numpy())

    val_loss = total_val_loss / len(test_loader)
    val_losses.append(val_loss)

    preds = np.array(preds)
    preds = (preds > 0.5).astype(int)

    acc = accuracy_score(true, preds)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Accuracy {acc:.4f}")

    # ======================
    # Early Stopping
    # ======================

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_drowsiness_model.pth")

    else:
        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:
        print("Early stopping triggered.")
        break


# ==========================
# Final Evaluation
# ==========================

print("\nClassification Report")
print(classification_report(true, preds))

print("\nConfusion Matrix")
print(confusion_matrix(true, preds))


# ==========================
# Plot Loss Curves
# ==========================

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.show()