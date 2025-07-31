import numpy as np
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ============================
# 1. Prepare full synthetic data
# ============================
setting = {'duration_name': 'duration', 'event_name': 'event'}
data_obj = combine_data(setting=setting, current_test_split=0, splits=1)

# Full dataset (no internal splitting)
X_tensor = data_obj.f_tensor
y_tensor = data_obj.s_tensor

X = X_tensor.numpy()
dur = y_tensor[:, 0].numpy()
event = y_tensor[:, 1].numpy()

# Standardize features (same as before)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Log-transform durations
dur_log = np.log1p(dur)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 2. Define Transformer Regressor
# ============================

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # shape: (B, 1, d_model)
        x = self.transformer(x)             # shape: (B, 1, d_model)
        out = self.output(x.squeeze(1))     # shape: (B, 1) → (B,)
        return out.squeeze(1)

# ============================
# 3. 5-Fold Cross Validation
# ============================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_index_scores = []

print("\n--- 5-Fold Cross Validation (Transformer) ---")

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n Transformer Fold {fold + 1}")

    # Split train/val sets
    X_tr = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_tr = torch.tensor(dur_log[train_idx], dtype=torch.float32).to(device)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val_dur = dur[val_idx]
    y_val_event = event[val_idx]

    # Initialize model
    model = TransformerRegressor(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    # Train model
    model.train()
    for epoch in range(30):
        for xb, yb in train_loader:
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate C-index
    model.eval()
    with torch.no_grad():
        y_pred_log = model(X_val).cpu().numpy()
    y_pred_dur = np.expm1(y_pred_log)

    ci = concordance_index_censored(
        y_val_event.astype(bool), y_val_dur, -y_pred_dur
    )[0]
    print(f"Fold {fold + 1} C-index: {ci:.4f}")
    c_index_scores.append(ci)

# ============================
# 4. Report Cross-Validation Summary
# ============================

print("\n Transformer 5-Fold C-index Results")
print(f"Mean c-index: {np.mean(c_index_scores):.4f}")
print(f"Std Dev:      {np.std(c_index_scores):.4f}")
