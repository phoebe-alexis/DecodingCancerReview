# Setup
import os, sys, torch as tc
import pandas as pd
import numpy as np
from updated_data import combine_data, Data_Collection
from NN import Simple_Model
from training import train_test
from LRP import calculate_LRP_simple

# Add current dir to path
sys.path.append('.')

# Define dummy settings_tt.py for synthetic data
with open("settings_tt.py", "w") as f:
    f.write("place = 'M'\n")

# Training settings
setting = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'reduce_lr_epochs': [5, 5],
    'training_batch_size': 128,
    'training_device': 'cuda' if tc.cuda.is_available() else 'cpu',
    'factor_hidden_nodes': 1.0,
    'hidden_depth_simple': 2,
    'input_dropout': 0.1,
    'dropout': 0.2
}

# Generate full synthetic dataset
full_data = combine_data({'duration_name': 'duration', 'event_name': 'event'}, current_test_split=0, splits=1)

# Split into CV set (80%) and external validation (20%)
n_total = full_data.nsamples
n_val = int(0.2 * n_total)
val_indices = full_data.random_sequence[:n_val]
cv_indices = full_data.random_sequence[n_val:]

cv_data = Data_Collection(
    full_data.full_data.iloc[cv_indices.numpy()],
    full_data.survival_data.iloc[cv_indices.numpy()],
    full_data.cancer_type.iloc[cv_indices.numpy()],
    current_test_split=0,
    splits=5
)

ext_data = Data_Collection(
    full_data.full_data.iloc[val_indices.numpy()],
    full_data.survival_data.iloc[val_indices.numpy()],
    full_data.cancer_type.iloc[val_indices.numpy()],
    current_test_split=0,
    splits=1
)

# Save initial model weights
os.makedirs("./results", exist_ok=True)
initial_model = Simple_Model(cv_data, setting)
tc.save(initial_model.state_dict(), './results/raw_params.pt')

# 5-Fold CV + LRP
from lifelines.utils import concordance_index

cv_cindices = []

for fold in range(5):
    print(f"\n Fold {fold} ----------------------------")

    cv_data.change_test_set(fold)

    model = Simple_Model(cv_data, setting)
    trained_model = train_test(model, cv_data, setting, fold)

    # Evaluate C-index
    x_test, y_test = cv_data.get_test_set()
    surv_time, surv_event = y_test[:, 0].numpy(), y_test[:, 1].numpy()
    pred = model(x_test).detach().cpu().numpy().squeeze()

    cindex = concordance_index(surv_time, -pred, surv_event)
    cv_cindices.append(cindex)

    print(f"Fold {fold} C-index: {cindex:.4f}")

    # Add LRP device setting and run LRP
    lrp_setting = {**setting, 'LRP_device': setting['training_device']}
    calculate_LRP_simple(trained_model, cv_data, lrp_setting, PATH='./results/LRP/', fold=fold)

# Summary
mean_cindex = np.mean(cv_cindices)
print(f"\n Mean C-index across 5 folds: {mean_cindex:.4f}")
