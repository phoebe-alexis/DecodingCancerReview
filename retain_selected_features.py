import os
import torch as tc
import numpy as np
from lifelines.utils import concordance_index
from NN import Simple_Model

# ---------------------------------------
# Import combine_data & Data_Collection from updated_data.py
# ---------------------------------------
import importlib.util

# Load updated_data.py
spec_data = importlib.util.spec_from_file_location("updated_data", "./updated_data.py")
updated_data = importlib.util.module_from_spec(spec_data)
spec_data.loader.exec_module(updated_data)

combine_data = updated_data.combine_data
Data_Collection = updated_data.Data_Collection

# Load train_test from updated_training.py
spec_train = importlib.util.spec_from_file_location("updated_training", "./updated_training.py")
updated_training = importlib.util.module_from_spec(spec_train)
spec_train.loader.exec_module(updated_training)

train_test = updated_training.train_test

# ---------------------------------------
# Step 4: Load full dataset and filter
# ---------------------------------------
original_data = combine_data({'duration_name': 'duration', 'event_name': 'event'}, current_test_split=0, splits=5)

# Filter columns based on selected features
filtered_full_data = original_data.full_data[selected_features]

# ---------------------------------------
# Step 5: Recreate Data_Collection object with filtered features
# ---------------------------------------
reduced_data = Data_Collection(
    filtered_full_data,
    original_data.survival_data,
    original_data.cancer_type,
    current_test_split=0,
    splits=5
)

# ---------------------------------------
# Step 6: Skip saving/loading weights (we're training from scratch)
# ---------------------------------------
print("\n⚠ Skipping base model weight save — training from scratch")

# ---------------------------------------
# Step 7: Train new model on reduced features (5-fold CV)
# ---------------------------------------
cv_cindices = []

for fold in range(5):
    print(f"\n Fold {fold} — Retraining on reduced features")
    reduced_data.change_test_set(fold)

    model = Simple_Model(reduced_data, setting)

    # Use load_weights=False to avoid shape mismatch
    trained_model = train_test(model, reduced_data, setting, fold, load_weights=False)

    # Evaluate
    x_test, y_test = reduced_data.get_test_set()
    surv_time, surv_event = y_test[:, 0].numpy(), y_test[:, 1].numpy()
    pred = model(x_test).detach().cpu().numpy().squeeze()

    cindex = concordance_index(surv_time, -pred, surv_event)
    cv_cindices.append(cindex)

    print(f"Fold {fold} C-index (reduced): {cindex:.4f}")

# ---------------------------------------
# Step 8: Summary
# ---------------------------------------
mean_cindex = np.mean(cv_cindices)
print(f"\n Mean C-index with LRP-selected features: {mean_cindex:.4f}")
