concisely describe the pipeline of the following script: from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
import numpy as np
from updated_data import combine_data
from sklearn.preprocessing import StandardScaler

# Setup synthetic data
setting = {'duration_name': 'duration', 'event_name': 'event'}
data_obj = combine_data(setting=setting, current_test_split=0, splits=1)

X_tensor = data_obj.f_tensor
y_tensor = data_obj.s_tensor
X = X_tensor.numpy()
dur = y_tensor[:, 0].numpy()
event = y_tensor[:, 1].numpy()

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Log-transform durations
dur_log = np.log1p(dur)

# Stratified KFold setup
# Stratify based on the binary censoring indicator (event occurred or not)
strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
c_index_scores = []

print("\n--- Stratified 5-Fold CV (XGBoost Regression) ---")
for fold, (train_index, val_index) in enumerate(strat_kf.split(X, event)):
    print(f"\n Fold {fold + 1}")

    # Train/validation split
    X_tr, X_va = X[train_index], X[val_index]
    y_tr_log = dur_log[train_index]
    y_va_dur = dur[val_index]
    y_va_event = event[val_index]

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_tr, y_tr_log)

    # Predict and back-transform
    y_pred_log = xgb_model.predict(X_va)
    y_pred_dur = np.expm1(y_pred_log)

    # Evaluate using survival-aware C-index
    ci = concordance_index_censored(
        y_va_event.astype(bool),
        y_va_dur,
        -y_pred_dur
    )[0]

    c_index_scores.append(ci)
    print(f"Fold {fold + 1} C-index: {ci:.4f}")

# Summary
print("\n Stratified 5-Fold C-index Results:")
print(f"Mean c-index: {np.mean(c_index_scores):.4f}")
print(f"Std Dev:      {np.std(c_index_scores):.4f}")
