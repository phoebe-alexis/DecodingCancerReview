import pandas as pd
import numpy as np
import torch as tc
from updated_data import combine_data, Data_Collection
from NN import Simple_Model
from training import train_test
import os

# ---------------------------------------
# Step 1: Load and concatenate LRP scores
# ---------------------------------------
lrp_dfs = []
for fold in range(5):
    path = f'./results/LRP/LRP_Simple Model_scores_input_{fold}.csv'
    df = pd.read_csv(path)
    lrp_dfs.append(df)

full_lrp = pd.concat(lrp_dfs, axis=0)

# ---------------------------------------
# Step 2: Aggregate relevance by feature
# ---------------------------------------
agg_lrp = full_lrp.groupby('therapy_diagnostics_antero')['LRP'].apply(lambda x: x.abs().mean())
agg_lrp_sorted = agg_lrp.sort_values(ascending=False)
import math

# ---------------------------------------
# Step 3: Select top-k features based on LRP mean feature count
# ---------------------------------------

# Round mean to nearest integer
TOP_K = int(round(lrp_feature_counts.mean()))

# Get top-K most important features from aggregated LRP scores
selected_features = agg_lrp_sorted.head(TOP_K).index.tolist()

print(f"\n Top {TOP_K} selected features based on LRP:")
print(selected_features)
