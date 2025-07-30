import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# === 7. Align features and remove 'Cancer' features ===
common_features = lrp_matrix.columns.intersection(shap_matrix.columns)
common_features = [f for f in common_features if 'cancer' not in f.lower()]

lrp_matrix = lrp_matrix[common_features]
shap_matrix = shap_matrix[common_features]

# === Custom darkened coolwarm colormap ===
deep_coolwarm = LinearSegmentedColormap.from_list(
    "deep_coolwarm",
    [
        (0.0, '#001f4d'),  # deep blue
        (0.5, '#f7f7f7'),  # light gray center
        (1.0, '#8B0000')   # deep red
    ]
)

# === 8. Plot heatmaps with shared colorbar ===
N = min(50, lrp_matrix.shape[0])  # first N samples

# Compute global color scale
combined = np.vstack([lrp_matrix.iloc[:N].values, shap_matrix.iloc[:N].values])
vmin, vmax = combined.min(), combined.max()

# Setup figure with gridspec
fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.3)

# LRP heatmap
ax0 = plt.subplot(gs[0])
sns.heatmap(
    lrp_matrix.iloc[:N],
    ax=ax0,
    cmap=deep_coolwarm,
    center=0,
    vmin=vmin,
    vmax=vmax,
    cbar=False
)
ax0.set_title("LRP Saliency Map")
ax0.set_xlabel("Features")
ax0.set_ylabel("Samples")

# SHAP heatmap
ax1 = plt.subplot(gs[1], sharey=ax0)
sns.heatmap(
    shap_matrix.iloc[:N],
    ax=ax1,
    cmap=deep_coolwarm,
    center=0,
    vmin=vmin,
    vmax=vmax,
    cbar=False
)
ax1.set_title("SHAP Saliency Map")
ax1.set_xlabel("Features")

# Shared colorbar
cax = plt.subplot(gs[2])
sm = plt.cm.ScalarMappable(cmap=deep_coolwarm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label("Attribution Value")

plt.tight_layout()
plt.show()
