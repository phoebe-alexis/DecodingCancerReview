import pandas as pd

# Load one LRP result CSV
df = pd.read_csv('./results/LRP/LRP_Simple Model_scores_input_0.csv')

# Step 1: Sort LRP values per sample
df_sorted = df.sort_values(by=['sample_name', 'LRP'], ascending=[True, False])

# Step 2: Compute cumulative LRP % per sample
df_sorted['cum_LRP'] = df_sorted.groupby('sample_name')['LRP'].cumsum()
df_sorted['total_LRP'] = df_sorted.groupby('sample_name')['LRP'].transform('sum')
df_sorted['cum_LRP_pct'] = 100 * df_sorted['cum_LRP'] / df_sorted['total_LRP']

# Step 3: Count how many features reach ≥90% of LRP
def count_90pct(group):
    return (group['cum_LRP_pct'] < 90).sum() + 1

lrp_feature_counts = df_sorted.groupby('sample_name').apply(count_90pct)

# Step 4: Summary stats
print("Variables needed for 90% of LRP per sample:")
print(f"Mean: {lrp_feature_counts.mean():.2f}")
print(f"Median: {lrp_feature_counts.median():.0f}")
print(f"Min: {lrp_feature_counts.min()}")
print(f"Max: {lrp_feature_counts.max()}")

# Optional: Plot histogram
import matplotlib.pyplot as plt
lrp_feature_counts.hist(bins=range(1, lrp_feature_counts.max()+2))
plt.xlabel("Number of features to reach 90% LRP")
plt.ylabel("Number of patients")
plt.title("LRP Feature Contribution Density")
plt.show()
