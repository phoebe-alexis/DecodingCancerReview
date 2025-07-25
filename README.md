# Revised DecodingCancer

This project extends the work done in [DecodingCancer](https://github.com/PhGK/DecodingCancer), aiming to [briefly describe your specific goal or contribution].

## Related Work

The original repository [DecodingCancer](https://github.com/PhGK/DecodingCancer) implements [briefly describe what it does].  
This project builds upon it by:

- streamlining the model using feature selection
- highlighting other models that consider feature interactions

## Reproducibility

To ensure reproducibility, ensure that you set a seed before you begin.

## Installation

Clone this repository and install dependencies:

```bash
!git clone https://github.com/PhGK/DecodingCancer.git
%cd DecodingCancer
```

## Synthetic Data Generator

The file `updated_data.py` generates a synthetic dataset with:

- 50 numerical features
- 5 cancer types (mutually exclusive)
- Survival time and event status

This is adapted from `data.py`in the original data.

### How to Use

```python
from updated_data import combine_data

setting = {
    'duration_name': 'duration',
    'event_name': 'event'
}
data = combine_data(setting, current_test_split=0, splits=5)

X_train, y_train = data.get_train_set()
X_test, y_test = data.get_test_set()
```

## Feature Selection Training

The file `updated_training.py` adapts the model so that it only considers selected features.

## Configuration

The script creates a minimal config file settings_tt.py that defines a single variable:

```python
#Create a dummy settings_tt.py with 'place' defined as 'M'
with open("settings_tt.py", "w") as f:
    f.write("place = 'M'\n")
```

## Full Pipeline

The `run_full_pipeline.py` script runs the complete pipeline using synthetic data:

1. Generates synthetic data with 50 features and 5 cancer types.
2. Splits data into:

- 80% for cross-validation (CV)
- 20% for external validation

3. Trains a simple neural Cox model over 5 folds.
4. Evaluates performance using the Concordance Index.
5. Applies Layer-wise Relevance Propagation (LRP) for interpretability.
6. Summarizes average model performance across folds.

## LRP Feature Contribution Analysis

The `analyze_lrp_contributions.py script` analyzes the LRP scores from `./results/LRP/` to answer:

How many features contribute to 90% of the model’s prediction relevance for each sample?

Key Steps:

1. Loads LRP scores from a CSV.
2. Sorts features per sample by relevance.
3. Calculates cumulative LRP % for each feature.
4. Counts how many features are needed to reach 90% cumulative relevance.
5. Prints summary stats (mean, median, min, max).
6. Plots a histogram showing the distribution across patients.

## Feature Selection via LRP
