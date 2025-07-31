# Revised DecodingCancer

This project extends the work done in [DecodingCancer](https://github.com/PhGK/DecodingCancer), aiming to leverage identifiable key variables and interactions.

## Related Work

The [DecodingCancer](https://github.com/PhGK/DecodingCancer) repository is the official codebase supporting the “Decoding pan‑cancer treatment outcomes using multimodal real‑world data and explainable AI” study. It implements deep learning models for survival analysis using multimodal clinical data and applies explainable AI techniques, specifically Layer-wise Relevance Propagation (LRP), to identify and interpret the contributions of individual clinical markers to patient prognosis across multiple cancer types.

This project builds upon it by:

1. streamlining the model using feature selection
2. highlighting other models that consider feature interactions

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

# 1. Feature Selection Training

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

The `select_features.py` script identifies the top-K most important features based on Layer-wise Relevance Propagation (LRP) across all cross-validation folds.

## Retrain Model with LRP-Selected Features

The `retain_selected_features.py` script evaluates how well the model performs when trained on only the top-K features selected using LRP (see select_features_from_lrp.py).

What It Does:

1. Loads the full synthetic dataset from updated_data.py.
2. Filters input features to keep only those selected based on LRP.
3. Reconstructs the dataset with reduced features.
4. Trains a new model from scratch (no weight reuse) over 5 CV folds.
5. Calculates C-index per fold to assess predictive performance.

### Notes:
The model is trained from scratch (no pre-loaded weights) to avoid shape mismatch from feature reduction.

The variable selected_features must be defined from `select_features_from_lrp.py`.

## Use Case:

This helps determine whether fewer, more relevant features (identified via LRP) can maintain or improve model performance, supporting interpretability without sacrificing accuracy.

# 2. Alternative models for Survival prediction

## Extreme Gradient Boosting (XGBoost)

The script `xgboost.py` assesses how well an XGBoost regressor can predict survival durations in a censored dataset, using stratified CV and survival-specific evaluation.

### XGBoost Pipleline

1. Data Preparation
2. Preprocessing
3. Cross-Validation Setup
4. Model Training & Evaluation (per fold)
5. Results Aggregation

## Transformer Models

# 3. Expanding the Explainability Framework

The `saliency.py`and `normalized_saliency.py` scripts compare two feature attribution methods—LRP (Layer-wise Relevance Propagation) and SHAP (SHapley Additive exPlanations).

1. Load LRP Results
2. Get SHAP Sample Names
3. Map Real Sample Names
4. Pivot LRP Matrix
5. Build SHAP Matrix
6. Align Samples
7. Align and Filter Features
8. Select Top 25 Features
9. Normalize Rows
10. Compute Difference
11. Define Custom Colormap
12. Plot Saliency Maps

