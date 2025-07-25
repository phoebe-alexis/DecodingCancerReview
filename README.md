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
