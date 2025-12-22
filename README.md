# Flight Delay Type Classification (2015 US Flights)

This project builds a **multi-class classification model** to predict the **dominant delay type** of a flight using the 2015 flights dataset.

## Problem Statement

Given flight schedule and airline info, predict the **main delay category**:

- `NO_DELAY`
- `AIR_SYSTEM_DELAY`
- `SECURITY_DELAY`
- `AIRLINE_DELAY`
- `LATE_AIRCRAFT_DELAY`
- `WEATHER_DELAY`

The target label is created from delay component columns by selecting the delay type with the **largest delay minutes** (or `NO_DELAY` if all are zero).

---

## Dataset
-link : https://www.kaggle.com/datasets/usdot/flight-delays
- File: `flights.csv`
- Common columns used:
  - Date features: `YEAR, MONTH, DAY, DAY_OF_WEEK`
  - Schedule features: `SCHEDULED_DEPARTURE, SCHEDULED_ARRIVAL`
  - Categorical: `AIRLINE`
  - Delay components:
    - `AIR_SYSTEM_DELAY`
    - `SECURITY_DELAY`
    - `AIRLINE_DELAY`
    - `LATE_AIRCRAFT_DELAY`
    - `WEATHER_DELAY`

> Note: Many rows contain missing delay components. Missing delay components are treated as **0**.

---

## Approach

### 1) Load & select columns
Only relevant columns are selected from the original dataset for modeling.

### 2) Handle missing values
- Delay component columns are filled with `0`.
- Flights with missing arrival-related values may exist (often cancelled flights). In this project, delay components are still used as the main signal to label delay type.

### 3) Feature engineering
- `all_delay` = sum of all delay components
- `delay_type`:
  - If `all_delay > 0`: label is the column name with the maximum value among delay components
  - Else: `NO_DELAY`

### 4) Encode categorical features
`AIRLINE` is encoded using **one-hot encoding** (`pd.get_dummies`).

### 5) Class imbalance handling
The dataset is highly imbalanced (most flights are `NO_DELAY`).
To address this, we use:

- `RandomOverSampler` (from `imblearn`) to oversample minority classes until all classes have equal counts.

### 6) Model
- `DecisionTreeClassifier` from scikit-learn

---

## Project Structure (suggested)

```
flight-delay-classification/
│── flights.csv
│── notebook.ipynb
│── README.md
│── requirements.txt
```

---

## Installation

### 1) Create environment (optional but recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Requirements

Create a `requirements.txt` like:

```txt
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
```

---

## Usage

### Training
Run the notebook cells in order:

1. Load dataset (`flights.csv`)
2. Clean data / fill missing values
3. Create `delay_type`
4. One-hot encode airlines
5. Apply RandomOverSampler
6. Train DecisionTreeClassifier

### Predicting a new flight
Example input (must match training columns after one-hot encoding):

```python
test_df = pd.DataFrame([{
    'YEAR': 2015,
    'MONTH': 8,
    'DAY': 4,
    'DAY_OF_WEEK': 3,
    'AIRLINE_AA': 0,
    'AIRLINE_AS': 0,
    'AIRLINE_B6': 0,
    'AIRLINE_DL': 1,
    'AIRLINE_F9': 0,
    'AIRLINE_HA': 0,
    'AIRLINE_NK': 0,
    'AIRLINE_OO': 0,
    'AIRLINE_UA': 0,
    'AIRLINE_VX': 0,
    'AIRLINE_WN': 0,
    'SCHEDULED_DEPARTURE': 513,
    'SCHEDULED_ARRIVAL': 720
}])

test_df = test_df.reindex(columns=X_resampled.columns, fill_value=0)
prediction = clf.predict(test_df)
print(prediction)
```

---

## Results

- Training accuracy (on oversampled training set): **~0.99**
- Because oversampling and training evaluation are done on the same dataset in the notebook, this score is **not a reliable real-world performance estimate**.

✅ Recommended next step:
- Use `train_test_split` + `classification_report` + confusion matrix
- Consider `StratifiedKFold` cross-validation

---

## Notes / Known Issues

### `SettingWithCopyWarning`
You may see warnings like:

- `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.`

Fix by using `.copy()` when creating slices, and `.loc[...]` when assigning new columns.

Example:
```python
flights_data = flights_data[['col1', 'col2']].copy()
flights_data.loc[:, 'new_col'] = ...
```

### `DtypeWarning: Columns have mixed types`
When reading `flights.csv`, you may see:
- `DtypeWarning: Columns (...) have mixed types`

Fix by specifying `dtype=` or using:
```python
pd.read_csv("flights.csv", low_memory=False)
```

---

## License

This project is for educational purposes.
