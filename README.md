# Heart Disease Risk Model

A machine learning model that estimates a person's probability of having heart disease from common risk factors, trained on the CDC's 2025 health survey of US adults.

It builds on my [Heart Health Pipeline](https://github.com/omathestudio/heart-health-pipeline), which cleans the raw survey data. This project takes that clean data and turns it into a predictive model.

> **Work in progress.** Sections marked _TBC_ are filled in as each step is done.

---

## The problem

- **Task:** binary classification. Has this person ever been told they had coronary heart disease or a heart attack?
- **Target:** `heart_disease` (CDC's `_MICHD` variable).
- **Data:** CDC BRFSS 2025, cleaned by the Heart Health Pipeline.
- **Intended use:** a portfolio and learning project showing how risk factors relate to heart disease. **It is not a diagnostic tool.**

## Features

| Feature | Type | Notes |
|---|---|---|
| `bmi` | Number | |
| `high_bp` | Yes / no | |
| `high_cholesterol` | Yes / no | Blank for people never checked, kept as its own group |
| `diabetes` | Yes / no | Diagnosed diabetes only |
| `inactive` | Yes / no | No exercise in the past 30 days |
| `heavy_drinker` | Yes / no | |
| `age_group` | Ordered category | 13 five-year bands |
| `education` | Ordered category | |
| `income` | Ordered category | About 1 in 5 blank, kept as its own group |
| `sex` | Category | |
| `smoking_status` | Category | Every day / some days / former / never |

## Columns left out, and why

| Column | Reason |
|---|---|
| `heart_attack`, `coronary_hd` | **Leakage.** These two define the target, so including them would let the model "see the answer". |
| `general_health` | **Reverse causation.** People with heart disease rate their health worse *because* of it. Tested later as an experiment. |
| `stroke` | Another outcome of the same disease process, not a risk factor. |
| `state_fips` | Left out of version 1 to keep the model simple. |

## Approach

_TBC_: exploratory analysis, preprocessing, models, evaluation, explainability, fairness.

## Results

_TBC_

## How to run it

1. Run the [Heart Health Pipeline](https://github.com/omathestudio/heart-health-pipeline) first so `heart.db` exists. Keep the two repos in the same parent folder, or set `HEART_DB` to the database path.
2. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Check the data:
   ```bash
   python src/check_data.py
   ```

## Project structure

```
heart-disease-risk-model/
├── src/
│   ├── config.py       # data path, target, features, excluded columns
│   ├── data.py         # load the pipeline's clean data
│   └── check_data.py   # class balance and missing values
├── requirements.txt
└── README.md
```

## Limitations

- **Self-reported survey data.** Undiagnosed cases aren't counted.
- **Associations, not causes.** One survey at one point in time.
- **US adults only**, and California, Mississippi and Nevada aren't in the 2025 data.
