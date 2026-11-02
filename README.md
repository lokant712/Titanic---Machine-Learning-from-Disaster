# Titanic - Machine Learning from Disaster

Predicting passenger survival on the RMS Titanic for the Kaggle [Titanic competition](https://www.kaggle.com/competitions/titanic), using a scikit-learn pipeline and a soft-voting ensemble.

**Kaggle notebook:** https://www.kaggle.com/code/lokanths/lab-da-9-10

## Project Overview

| Item | Details |
|------|---------|
| Task | Binary classification (`Survived`: 0 = died, 1 = survived) |
| Dataset | Kaggle Titanic: `train.csv` (891 passengers), `test.csv` (418 passengers) |
| Language | Python 3 |
| Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn |
| Final model | Soft-voting ensemble of Logistic Regression, Random Forest, Gradient Boosting and SVM |

## Repository Contents

```
.
├── lab-da-9-10.ipynb   # Main notebook: EDA, features, models, submission
└── README.md           # Project description (this file)
```

The dataset is not included. Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/competitions/titanic/data).

## Methodology

1. **Data loading** - read `train.csv` and `test.csv`.
2. **Exploratory analysis** - missing values and survival rates by sex, class, age and port of embarkation, with plots.
3. **Feature engineering** (no survival labels are used, so there is no label leakage)
   - `Title` taken from the name (Mr, Mrs, Miss, Master, Rare)
   - `Age` filled with the median age of the same title and class
   - `IsChild`, `FamilySize`, `IsAlone`
   - `TicketGroup`, `GroupSize`, `SmallGroup` (groups of 2 to 4 survived most often)
   - `FarePerPerson`, `Deck`, `HasCabin`
4. **Preprocessing pipeline** - imputation, scaling and one-hot encoding inside a scikit-learn `Pipeline`.
5. **Model comparison** - five models compared with 5-fold x 3 repeated stratified cross-validation.
6. **Final model** - soft-voting ensemble (averages the predicted probabilities of four models).
7. **Evaluation** - hold-out accuracy, classification report and confusion matrix.
8. **Feature importance** and **submission file** (`submission.csv`).

## Results

| Model | Cross-validation accuracy |
|-------|---------------------------|
| Logistic Regression | 0.826 |
| Random Forest | 0.831 |
| Gradient Boosting | 0.838 |
| SVM | 0.827 |
| **Voting Ensemble (final)** | **0.834** |

| Metric | Value |
|--------|-------|
| Hold-out accuracy (20% split) | 0.827 |
| Kaggle public leaderboard score | _fill in after submitting_ |

Cross-validation accuracy is higher than the leaderboard score because the test set is small (418 passengers), so the leaderboard figure is noisier and typically lower.

## Key Findings

- Women survived far more often than men (about 74% vs 19%).
- First-class passengers survived more often than third-class passengers (63% vs 24%).
- Passengers travelling in small groups (2 to 4 people) survived at about 55%, compared with 27% for everyone else.
- Sex, title and passenger class are the strongest predictors.

## How to Run

### On Kaggle
1. Open the [notebook](https://www.kaggle.com/code/lokanths/lab-da-9-10) and click **Edit** (or **Copy & Edit**).
2. Click **Run All**.
3. The file `submission.csv` appears in the **Output** tab.

### Locally
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook lab-da-9-10.ipynb
```
Place `train.csv` and `test.csv` in the same folder as the notebook. The data loading cell checks the current folder automatically.

## Future Improvements

- Tune hyperparameters with GridSearchCV
- Try XGBoost or LightGBM
- Add out-of-fold family survival features

## Team Members

| Name | Register No. |
|------|--------------|
| Lokanth | 23MID0037 |
| Theebak S | 23MID0227 |
| Balasubramanian | 23MID0420 |
| Hari Prasad | 23MIC0022 |


## Acknowledgements

Dataset provided by [Kaggle](https://www.kaggle.com/competitions/titanic).
