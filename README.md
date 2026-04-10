# Personal Loan Campaign — ML Targeting Model

Supervised classification model to identify AllLife Bank liability customers most likely to convert on a personal loan offer. Reduces cost-per-acquisition by concentrating marketing spend on the highest-probability targets.

## Business Problem
AllLife Bank's prior campaign achieved ~9.5% conversion across the full base. A predictive model lets the next campaign target a high-probability subset, improving conversion rates and cutting wasted spend.

## Dataset
- `Loan_Modelling.csv` — 5,000 rows × 14 columns
- Target: `Personal_Loan` (binary, 9.5% positive class)
- Key features: `Income`, `CCAvg`, `CD_Account`, `Education`, `Family`, `Mortgage`

## Approach
1. EDA — no missing values; identified negative `Experience` values as data entry errors; removed `ID`
2. Feature analysis — correlation heatmap and bivariate plots to rank predictors
3. Preprocessing — one-hot encoding, 70/30 train/test split; noted ~90.5% class imbalance
4. Modeling — Decision Tree variants: default, class-weight balanced, pre-pruning, post-pruning (Cost Complexity Pruning)
5. Model selection — optimised for recall on the minority class; pruned tree selected

## Results

| Model | Accuracy | Recall | Precision | F1 |
|-------|----------|--------|-----------|-----|
| Decision Tree (default) | 98.4% | 87.9% | 95.6% | 91.6% |
| Decision Tree (balanced weights) | 97.5% | 88.6% | 86.8% | 87.7% |
| Post-Pruning (CCP) | 94.9% | **99.3%** | 66.1% | 79.4% |

**Selected model:** Post-pruned CCP tree — catches 99%+ of true converters. Acceptable precision trade-off when the marginal cost of an extra mailer is low.

## Key Findings
- **Top features by importance:** Income (0.877), CCAvg/credit card spend (0.067), Family size (0.057)
- **Top correlations with target:** Income (r=0.50), CCAvg (r=0.37), CD_Account (r=0.32)
- **Decision rules:** Customers with `Income > $92.5k` OR (`CCAvg > $2.95` AND `CD_Account = 1`) are the strongest conversion candidates
- Class 0 accounts for 90.5% of the dataset — imbalance is preserved in train/test splits

## Recommendations
- Use the recall-optimised model to generate a priority target list; A/B test offer creative on a held-out cohort
- Customers near the boundary (Income ~$85–95k) are worth enriching with third-party data
- Retrain quarterly with updated campaign outcomes

## Technologies
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Code
Notebook: [`Personal Loan Campaign.ipynb`](<Personal Loan Campaign.ipynb>)

---
*Author: Chinmay Rozekar*
