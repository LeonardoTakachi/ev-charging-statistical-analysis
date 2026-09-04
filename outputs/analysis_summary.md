# Analysis Summary

## Dataset
- Valid observations used: **1254**
- Variables: **Charging Duration (hours)** and **Energy Consumed (kWh)**

## Probability
- Mean energy consumption: **42.64 kWh**
- Median energy consumption: **42.69 kWh**
- Standard deviation: **22.41 kWh**
- Probability above the fitted-distribution median threshold: **49.91%**
- Theoretical Normal probability within mean ± 2 standard deviations: **95.45%**
- Empirical share of observations within the same interval: **99.52%**
- Normality diagnostic: D'Agostino-Pearson p-value: 2.076e-09. Reject normality at α=0.05: yes.

> The Normal Distribution calculations are model-based estimates. The Q-Q plot
> and normality test should be considered when deciding whether the assumption
> is appropriate.

## Linear Regression
- Equation: **Energy = 41.55 + (0.59 × Duration)**
- Train R²: **0.0008**
- Test R²: **-0.0024**
- Test MAE: **18.37 kWh**
- Test RMSE: **21.30 kWh**
- Baseline RMSE: **21.31 kWh**

The regression model is only marginally better than the mean-value baseline according to RMSE. The difference is practically negligible, which reinforces the conclusion that charging duration alone does not provide useful predictive power.

## Next Steps
A stronger model should test additional explanatory variables carefully, while avoiding target leakage and validating performance on unseen data.
