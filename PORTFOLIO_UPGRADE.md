# Portfolio Upgrade Notes

This repository deliberately separates the original academic solution from the
portfolio evolution.

## Academic baseline

`academic/challenge_sprint3_final.py`

The academic script fits a simple Linear Regression to all valid observations
and evaluates R² on the same data used to fit the model. That is appropriate for
demonstrating the classroom concept, but it is not enough to estimate how well a
model generalizes to unseen data.

## Portfolio version

`src/analysis.py`

The portfolio version adds:

1. Project-relative paths with `pathlib`.
2. Validation of required columns.
3. Numeric coercion, missing-value handling and duplicate removal.
4. Descriptive statistics exported to CSV.
5. Energy-consumption histogram and scatter plot.
6. D'Agostino-Pearson normality diagnostic.
7. Q-Q plot.
8. Reproducible train/test split.
9. Test-set MAE, RMSE and R².
10. Mean-prediction baseline for context.
11. Residual plot.
12. Machine-readable JSON results.
13. Automatically generated Markdown interpretation.
14. Jupyter notebook for exploratory review.
15. Unit tests.
16. GitHub Actions continuous-integration checks.

## Why the changes matter

A portfolio project should demonstrate not only that a model can be fitted, but
also that its assumptions, limitations and predictive performance are evaluated.

The goal is not to manufacture a high R². If charging duration has weak
predictive power, that is itself an analytical finding and a reason to test
additional explanatory variables.
