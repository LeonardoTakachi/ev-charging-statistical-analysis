"""EV charging statistical analysis — portfolio version of FIAP Sprint 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, normaltest, probplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "ev_charging_patterns.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ASSET_DIR = PROJECT_ROOT / "assets"

ENERGY_COLUMN = "Energy Consumed (kWh)"
DURATION_COLUMN = "Charging Duration (hours)"
REQUIRED_COLUMNS = [ENERGY_COLUMN, DURATION_COLUMN]


def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DEFAULT_DATASET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place ev_charging_patterns.csv in data/."
        )
    data = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    clean = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.dropna().drop_duplicates()


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data[REQUIRED_COLUMNS].describe().T


def classify_probability(probability: float) -> str:
    if probability < 0.10:
        return "rare"
    if probability < 0.40:
        return "unlikely"
    if probability < 0.90:
        return "likely"
    return "almost certain"


def normality_check(series: pd.Series) -> dict:
    values = series.dropna().to_numpy()
    if len(values) < 8:
        return {
            "test": "D'Agostino-Pearson",
            "available": False,
            "reason": "At least 8 observations are required.",
        }
    statistic, p_value = normaltest(values)
    return {
        "test": "D'Agostino-Pearson",
        "available": True,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": 0.05,
        "reject_normality_at_5_percent": bool(p_value < 0.05),
    }


def probability_analysis(data: pd.DataFrame) -> dict:
    series = data[ENERGY_COLUMN]
    mean = float(series.mean())
    median = float(series.median())
    std = float(series.std(ddof=1))
    if np.isclose(std, 0):
        raise ValueError("Standard deviation is zero.")

    above_median = float(1 - norm.cdf(median, loc=mean, scale=std))
    lower, upper = mean - 2 * std, mean + 2 * std
    within_2std = float(
        norm.cdf(upper, loc=mean, scale=std)
        - norm.cdf(lower, loc=mean, scale=std)
    )
    empirical = float(((series >= lower) & (series <= upper)).mean())

    return {
        "mean_kwh": mean,
        "median_kwh": median,
        "std_kwh": std,
        "probability_above_median": above_median,
        "probability_above_median_percent": above_median * 100,
        "probability_above_median_classification": classify_probability(above_median),
        "mean_minus_2std_kwh": lower,
        "mean_plus_2std_kwh": upper,
        "probability_within_2std": within_2std,
        "probability_within_2std_percent": within_2std * 100,
        "probability_within_2std_classification": classify_probability(within_2std),
        "empirical_within_2std": empirical,
        "empirical_within_2std_percent": empirical * 100,
        "normality_test": normality_check(series),
    }


def train_linear_regression(
    data: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> dict:
    if len(data) < 5:
        raise ValueError("At least 5 valid rows are required for train/test evaluation.")

    X = data[[DURATION_COLUMN]]
    y = data[ENERGY_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression().fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    baseline = np.full(len(y_test), y_train.mean())

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "test_predictions": test_pred,
        "residuals": y_test.to_numpy() - test_pred,
        "intercept": float(model.intercept_),
        "coefficient": float(model.coef_[0]),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "rmse": float(mean_squared_error(y_test, test_pred) ** 0.5),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "baseline_rmse": float(mean_squared_error(y_test, baseline) ** 0.5),
        "test_size": test_size,
        "random_state": random_state,
    }


def save_exploratory_tables(raw: pd.DataFrame) -> None:
    numeric = raw.select_dtypes(include="number")
    if ENERGY_COLUMN in numeric.columns:
        corr = (
            numeric.corr(numeric_only=True)[ENERGY_COLUMN]
            .drop(labels=[ENERGY_COLUMN], errors="ignore")
            .dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .rename("correlation_with_energy")
        )
        corr.to_csv(OUTPUT_DIR / "numeric_correlations.csv", header=True)

    for column, filename in [
        ("Charger Type", "energy_by_charger_type.csv"),
        ("Time of Day", "energy_by_time_of_day.csv"),
    ]:
        if column in raw.columns:
            raw.groupby(column)[ENERGY_COLUMN].agg(
                ["count", "mean", "median", "std"]
            ).sort_values("mean", ascending=False).to_csv(OUTPUT_DIR / filename)


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ASSET_DIR / f"{name}.svg")
    plt.close(fig)


def create_charts(raw: pd.DataFrame, data: pd.DataFrame, regression: dict) -> None:
    series = data[ENERGY_COLUMN]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(series, bins=25, alpha=0.75)
    ax.axvline(series.mean(), linestyle="--", label=f"Mean: {series.mean():.2f}")
    ax.axvline(series.median(), linestyle=":", label=f"Median: {series.median():.2f}")
    ax.set(title="Distribution of Energy Consumed", xlabel="Energy (kWh)", ylabel="Frequency")
    ax.legend(); ax.grid(alpha=0.2)
    save_figure(fig, "energy_distribution")

    model = regression["model"]
    ordered = data[[DURATION_COLUMN]].sort_values(DURATION_COLUMN)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[DURATION_COLUMN], data[ENERGY_COLUMN], alpha=0.45)
    ax.plot(ordered[DURATION_COLUMN], model.predict(ordered), linewidth=2)
    ax.set(title="Linear Regression: Duration → Energy", xlabel="Charging Duration (hours)", ylabel="Energy Consumed (kWh)")
    ax.grid(alpha=0.2)
    save_figure(fig, "linear_regression")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(regression["test_predictions"], regression["residuals"], alpha=0.6)
    ax.axhline(0, linestyle="--")
    ax.set(title="Residual Analysis — Test Set", xlabel="Predicted Energy (kWh)", ylabel="Residual")
    ax.grid(alpha=0.2)
    save_figure(fig, "residuals")

    fig, ax = plt.subplots(figsize=(8, 8))
    probplot(series, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot — Energy Consumed")
    save_figure(fig, "energy_qq_plot")

    numeric = raw.select_dtypes(include="number").dropna(axis=1, how="all")
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=8)
    ax.set_title("Correlation Matrix — Numeric Variables")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    save_figure(fig, "numeric_correlation_matrix")

    if "Charger Type" in raw.columns:
        frame = raw[["Charger Type", ENERGY_COLUMN]].dropna()
        groups = [g[ENERGY_COLUMN].to_numpy() for _, g in frame.groupby("Charger Type")]
        labels = [str(k) for k, _ in frame.groupby("Charger Type")]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(groups, tick_labels=labels)
        ax.set(title="Energy Consumed by Charger Type", xlabel="Charger Type", ylabel="Energy (kWh)")
        ax.grid(alpha=0.2)
        save_figure(fig, "energy_by_charger_type")

    if "Time of Day" in raw.columns:
        summary = raw.groupby("Time of Day")[ENERGY_COLUMN].mean().dropna().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(summary.index.astype(str), summary.values)
        ax.set(title="Average Energy Consumed by Time of Day", xlabel="Time of Day", ylabel="Average Energy (kWh)")
        ax.grid(axis="y", alpha=0.2)
        save_figure(fig, "energy_by_time_of_day")


def serializable_regression(result: dict) -> dict:
    return {
        "intercept": result["intercept"],
        "coefficient": result["coefficient"],
        "train_r2": result["train_r2"],
        "test_r2": result["test_r2"],
        "mae_kwh": result["mae"],
        "rmse_kwh": result["rmse"],
        "baseline_mae_kwh": result["baseline_mae"],
        "baseline_rmse_kwh": result["baseline_rmse"],
        "test_size": result["test_size"],
        "random_state": result["random_state"],
    }


def run_analysis(dataset_path: Path = DEFAULT_DATASET) -> None:
    ensure_output_directories()
    raw = load_dataset(dataset_path)
    data = clean_data(raw)
    if data.empty:
        raise ValueError("No valid rows remained after cleaning.")

    stats = descriptive_statistics(data)
    probability = probability_analysis(data)
    regression = train_linear_regression(data)

    stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    save_exploratory_tables(raw)
    create_charts(raw, data, regression)

    payload = {
        "valid_rows": len(data),
        "probability_analysis": probability,
        "linear_regression": serializable_regression(regression),
    }
    (OUTPUT_DIR / "analysis_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 72)
    print("EV CHARGING STATISTICAL ANALYSIS")
    print("=" * 72)
    print(f"Valid rows: {len(data)}")
    print(f"Mean energy: {probability['mean_kwh']:.2f} kWh")
    print(f"Test R²: {regression['test_r2']:.4f}")
    print(f"MAE: {regression['mae']:.2f} kWh")
    print(f"RMSE: {regression['rmse']:.2f} kWh")
    print(f"Baseline RMSE: {regression['baseline_rmse']:.2f} kWh")


if __name__ == "__main__":
    run_analysis()
