# Ice Cream Revenue Prediction

Predicting ice cream shop revenue from daily temperature, using regression models in scikit-learn. This project compares 9 different models with cross-validation and full residual diagnostics, then saves the best one for reuse.

## Project Files

| File | Description |
|---|---|
| `Icecream_Revenue_Extended.ipynb` | Main notebook — full EDA, model comparison, diagnostics (executed, with outputs/plots embedded) |
| `full_analysis.py` | Same analysis as a plain `.py` script |
| `IceCreamData.csv` | Dataset — `Temperature` (°C) and `Revenue` columns, 500 rows |
| `best_icecream_model.pkl` | Best-performing model, saved with `joblib` |
| `icecream_prediction_standalone.py` | Self-contained script that generates its own synthetic data (no CSV needed) |
| `plots_*.png` | Exported charts (distributions, scatter, boxplots, model comparison, actual-vs-predicted, residuals) |

## Dataset

- **Temperature** — daily temperature in °C
- **Revenue** — ice cream shop revenue for that day
- 500 rows, no missing values (some duplicates/outliers removed during cleaning)

## How to Run

### Notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook Icecream_Revenue_Extended.ipynb
```
Run all cells top to bottom. Requires `IceCreamData.csv` in the same folder.

### Script
```bash
python full_analysis.py
```
Requires `dataset/IceCreamData.csv` relative to the script (or edit the path).

### Standalone script (no dataset required)
```bash
python icecream_prediction_standalone.py
```

## Approach

1. **EDA** — shape, dtypes, summary stats, missing/duplicate checks, correlation, distribution plots, scatter plot with regression line, boxplots for outliers.
2. **Cleaning** — outliers removed using the IQR method (500 → 496 rows).
3. **Modeling** — 9 regressors trained and compared:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Polynomial Regression (degree 2)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - KNN Regressor
   - SVR (RBF kernel)
4. **Evaluation** — MAE, MSE, RMSE, MAPE, R², plus 5-fold cross-validated R² for every model.
5. **Diagnostics** — model comparison bar chart, actual-vs-predicted scatter, residuals-vs-predicted, and residual distribution for the best model.
6. **Persistence** — best model saved to `best_icecream_model.pkl`; a `predict_revenue()` helper wraps it for new inputs.

## Results

The relationship between temperature and revenue is close to linear, so linear-type models (Linear / Ridge / Lasso / Polynomial) performed best:

| Metric | Value (best model) |
|---|---|
| R² (test) | ≈ 0.98 |
| MAE | ≈ 19 |
| MAPE | ≈ 4–6% |

More complex models (Random Forest, Gradient Boosting, SVR) did not outperform the linear models here — a sign the underlying relationship genuinely doesn't need extra complexity.

## Example: Predicting Revenue for a New Temperature

```python
import joblib
import pandas as pd

model = joblib.load("best_icecream_model.pkl")

def predict_revenue(temperature_celsius, model=model):
    temp_df = pd.DataFrame({"Temperature": [temperature_celsius]})
    return model.predict(temp_df)[0]

print(predict_revenue(30))  # e.g. -> ~707
```

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

## Notes

- The original source notebook loaded data from a GitHub URL; this project uses the locally uploaded `IceCreamData.csv` instead.
- `icecream_prediction_standalone.py` is a separate, simpler script for cases where no dataset file is available — it generates its own synthetic data with a similar Temperature→Revenue relationship.
