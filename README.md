# CO₂ Emission Predictor & Recommender

A Streamlit app that estimates a vehicle's tailpipe CO₂ (g/km) from early design specifications and suggests real vehicles with similar specs that emit less.

## Problem

In early vehicle design, engine size, vehicle class and transmission are known long before a prototype can be tested. A design that is likely to miss an emissions target is far cheaper to change early than late.

## Solution

1. **Prediction model.** Tuned XGBoost regression on Natural Resources Canada fuel-consumption ratings (model years 2015–2024, 10,059 vehicles). On vehicle lines the model has never seen: **R² 0.83, RMSE 22.6 g/km, MAE 16.7 g/km.**
2. **Recommender.** The input specs are encoded with the model's own preprocessor and the nearest real vehicles are found with cosine similarity (k-nearest neighbours). Only vehicles whose measured emissions are at least one typical model error (16.7 g/km) below the estimate are shown.

## App pages

| Page | Content |
| --- | --- |
| Overview | Headline metrics and navigation |
| Predict | Spec form, estimate with likely range, position in the dataset |
| Recommend | Lower-emission real vehicles, table and comparison chart |
| Model performance | Validation design, model comparison, error analysis |
| Data explorer | Filterable distribution, engine/fuel, class and trend charts |
| Methodology | Data, features, model, validation, limitations |

## Tech stack

Python 3.13 · pandas, NumPy · scikit-learn, XGBoost, statsmodels · Streamlit, Plotly · joblib

## Project structure

```
├── app.py                       # Streamlit entry point (navigation)
├── views/                       # app pages
├── ui.py  utils.py  recommender.py
├── scripts/export_test_predictions.py
├── notebooks/
│   ├── 01_Cleaned.ipynb
│   ├── 02_EDA.ipynb             # includes repetition, noise-floor and fuel-ratio checks
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model.ipynb           # feature selection, grouped validation, tuning, final model
├── model/                       # trained pipeline, recommender, metrics, held-out predictions
├── datasets/                    # raw, cleaned and feature-engineered CSVs
├── .streamlit/config.toml       # theme
└── requirements.txt
```

## Model evaluation

**Validation design.** Most vehicle lines appear many times in the data (trims, model years, drivetrains); in a random 80/20 split, 99.7% of test rows have a same-line row in the training set. To measure accuracy on a *new* vehicle line, each line (`Make` + first word of `Model`, 443 lines) is kept entirely inside either the training folds or the evaluation set. The test set (20% of lines, 1,841 vehicles) was scored once, after the model had been chosen on cross-validation alone; the final model was then refit on all data.

Untuned defaults, 5-fold grouped cross-validation on the training portion:

| Model | CV R² | CV RMSE (g/km) | CV MAE (g/km) |
| --- | --- | --- | --- |
| Mean predictor (baseline) | -0.048 | 61.4 | 48.6 |
| Ridge | 0.788 | 27.4 | 20.3 |
| Decision Tree | 0.726 | 31.1 | 21.9 |
| Random Forest | 0.791 | 27.3 | 19.4 |
| XGBoost | 0.813 | 25.7 | 18.6 |
| XGBoost (log target) | 0.808 | 26.2 | 19.1 |

After tuning (randomised search, grouped CV): XGBoost 0.832 ± 0.039, XGBoost log-target 0.830, Random Forest 0.819, Ridge 0.792. **Production model: tuned XGBoost**, chosen by the highest cross-validated R²; the top two are within noise of each other.

| Production model | R² | RMSE (g/km) | MAE (g/km) |
| --- | --- | --- | --- |
| Test, unseen vehicle lines | 0.833 | 22.6 | 16.7 |
| Train ≤ 2021 → test 2022–24 | 0.869 | 23.1 | 16.4 |

**Sensitivity to the definition of a vehicle line.** The score depends on how strictly "new vehicle line" is defined. With 5-fold cross-validation on the full data, XGBoost R² is about 0.78 when whole makes are held out, 0.84 for the definition used (`Make` + first word of `Model`), 0.88 for the first two words of `Model` and 0.905 for exact model names, against 0.94 for a random split. The headline of about 0.8 is therefore relative to the definition used; even the most lenient grouping scores below the random split.

Error is larger for large pickups and two-seaters (MAE about 27–29 g/km) and above 300 g/km, and smaller for mid-size and wagon classes (about 9–13 g/km). Dropping `Make` costs about 0.02 R², and dropping `Make`, `Gears` and `Model year` about 0.03. A random 80/20 split of the same data reports R² ≈ 0.95 (notebook `04_Model.ipynb`, section 3.3), which overstates accuracy for new vehicle lines; the figures above are the ones to use.

## Limitations

- Typical error is ±17 g/km on a new vehicle line, and larger for very high-emission vehicles. R² is about 0.8 and ranges from about 0.78 to 0.90 depending on how strictly a new vehicle line is defined.
- Trained on Canadian laboratory ratings for 2015–2024 vehicles; real-world emissions differ, and electric or plug-in powertrains are not modelled.
- Inputs outside the range of the data are extrapolations.
- Recommendations are often close siblings of the input vehicle, so savings are typically modest.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`. To rebuild the recommender after retraining, run `python recommender.py`; to regenerate the held-out predictions, run `python scripts/export_test_predictions.py`.

## Data

Natural Resources Canada, fuel consumption ratings for light-duty vehicles (model years 2015–2024).
