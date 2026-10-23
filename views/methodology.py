import streamlit as st

import ui
from utils import load_metrics, get_reference_df

M = load_metrics()
test = M['test_unseen_families']
ref = get_reference_df()

ui.header("Methodology", "How the data was prepared, how the model was built and checked, and what it cannot do.")

st.subheader("1 · Data")
st.write(f"Natural Resources Canada fuel-consumption ratings, model years {int(ref['Model year'].min())}–"
         f"{int(ref['Model year'].max())}: {M['n_rows']:,} vehicles after removing the single natural-gas record "
         "(too few to learn from).")

st.subheader("2 · Cleaning")
st.markdown(
    "- Dropped the duplicate-unit column (mpg) and the City/Highway columns after checking that the Combined figure "
    "equals the 55/45 weighted blend of them (differences were rounding only).\n"
    "- Dropped the Smog and CO₂ *ratings*: they are bands derived from the target and would leak it.\n"
    "- High-emission vehicles were inspected and kept: they are real performance cars, not data errors.")

st.subheader("3 · Features")
st.markdown(
    "Seven inputs: **model year, make, vehicle class, engine size, fuel type, transmission type and gear count** "
    "(transmission code split into type and gears; 0 is stored when no gear number is listed, which only happens for CVTs).\n\n"
    "- **Combined fuel consumption is excluded.** It is almost a re-statement of CO₂ (correlation 0.94) and is not known "
    "before a prototype exists.\n"
    "- **Cylinders is excluded.** It is highly correlated with engine size (0.92) and added no accuracy once engine size was present.\n"
    "- **Model name is excluded.** Over 2,000 distinct values; the other specs already describe the vehicle.")

st.subheader("4 · Model")
st.markdown(
    "One scikit-learn pipeline: one-hot encoding for categorical inputs, standardisation for numeric inputs, then the regressor. "
    "A mean-predictor baseline, Ridge regression, a decision tree, Random Forest and XGBoost (also on a log-transformed target) "
    "were compared, and the strongest models were tuned with randomised search. "
    f"**{M['model']}** had the best cross-validated score and was refit on all {M['n_rows']:,} vehicles for the app.")

st.subheader("5 · Validation")
st.markdown(
    "Vehicle lines (make plus the first word of the model name, 443 in total) are kept whole inside either the training or "
    "the evaluation data, using grouped cross-validation and a held-out set of unseen lines. Model choice used cross-validation only; "
    "the held-out set was scored once. "
    f"Result on unseen lines: **R² {test['R2']:.2f}, RMSE {test['RMSE']:.1f} g/km, MAE {test['MAE']:.1f} g/km.** "
    f"A separate time-based check (train 2015–2021, test 2022–2024) gives R² {M['temporal_2022_24']['R2']:.2f}.\n\n"
    "The score depends on how strictly a vehicle line is defined. In cross-validation on the full data, XGBoost R² is about 0.78 when "
    "whole makes are held out, 0.84 for the definition used, and 0.905 when only exact model names are held out, "
    "compared with 0.94 for a random split. Read the headline of about 0.8 as relative to the definition used.")

st.subheader("6 · Recommender")
st.markdown(
    "The input specs are encoded with the model's own preprocessor, and the nearest real vehicles are found with cosine similarity "
    f"(k-nearest neighbours). Only vehicles whose *measured* emissions are at least {test['MAE']:.0f} g/km below the estimate "
    "are shown, so a suggestion is lower by more than the model's typical error.")

st.subheader("Limitations")
st.markdown(
    f"- Typical error is ±{test['MAE']:.0f} g/km on a new vehicle line, and larger above 300 g/km.\n"
    "- Trained on Canadian laboratory ratings for 2015–2024 vehicles; real-world emissions differ, and electric or plug-in "
    "powertrains are not modelled.\n"
    "- Inputs far outside the data (for example engine sizes or years beyond the range) are extrapolations.\n"
    "- Recommendations are often close siblings of the input vehicle, so the savings are typically modest.")
