import os
import json
import joblib
import pandas as pd
import streamlit as st
from recommender import recommend_lower_emission, FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(rel):
    return os.path.join(BASE_DIR, rel)


@st.cache_resource
def load_model(model_path='model/co2_production_model.pkl'):
    """
    Loads and caches the production pipeline.
    """
    return joblib.load(_path(model_path))


@st.cache_resource
def load_recommender(recommender_path='model/co2_knn_recommender.pkl'):
    """
    Loads and caches the fitted recommender structures.
    """
    return joblib.load(_path(recommender_path))


@st.cache_data
def load_metrics(metrics_path='model/metrics.json'):
    """
    Loads the metrics written by 04_Model.ipynb, so the app never hard-codes scores.
    """
    with open(_path(metrics_path), encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def get_reference_df():
    """
    The full vehicle table (real NRCan records) used for charts and as the recommender's candidate pool.
    """
    return load_recommender()['reference_df'].copy()


@st.cache_data
def load_test_predictions(path='model/test_predictions.csv'):
    """
    Held-out predictions on vehicle lines the model never saw (written by scripts/export_test_predictions.py).
    """
    return pd.read_csv(_path(path))


def predict_co2(input_dict):
    """
    Takes a dictionary of features, runs it through the pipeline,
    and returns the predicted CO2 emissions (g/km) as a float.
    """
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    return float(load_model().predict(input_df)[0])


def get_recommendations(input_dict, predicted_co2, k=5):
    """
    Takes the user inputs and prediction, and returns a DataFrame of eco-friendly recommendations.
    Suggestions must be lower than the prediction by at least the model's test MAE on unseen vehicle lines.
    """
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    margin = load_metrics()['test_unseen_families']['MAE']
    return recommend_lower_emission(input_df, predicted_co2, k=k, margin=margin)


@st.cache_data
def get_categorical_options():
    """
    Returns unique sorted categorical options from the dataset for UI dropdowns.
    """
    ref_df = load_recommender()['reference_df']
    return {
        'Make': sorted(ref_df['Make'].unique().tolist()),
        'Vehicle class': sorted(ref_df['Vehicle class'].unique().tolist()),
        'Fuel type': sorted(ref_df['Fuel type'].unique().tolist()),
        'Transmission type': sorted(ref_df['Transmission type'].unique().tolist())
    }
