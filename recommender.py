import os
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLS = ['Model year', 'Make', 'Vehicle class', 'Engine size (L)', 'Fuel type', 'Transmission type', 'Gears']


def _resolve(path):
    """Resolve a project-relative path against this file's folder, so it works from any cwd (e.g. Streamlit Cloud)."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def fit_and_save_recommender(model_path='model/co2_production_model.pkl',
                             data_path='datasets/feature_engineered_data/feature_engineered_cars.csv',
                             recommender_output_path='model/co2_knn_recommender.pkl'):
    """
    Fits NearestNeighbors on the preprocessed feature matrix of the dataset and saves it.
    Excludes Fuel type == 'N' from the recommendation pool.
    The preprocessor is the production pipeline's own (one-hot + standardised numerics), so numeric
    columns such as Model year do not dominate the cosine distance.
    """
    model_path, data_path, recommender_output_path = map(_resolve, (model_path, data_path, recommender_output_path))

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']

    df = pd.read_csv(data_path)

    # The Model column was dropped in the feature-engineered CSV but is needed for readable recommendations.
    cleaned_data_path = os.path.join(os.path.dirname(os.path.dirname(data_path)), 'cleaned_data', 'cleaned_cars_info.csv')
    df_cleaned = pd.read_csv(cleaned_data_path)
    assert len(df) == len(df_cleaned), 'feature-engineered and cleaned CSVs are not row-aligned'
    df['Model'] = df_cleaned['Model']

    df_filtered = df[df['Fuel type'] != 'N'].copy().reset_index(drop=True)

    X_encoded = preprocessor.transform(df_filtered[FEATURE_COLS])

    knn = NearestNeighbors(n_neighbors=100, metric='cosine')
    knn.fit(X_encoded)

    payload = {
        'knn': knn,
        'reference_df': df_filtered,
        'feature_cols': FEATURE_COLS,
    }

    os.makedirs(os.path.dirname(recommender_output_path), exist_ok=True)
    joblib.dump(payload, recommender_output_path)
    print(f"Recommender system fitted and saved to {recommender_output_path}")


def recommend_lower_emission(user_specs_df, predicted_co2, k=5, margin=0.0,
                             recommender_path='model/co2_knn_recommender.pkl',
                             model_path='model/co2_production_model.pkl'):
    """
    Given a user's vehicle specs and predicted CO2 emission, returns up to k similar vehicles
    from the reference pool whose actual emissions are lower than predicted_co2 - margin.
    `margin` (g/km) should be about the model's typical error, so a "lower-emission" suggestion is
    lower beyond prediction noise.
    """
    recommender_path, model_path = _resolve(recommender_path), _resolve(model_path)

    payload = joblib.load(recommender_path)
    knn = payload['knn']
    reference_df = payload['reference_df']
    feature_cols = payload['feature_cols']

    preprocessor = joblib.load(model_path).named_steps['preprocessor']
    user_encoded = preprocessor.transform(user_specs_df[feature_cols])

    # Query a wide neighbourhood so enough candidates remain after the emission filter.
    distances, indices = knn.kneighbors(user_encoded, n_neighbors=min(300, len(reference_df)))
    distances, indices = distances[0], indices[0]

    neighbors_df = reference_df.iloc[indices].copy()
    neighbors_df['Similarity'] = 1 - distances  # cosine similarity = 1 - cosine distance

    alternatives = neighbors_df[neighbors_df['CO2 emissions (g/km)'] < predicted_co2 - margin]
    alternatives = alternatives.sort_values(by=['Similarity', 'CO2 emissions (g/km)'], ascending=[False, True])

    cols_to_return = ['Make', 'Model', 'Vehicle class', 'Engine size (L)', 'Fuel type', 'Transmission type',
                      'Gears', 'CO2 emissions (g/km)', 'Similarity']
    return alternatives[cols_to_return].head(k)


if __name__ == "__main__":
    fit_and_save_recommender()
