"""Re-creates the held-out evaluation from 04_Model.ipynb and saves per-row test predictions.

Same family-grouped split (GroupShuffleSplit, random_state=42) and the tuned hyper-parameters stored in
model/metrics.json. Output: model/test_predictions.csv, used by the Model Performance page.
Run from the project root:  python scripts/export_test_predictions.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 'CO2 emissions (g/km)'
CAT = ['Make', 'Vehicle class', 'Fuel type', 'Transmission type']

metrics = json.load(open(os.path.join(ROOT, 'model', 'metrics.json'), encoding='utf-8'))
feats = metrics['features']

clean = pd.read_csv(os.path.join(ROOT, 'datasets', 'cleaned_data', 'cleaned_cars_info.csv'))
df = pd.read_csv(os.path.join(ROOT, 'datasets', 'feature_engineered_data', 'feature_engineered_cars.csv'))
df['Model'] = clean['Model'].values
df = df[df['Fuel type'] != 'N'].reset_index(drop=True)
df['Family'] = df['Make'] + ' ' + df['Model'].str.split().str[0]

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, df[T], groups=df['Family']))
train, test = df.iloc[tr], df.iloc[te].reset_index(drop=True)

params = {k.replace('model__', ''): v for k, v in metrics['best_params'].items()}
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), [c for c in feats if c in CAT]),
                         ('num', StandardScaler(), [c for c in feats if c not in CAT])])
pipe = Pipeline([('preprocessor', pre), ('model', XGBRegressor(random_state=42, n_jobs=1, **params))])
pipe.fit(train[feats], train[T])

test['Predicted'] = pipe.predict(test[feats])
r2 = r2_score(test[T], test['Predicted'])
print(f"Reproduced test R2 = {r2:.4f} (notebook: {metrics['test_unseen_families']['R2']:.4f})")

out = test[['Make', 'Model', 'Vehicle class', 'Fuel type', 'Model year', T, 'Predicted']].rename(columns={T: 'Actual'})
out.to_csv(os.path.join(ROOT, 'model', 'test_predictions.csv'), index=False)
print('saved', len(out), 'rows')
