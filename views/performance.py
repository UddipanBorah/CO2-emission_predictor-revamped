import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from utils import load_metrics, load_test_predictions

M = load_metrics()
test = M['test_unseen_families']
temporal = M['temporal_2022_24']
preds = load_test_predictions()
preds['Error'] = preds['Predicted'] - preds['Actual']
preds['Abs error'] = preds['Error'].abs()

ui.header("Model performance", f"Production model: {M['model']} · trained on {M['n_rows']:,} vehicles.")

st.info(
    "**How the model is validated.** Most vehicle lines appear many times in the data (trims, years, drivetrains). "
    "To avoid scoring the model on near-copies of its training data, every vehicle line (for example *Ford F-150*) is kept "
    "entirely inside either the training folds or the evaluation set. The scores below therefore estimate accuracy on a "
    "**new vehicle line**, which is the situation in early design.",
    icon=":material/info:")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Cross-validated R²", f"{M['grouped_cv']['R2']:.3f}", help=f"5-fold, std {M['grouped_cv']['R2_std']:.3f}")
c2.metric("Test R²", f"{test['R2']:.3f}")
c3.metric("Test RMSE", f"{test['RMSE']:.1f} g/km")
c4.metric("Test MAE", f"{test['MAE']:.1f} g/km")
st.caption(f"Test set: {len(preds):,} vehicles from vehicle lines not used for training or model selection. "
           f"Drift check (train on 2015–2021, test on 2022–2024): R² {temporal['R2']:.3f}, MAE {temporal['MAE']:.1f} g/km.")

st.divider()

st.subheader("Model comparison")
rows = []
for name, v in M['comparison'].items():
    if name.startswith('Dummy'):
        continue
    rows.append({'Model': name, 'Stage': 'Default settings', 'R²': v['CV_R2']})
for name, v in M['tuned_cv_r2'].items():
    rows.append({'Model': name, 'Stage': 'Tuned', 'R²': v})
comp = pd.DataFrame(rows)
fig = px.bar(comp, x='R²', y='Model', color='Stage', barmode='group', orientation='h',
             color_discrete_map={'Default settings': '#cbd5d1', 'Tuned': ui.TEAL}, range_x=[0.6, 0.9])
fig.update_layout(xaxis_title="Cross-validated R² (higher is better)", yaxis_title=None)
st.plotly_chart(ui.style_fig(fig, 360), width="stretch")
st.caption(f"A model that always predicts the average scores R² ≈ {M['comparison']['Dummy (mean)']['CV_R2']:.2f}. "
           "Cross-validated scores carry a standard deviation of about 0.04, so the top models are close; "
           "the production model was chosen as the best cross-validated score before the test set was scored.")

with st.expander("Full comparison table"):
    table = pd.DataFrame([
        {'Model': n, 'CV R²': v['CV_R2'], 'CV RMSE (g/km)': v['CV_RMSE'], 'CV MAE (g/km)': v['CV_MAE']}
        for n, v in M['comparison'].items()])
    st.dataframe(table.style.format({'CV R²': '{:.3f}', 'CV RMSE (g/km)': '{:.1f}', 'CV MAE (g/km)': '{:.1f}'}),
                 hide_index=True, width="stretch")

st.divider()
st.subheader("Where the model is accurate, and where it is not")

left, right = st.columns(2, gap="large")
with left:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=preds['Actual'], y=preds['Predicted'], mode='markers',
                               marker=dict(size=5, color=ui.TEAL, opacity=0.35), name='Vehicles'))
    lo, hi = float(preds['Actual'].min()), float(preds['Actual'].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line=dict(color=ui.AMBER, dash='dash'), name='Perfect'))
    fig.update_layout(title="Predicted vs actual (unseen vehicle lines)", xaxis_title="Actual (g/km)",
                      yaxis_title="Predicted (g/km)", showlegend=False)
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch")
with right:
    fig = px.histogram(preds, x='Error', nbins=50, color_discrete_sequence=[ui.TEAL])
    fig.add_vline(x=0, line_color=ui.AMBER, line_dash='dash')
    fig.update_layout(title="Prediction error (predicted − actual)", xaxis_title="g/km", yaxis_title="Vehicles")
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch")

left, right = st.columns(2, gap="large")
with left:
    by_class = preds.groupby('Vehicle class')['Abs error'].agg(['mean', 'count'])
    by_class = by_class[by_class['count'] >= 30].sort_values('mean')
    fig = px.bar(by_class.reset_index(), x='mean', y='Vehicle class', orientation='h', color_discrete_sequence=[ui.TEAL])
    fig.update_layout(title="Mean absolute error by vehicle class (n ≥ 30)", xaxis_title="MAE (g/km)", yaxis_title=None)
    st.plotly_chart(ui.style_fig(fig, 420), width="stretch")
with right:
    preds['Band'] = pd.cut(preds['Actual'], [0, 200, 250, 300, 400, 700],
                           labels=['< 200', '200–250', '250–300', '300–400', '> 400'])
    by_band = preds.groupby('Band', observed=True)['Abs error'].agg(['mean', 'count']).reset_index()
    fig = px.bar(by_band, x='Band', y='mean', text=by_band['count'].map(lambda n: f"n={n}"),
                 color_discrete_sequence=[ui.TEAL])
    fig.update_layout(title="Mean absolute error by actual emission level", xaxis_title="Actual CO₂ (g/km)",
                      yaxis_title="MAE (g/km)")
    st.plotly_chart(ui.style_fig(fig, 420), width="stretch")

st.caption("Error grows for the highest-emission vehicles (large pickups, two-seaters and exotics), which are rare in the data.")
