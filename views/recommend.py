import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from utils import get_recommendations, load_metrics

MAE = load_metrics()['test_unseen_families']['MAE']

ui.header("Recommend alternatives",
          "Real vehicles with similar specifications whose measured emissions are lower than your estimate.")

if 'prediction' not in st.session_state or 'input_specs' not in st.session_state:
    st.info("Run a prediction first, then come back here.", icon=":material/info:")
    st.page_link("views/predict.py", label="Go to Predict", icon=":material/speed:")
    st.stop()

pred = st.session_state['prediction']
specs = st.session_state['input_specs']

with st.container(border=True):
    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    c1.metric("Your estimate", f"{pred:.0f} g/km")
    c2.markdown(
        f"**{specs['Make']} · {specs['Vehicle class']}**  \n"
        f"{specs['Engine size (L)']} L · {ui.trans_label(specs['Transmission type'])} · {specs['Gears']} gears · "
        f"{ui.fuel_label(specs['Fuel type'])} · {specs['Model year']}"
    )
    k = c3.slider("Results", 3, 10, 5)

try:
    recs = get_recommendations(specs, pred, k=k)
except Exception as e:
    st.error(f"Could not fetch recommendations: {e}")
    st.stop()

st.caption(f"Matches are the most similar vehicles (cosine similarity on the same encoded specs the model uses) whose "
           f"measured CO₂ is at least {MAE:.0f} g/km below your estimate, i.e. a saving larger than the model's typical error.")

if recs.empty:
    st.success("No clearly lower-emission vehicle with similar specs exists in the data. "
               "Your estimate is already at the low end for this configuration.", icon=":material/check_circle:")
    st.stop()

table = pd.DataFrame({
    'Vehicle': recs['Make'] + ' ' + recs['Model'],
    'Class': recs['Vehicle class'],
    'Engine (L)': recs['Engine size (L)'],
    'Fuel': recs['Fuel type'].map(ui.fuel_label),
    'Transmission': recs['Transmission type'].map(ui.trans_label) + ' · ' + recs['Gears'].astype(str) + ' gears',
    'CO₂ (g/km)': recs['CO2 emissions (g/km)'].astype(int),
    'Saving vs estimate': ((pred - recs['CO2 emissions (g/km)']) / pred * 100).round(0).astype(int).astype(str) + '%',
    'Similarity': recs['Similarity'].round(3),
})
st.dataframe(table, hide_index=True, width="stretch",
             column_config={'Similarity': st.column_config.ProgressColumn('Similarity', min_value=0, max_value=1, format="%.2f")})

labels = ['Your estimate'] + list(table['Vehicle'] + ' (' + table['Engine (L)'].astype(str) + ' L)')
values = [pred] + list(table['CO₂ (g/km)'])
fig = go.Figure(go.Bar(x=values[::-1], y=labels[::-1], orientation='h',
                       marker_color=[ui.TEAL] * len(values[1:]) + [ui.AMBER], text=[f"{v:.0f}" for v in values[::-1]],
                       textposition='outside'))
fig.update_layout(xaxis_title="CO₂ emissions (g/km)", yaxis_title=None, showlegend=False)
st.plotly_chart(ui.style_fig(fig, 90 + 42 * len(values)), width="stretch")
