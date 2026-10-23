import plotly.express as px
import streamlit as st

import ui
from utils import get_reference_df

T = 'CO2 emissions (g/km)'
df = get_reference_df()
df['Fuel'] = df['Fuel type'].map(ui.fuel_label)

ui.header("Data explorer", "The NRCan vehicle ratings behind the model. Filters apply to every chart.")

f1, f2, f3 = st.columns([1.2, 1.2, 2])
years = f1.slider("Model year", int(df['Model year'].min()), int(df['Model year'].max()),
                  (int(df['Model year'].min()), int(df['Model year'].max())))
fuels = f2.multiselect("Fuel", sorted(df['Fuel'].unique()), default=sorted(df['Fuel'].unique()))
classes = f3.multiselect("Vehicle class", sorted(df['Vehicle class'].unique()), placeholder="All classes")

view = df[df['Model year'].between(*years) & df['Fuel'].isin(fuels)]
if classes:
    view = view[view['Vehicle class'].isin(classes)]

if view.empty:
    st.warning("No vehicles match these filters.")
    st.stop()

k1, k2, k3 = st.columns(3)
k1.metric("Vehicles", f"{len(view):,}")
k2.metric("Mean CO₂", f"{view[T].mean():.0f} g/km")
k3.metric("Median CO₂", f"{view[T].median():.0f} g/km")

tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Engine & fuel", "Vehicle class", "Trend"])

with tab1:
    fig = px.histogram(view, x=T, nbins=50, color='Fuel', color_discrete_map=ui.FUEL_COLORS, barmode='stack')
    fig.update_layout(xaxis_title="CO₂ emissions (g/km)", yaxis_title="Vehicles", legend_title=None)
    st.plotly_chart(ui.style_fig(fig, 420), width="stretch")

with tab2:
    fig = px.scatter(view, x='Engine size (L)', y=T, color='Fuel', color_discrete_map=ui.FUEL_COLORS, opacity=0.45,
                     hover_data=['Make', 'Model', 'Model year'])
    fig.update_layout(xaxis_title="Engine size (L)", yaxis_title="CO₂ emissions (g/km)", legend_title=None)
    st.plotly_chart(ui.style_fig(fig, 460), width="stretch")
    st.caption("CO₂ rises with engine size for every fuel type; the spread at a given size reflects class, weight and drivetrain.")

with tab3:
    g = view.groupby('Vehicle class')[T].agg(['mean', 'count']).reset_index().sort_values('mean')
    fig = px.bar(g, x='mean', y='Vehicle class', orientation='h', text=g['count'].map(lambda n: f"n={n}"),
                 color_discrete_sequence=[ui.TEAL])
    fig.update_layout(xaxis_title="Mean CO₂ (g/km)", yaxis_title=None)
    st.plotly_chart(ui.style_fig(fig, 560), width="stretch")
    st.caption("Classes with few vehicles (small n) should not be over-interpreted.")

with tab4:
    g = view.groupby('Model year')[T].mean().reset_index()
    fig = px.line(g, x='Model year', y=T, markers=True, color_discrete_sequence=[ui.TEAL])
    fig.update_layout(yaxis_title="Mean CO₂ (g/km)")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(ui.style_fig(fig, 380), width="stretch")
    st.caption("The yearly average rises after 2019. The mix of vehicles changes over time, so this is not a like-for-like trend.")
