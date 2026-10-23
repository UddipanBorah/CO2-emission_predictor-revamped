import plotly.graph_objects as go
import streamlit as st

import ui
from utils import predict_co2, get_categorical_options, get_reference_df, load_metrics

M = load_metrics()
MAE = M['test_unseen_families']['MAE']
opts = get_categorical_options()
ref = get_reference_df()
T = 'CO2 emissions (g/km)'


def default_index(options, value):
    return options.index(value) if value in options else 0


ui.header("Predict emissions", "Enter early-stage specifications to estimate tailpipe CO₂ (g/km).")

left, right = st.columns([1, 1.3], gap="large")

with left:
    with st.form("specs_form"):
        make = st.selectbox("Make", opts['Make'], index=default_index(opts['Make'], 'Toyota'))
        vehicle_class = st.selectbox("Vehicle class", opts['Vehicle class'],
                                     index=default_index(opts['Vehicle class'], 'Compact'))
        fuel_type = st.selectbox("Fuel type", opts['Fuel type'], index=default_index(opts['Fuel type'], 'X'),
                                 format_func=ui.fuel_label)
        transmission = st.selectbox("Transmission", opts['Transmission type'],
                                    index=default_index(opts['Transmission type'], 'AS'), format_func=ui.trans_label)
        c3, c4 = st.columns(2)
        engine_size = c3.number_input("Engine size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        gears = c4.number_input("Gears", min_value=0, max_value=10, value=8, step=1,
                                help="0 means no gear count is listed in the data (this only occurs for CVTs). "
                                     "Some CVTs list simulated gear steps (6-10), so a CVT with gears is valid.")
        model_year = st.number_input("Model year", min_value=2015, max_value=2026, value=2024, step=1)
        submitted = st.form_submit_button("Predict CO₂", type="primary", width="stretch")

    notes = []
    lo, hi = ref['Engine size (L)'].min(), ref['Engine size (L)'].max()
    if not lo <= engine_size <= hi:
        notes.append(f"Engine size is outside the range seen in the data ({lo:.1f}–{hi:.1f} L); expect a less reliable estimate.")
    if engine_size > 6.0 and 0 < gears < 4:
        notes.append("A large engine (>6 L) with fewer than 4 gears is very rare in the data.")
    if transmission != 'AV' and gears == 0:
        notes.append("0 gears only occurs for CVT transmissions in the data.")
    if model_year > ref['Model year'].max():
        notes.append(f"The data ends at model year {int(ref['Model year'].max())}; later years are extrapolated.")
    for n in notes:
        st.warning(n, icon=":material/warning:")

if submitted:
    specs = {'Model year': int(model_year), 'Make': make, 'Vehicle class': vehicle_class,
             'Engine size (L)': float(engine_size), 'Fuel type': fuel_type,
             'Transmission type': transmission, 'Gears': int(gears)}
    try:
        st.session_state['prediction'] = predict_co2(specs)
        st.session_state['input_specs'] = specs
    except Exception as e:
        st.error(f"Could not compute a prediction: {e}")
        st.stop()

with right:
    if 'prediction' not in st.session_state:
        with st.container(border=True):
            st.markdown("#### Estimated CO₂ will appear here")
            st.caption("Fill in the specifications and press **Predict CO₂**.")
    else:
        pred = st.session_state['prediction']
        pct = float((ref[T] < pred).mean() * 100)
        with st.container(border=True):
            st.metric("Estimated CO₂ emissions", f"{pred:.0f} g/km")
            st.caption(f"Likely range on a new vehicle line: **{pred - MAE:.0f}–{pred + MAE:.0f} g/km** "
                       f"(typical error ±{MAE:.0f} g/km). Higher than **{pct:.0f}%** of the {len(ref):,} vehicles in the dataset.")

            fig = go.Figure(go.Histogram(x=ref[T], nbinsx=50, marker_color='#cbd5d1', name='Dataset'))
            fig.add_vrect(x0=pred - MAE, x1=pred + MAE, fillcolor=ui.TEAL, opacity=0.15, line_width=0)
            fig.add_vline(x=pred, line_color=ui.TEAL, line_width=3, annotation_text="Your estimate", annotation_position="top")
            fig.update_layout(xaxis_title="CO₂ emissions (g/km)", yaxis_title="Vehicles", showlegend=False)
            st.plotly_chart(ui.style_fig(fig, 300), width="stretch")

        st.page_link("views/recommend.py", label="Find lower-emission alternatives", icon=":material/eco:")
