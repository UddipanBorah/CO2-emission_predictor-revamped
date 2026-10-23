import streamlit as st

import ui
from utils import load_metrics, get_reference_df

M = load_metrics()
ref = get_reference_df()
test = M['test_unseen_families']

ui.header("CO₂ Emission Predictor",
          "Estimate a vehicle's tailpipe CO₂ from early design specifications, then find real vehicles that emit less.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test R²", f"{test['R2']:.2f}", help="Measured on vehicle lines the model never saw during training.")
c2.metric("Typical error (MAE)", f"±{test['MAE']:.0f} g/km")
c3.metric("Vehicles in dataset", f"{len(ref):,}")
c4.metric("Model years", f"{int(ref['Model year'].min())}–{int(ref['Model year'].max())}")

st.divider()

a, b, c = st.columns(3, gap="large")
with a:
    with st.container(border=True):
        st.subheader("1 · Predict")
        st.write("Enter engine size, class, fuel, transmission and a few more specs to get an estimated CO₂ in g/km.")
        st.page_link("views/predict.py", label="Open Predict", icon=":material/arrow_forward:")
with b:
    with st.container(border=True):
        st.subheader("2 · Recommend")
        st.write("Find real vehicles with similar specifications whose measured emissions are clearly lower.")
        st.page_link("views/recommend.py", label="Open Recommend", icon=":material/arrow_forward:")
with c:
    with st.container(border=True):
        st.subheader("3 · Evaluate")
        st.write("See how the model was validated, where it is accurate and where it is not.")
        st.page_link("views/performance.py", label="Open Model performance", icon=":material/arrow_forward:")

st.divider()
st.markdown(
    f"""
**Use case.** In early vehicle design, engine size, vehicle class and transmission are known long before a prototype
can be tested. This tool turns those specs into an emissions estimate so a design that is likely to miss a target can be
reconsidered early.

**Data.** Natural Resources Canada fuel-consumption ratings, model years {int(ref['Model year'].min())}–{int(ref['Model year'].max())}.

**Read the numbers correctly.** The estimate is typically within about ±{test['MAE']:.0f} g/km for a *new* vehicle line, and
larger for very high-emission vehicles (above 300 g/km). It is a screening tool, not a certification value.
"""
)
