import streamlit as st

import ui

st.set_page_config(page_title="CO₂ Emission Predictor", page_icon="🍃", layout="wide")
ui.inject_css()

pages = [
    st.Page("views/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/predict.py", title="Predict", icon=":material/speed:"),
    st.Page("views/recommend.py", title="Recommend", icon=":material/eco:"),
    st.Page("views/performance.py", title="Model performance", icon=":material/monitoring:"),
    st.Page("views/explorer.py", title="Data explorer", icon=":material/bar_chart:"),
    st.Page("views/methodology.py", title="Methodology", icon=":material/menu_book:"),
]
st.navigation(pages).run()
