"""Shared UI helpers: labels, colours, page header and light CSS."""
import streamlit as st

FUEL_LABELS = {'X': 'Regular gasoline', 'Z': 'Premium gasoline', 'D': 'Diesel', 'E': 'Ethanol (E85)'}
TRANS_LABELS = {'A': 'Automatic', 'AM': 'Automated manual', 'AS': 'Automatic, select shift',
                'AV': 'Continuously variable (CVT)', 'M': 'Manual'}

TEAL = '#0f766e'
AMBER = '#d97706'
GREY = '#6b7280'
FUEL_COLORS = {'Regular gasoline': '#0f766e', 'Premium gasoline': '#0ea5e9', 'Diesel': '#d97706', 'Ethanol (E85)': '#9333ea'}

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1200px; }
h1 { font-weight: 700; letter-spacing: -0.01em; }
div[data-testid="stMetric"] { background: #f3f6f5; border-radius: 10px; padding: 14px 16px; }
div[data-testid="stMetricLabel"] p { color: #4b5563; font-size: 0.85rem; }
footer { visibility: hidden; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def fuel_label(code):
    return FUEL_LABELS.get(code, code)


def trans_label(code):
    return TRANS_LABELS.get(code, code)


def style_fig(fig, height=360):
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color='#1f2937'), legend=dict(orientation='h', y=-0.2))
    return fig
