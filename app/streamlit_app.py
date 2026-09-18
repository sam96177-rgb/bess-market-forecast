"""Entry point: sets page config and wires up the four monitoring pages.
Each page reads only results/*.parquet (see app/data_loader.py) — this
file contains no data logic of its own.
"""
import streamlit as st

st.set_page_config(page_title="BESS Market Forecast Monitor", layout="wide")

pages = [
    st.Page("pages/overview.py", title="Overview", icon="🏆", default=True),
    st.Page("pages/forecast_inspector.py", title="Forecast Inspector", icon="🔍"),
    st.Page("pages/error_analysis.py", title="Error Analysis", icon="📊"),
    st.Page("pages/calibration.py", title="Calibration", icon="🎯"),
]

nav = st.navigation(pages)
nav.run()
