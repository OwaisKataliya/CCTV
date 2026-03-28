"""
app.py
------
Streamlit entry point for the Advanced Person Tracking application.

Responsibilities:
  - Initialise session state variables on first load.
  - Apply global CSS layout adjustments.
  - Render sidebar (settings), uploader (left column), results (right column).

All tracking logic lives in tracker/.
All UI logic lives in ui/.
"""

import warnings
import streamlit as st

# Suppress protobuf version warnings from Google libraries
warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from ui.sidebar import render_sidebar
from ui.uploader import render_uploader
from ui.results import render_results
from ui.pipeline import cleanup_temp_files


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Advanced Person Tracker",
    layout="wide",
)

# Reduce default Streamlit padding for a more compact layout
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

if "output_video" not in st.session_state:
    st.session_state.output_video = None
if "tracking_data" not in st.session_state:
    st.session_state.tracking_data = None
if "err_message" not in st.session_state:
    st.session_state.err_message = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# ------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------

st.title("Advanced Entry / Exit Tracker")
st.markdown("*Person tracking with part-based ReID, Kalman filtering, and persistent track gallery.*")

# Render sidebar and get settings dict
settings = render_sidebar()

# Two-column layout: upload controls (left) | results (right)
col_upload, col_results = st.columns([1, 2])

with col_upload:
    render_uploader(settings)

with col_results:
    render_results()


# ------------------------------------------------------------------
# Startup cleanup — remove any stale temp files from previous runs
# ------------------------------------------------------------------

if __name__ == "__main__":
    cleanup_temp_files()
