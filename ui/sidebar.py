"""
sidebar.py
----------
Renders the Streamlit sidebar with all configurable tracking parameters.
Returns a settings dict passed directly into the tracking pipeline.
"""

import streamlit as st
import torch

from tracker.config import (
    HIGH_CONF_THR,
    LOW_CONF_THR,
    REID_STRICT_THRESH,
    REID_CENTER_MAX_DIST,
    IOU_THR,
    MAX_AGE,
    MIN_HITS_TO_CONFIRM,
    KALMAN_PROCESS_NOISE,
    KALMAN_MEAS_NOISE,
    NUM_PARTS,
    EMBED_CACHE_SIZE,
)


def render_sidebar() -> dict:
    """
    Render sidebar controls and return the selected settings.

    Returns:
        Dict with keys matching the arguments of PersonReIDTracker
        and run_tracking_pipeline.
    """
    with st.sidebar:
        st.header("Tracking Settings")

        st.subheader("Detection")
        high_conf_thresh = st.slider(
            "High Confidence Threshold", 0.1, 0.9, HIGH_CONF_THR, 0.05,
            help="Only detections above this are passed to the tracker.",
        )
        low_conf_thresh = st.slider(
            "Low Confidence Threshold", 0.05, 0.5, LOW_CONF_THR, 0.05,
            help="YOLO prediction floor — detections below this are ignored.",
        )

        st.subheader("ReID Parameters")
        reid_strict_thresh = st.slider(
            "ReID Strict Threshold", 0.1, 1.0, REID_STRICT_THRESH, 0.05,
            help="Max cosine distance to accept a ReID-based re-attachment.",
        )
        reid_center_max_dist = st.number_input(
            "ReID Max Center Distance (px)", 50, 500, REID_CENTER_MAX_DIST, 10,
            help="Max pixel distance between track and detection center for ReID.",
        )

        st.subheader("Tracking")
        iou_thresh = st.slider(
            "IoU Threshold", 0.1, 0.8, IOU_THR, 0.05,
            help="Minimum IoU to directly accept a track-detection match.",
        )
        max_age = st.number_input(
            "Max Age (frames)", 10, 500, MAX_AGE, 10,
            help="Frames without a match before a track is retired to the gallery.",
        )
        min_hits = st.number_input(
            "Min Hits to Confirm", 1, 10, MIN_HITS_TO_CONFIRM, 1,
            help="Detections required before a track is marked as confirmed.",
        )

        with st.expander("Advanced Info (read-only)"):
            st.write(f"**Kalman Process Noise:** {KALMAN_PROCESS_NOISE}")
            st.write(f"**Kalman Measurement Noise:** {KALMAN_MEAS_NOISE}")
            st.write(f"**Body Parts for ReID:** {NUM_PARTS}")
            st.write(f"**Embedding Cache Size:** {EMBED_CACHE_SIZE} frames")

        st.subheader("System")
        device_label = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.info(f"Running on: {device_label}")

    return {
        "high_conf_thresh"    : high_conf_thresh,
        "low_conf_thresh"     : low_conf_thresh,
        "reid_strict_thresh"  : reid_strict_thresh,
        "reid_center_max_dist": reid_center_max_dist,
        "iou_thresh"          : iou_thresh,
        "max_age"             : max_age,
        "min_hits"            : min_hits,
    }
