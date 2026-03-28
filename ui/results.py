"""
results.py
----------
Renders the results panel on the right column of the main layout.

Displays four tabs:
  1. Video Download  — download the annotated output video
  2. CSV Data        — view and download the tracking log table
  3. Analytics       — summary metrics and status distribution chart
  4. Algorithm Info  — description of the tracking pipeline
"""

import os
import streamlit as st

from tracker.config import EMBED_CACHE_SIZE


def render_results():
    """
    Render the right-column results area based on session state.

    Reads from:
        st.session_state.processing    — True while tracking is running
        st.session_state.err_message   — Error string, or None on success
        st.session_state.output_video  — Path to output video file
        st.session_state.tracking_data — DataFrame with tracking logs
    """
    st.header("Results and Analytics")

    processing    = st.session_state.get("processing", False)
    err_message   = st.session_state.get("err_message")
    output_video  = st.session_state.get("output_video")
    tracking_data = st.session_state.get("tracking_data")

    if processing:
        st.info("Processing in progress...")
        return

    if err_message:
        st.error(f"Error: {err_message}")
        with st.expander("Troubleshooting"):
            st.markdown("""
**Common causes:**
- Video format not supported (use MP4)
- Missing model file `yolo11m.pt` in working directory
- Insufficient GPU memory (try CPU mode)
- torchvision / PyTorch version mismatch

**Solutions:**
1. Convert video to MP4 format
2. Ensure `yolo11m.pt` exists in the project folder
3. Reduce video resolution or length
4. Update PyTorch and torchvision
""")
        return

    if output_video and tracking_data is not None:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Video Download", "CSV Data", "Analytics", "Algorithm Info"]
        )

        with tab1:
            _render_video_tab(output_video)

        with tab2:
            _render_csv_tab(tracking_data)

        with tab3:
            _render_analytics_tab(tracking_data)

        with tab4:
            _render_info_tab()

    else:
        st.info("Upload a video and click 'Start Tracking' to begin.")


# ------------------------------------------------------------------
# Tab renderers
# ------------------------------------------------------------------

def _render_video_tab(output_video: str):
    st.subheader("Annotated Tracked Video")

    if st.checkbox("Show debug info"):
        st.write(f"Session path: `{output_video}`")
        st.write(f"File exists: {os.path.exists(output_video)}")
        if os.path.exists(output_video):
            st.write(f"File size: {os.path.getsize(output_video)} bytes")
        video_files = [f for f in os.listdir(".") if f.endswith(".mp4")]
        st.write(f"All MP4 files in working dir: {video_files}")

    if os.path.exists(output_video):
        with open(output_video, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            "Download Tracked Video",
            data=video_bytes,
            file_name="tracked_output.mp4",
            mime="video/mp4",
            use_container_width=True,
            type="primary",
        )
        size_mb = len(video_bytes) / (1024 * 1024)
        st.success("Video ready for download.")
        st.info(f"Size: {size_mb:.1f} MB")
        st.markdown("""
**Output video contains:**
- Part-based ReID: 3-part feature extraction per person
- Kalman filtering: smooth motion prediction
- Hungarian assignment: globally optimal track-detection matching
- Gallery re-identification: returning persons keep their original ID
- Part division lines visible on each confirmed track
- Frame-level stats in the top-left corner
""")
    else:
        st.error("Tracked video file not found.")
        with st.expander("Why is the video missing?"):
            st.markdown("""
Possible causes:
1. Processing error during tracking
2. Insufficient disk space or memory
3. OpenCV codec issue — try converting input to H.264 MP4
""")


def _render_csv_tab(tracking_data):
    st.subheader("Tracking Logs")
    st.dataframe(tracking_data, use_container_width=True)
    csv_data = tracking_data.to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="tracking_logs.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )


def _render_analytics_tab(tracking_data):
    if tracking_data.empty:
        st.info("No tracking data available.")
        return

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Total People", tracking_data["ID"].nunique())

    with col_b:
        active_count = (tracking_data["Status"] == "Active").sum()
        st.metric("Currently Active", active_count)

    with col_c:
        completed = (tracking_data["Status"] == "Completed").sum()
        st.metric("Completed Visits", completed)

    with col_d:
        completed_df = tracking_data[
            (tracking_data["Status"] == "Completed") &
            (tracking_data["Duration"].str.contains("s", na=False))
        ]
        if not completed_df.empty:
            durations    = completed_df["Duration"].str.replace("s", "", regex=False).astype(float)
            avg_duration = durations.mean()
            st.metric("Avg Duration", f"{avg_duration:.1f}s")
        else:
            st.metric("Avg Duration", "N/A")

    st.subheader("Status Distribution")
    status_counts = tracking_data["Status"].value_counts()
    st.bar_chart(status_counts)


def _render_info_tab():
    st.markdown(f"""
### Tracking Pipeline

**Algorithm steps per frame:**
1. Kalman predict — advances each track by one time step using estimated velocity
2. Hungarian assignment — globally optimal matching using combined IoU + ReID cost
3. ReID gating — accepted only if IoU >= threshold OR strict cosine + spatial gates pass
4. Fallback IoU attach — catches moderate-overlap pairs rejected by Hungarian
5. Gallery query — re-attaches returning persons to their original track ID
6. New track creation — truly unmatched detections become new tracks

**Track Gallery:**
- Stores a confidence-weighted embedding + spatial metadata on track retirement
- 4-gate retrieval: TTL > spatial distance > area ratio > cosine distance
- Gallery entries expire after 10 minutes to prevent stale matches
- Max gallery size: 100 entries (oldest evicted when full)

**Technical parameters:**
- Feature dimensions: 2048D per body part (ResNet50)
- Embedding cache: {EMBED_CACHE_SIZE} frames sliding window per active track
- Kalman state: 8D (position [cx, cy, w, h] + velocity [vx, vy, vw, vh])
- Assignment cost: {0.4} x (1 - IoU) + {0.6} x ReID cosine distance
""")
