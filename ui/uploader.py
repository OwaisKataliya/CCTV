"""
uploader.py
-----------
Renders the video upload panel, preview, and the process button.
Triggers run_tracking_pipeline() when the user clicks Start Tracking.
"""

import streamlit as st

from ui.pipeline import run_tracking_pipeline, cleanup_temp_files, cleanup_old_tracked_videos


def render_uploader(settings: dict):
    """
    Render the left column of the main layout:
      - Video file uploader
      - Video preview
      - File info (name, size)
      - Start Tracking button
      - Downloads section (CSV and video)

    Args:
        settings: Settings dict returned by render_sidebar().
    """
    st.header("Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a video for person tracking.",
    )

    if uploaded_file:
        st.subheader("Preview")
        st.video(uploaded_file)
        st.info(f"File: {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size / (1024 * 1024):.1f} MB")

        if st.button("Start Tracking", type="primary", use_container_width=True):
            st.session_state.processing    = True
            st.session_state.err_message   = None
            st.session_state.output_video  = None
            st.session_state.tracking_data = None

            output_path, tracking_df, err = run_tracking_pipeline(uploaded_file, settings)

            st.session_state.output_video  = output_path
            st.session_state.tracking_data = tracking_df
            st.session_state.err_message   = err
            st.session_state.processing    = False

            if output_path:
                st.success("Tracking completed.")
                st.rerun()
            else:
                st.error(f"Processing failed: {err}")

    # Downloads section
    st.header("Downloads")

    tracking_data = st.session_state.get("tracking_data")
    if tracking_data is not None and not tracking_data.empty:
        csv_data = tracking_data.to_csv(index=False)
        st.download_button(
            "Download CSV Logs",
            data=csv_data,
            file_name="tracking_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info(f"CSV: {len(tracking_data)} records")
    else:
        st.button("Download CSV", disabled=True, use_container_width=True)

    if st.button("Clean Temp Files", use_container_width=True):
        cleanup_temp_files()
        cleanup_old_tracked_videos()
        st.success("Temporary files cleaned.")
