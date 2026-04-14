"""
pipeline.py
-----------
Core video processing pipeline for the Streamlit app.

run_tracking_pipeline():
  - Accepts an uploaded video and settings dict from the sidebar.
  - Runs YOLO detection + PersonReIDTracker on every frame.
  - Draws annotated bounding boxes and writes output video.
  - Returns (output_video_path, tracking_dataframe, error_message).
"""

import os
import time
import uuid
import tempfile

import cv2
import numpy as np
import torch
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from tracker.config import (
    HIGH_CONF_THR,
    LOW_CONF_THR,
    NUM_PARTS,
    IOU_THR,
    REID_STRICT_THRESH,
    MAX_AGE,
    MIN_HITS_TO_CONFIRM,
)
from tracker.reid_tracker import PersonReIDTracker


@st.cache_resource
def load_yolo_model():
    """
    Load and cache YOLO11m.
    Called once per Streamlit session thanks to @st.cache_resource.
    """
    try:
        device     = "cuda" if torch.cuda.is_available() else "cpu"
        yolo_model = YOLO("yolo11m.pt")
        if device == "cuda":
            yolo_model.to(device)
        return yolo_model, device
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None


def run_tracking_pipeline(uploaded_video, settings: dict):
    """
    Execute the full detection + tracking pipeline on an uploaded video.

    Args:
        uploaded_video: Streamlit UploadedFile object.
        settings:       Dict from render_sidebar() with user-selected thresholds.

    Returns:
        (output_video_path, tracking_df, error_message)
        output_video_path is None on failure.
        error_message is None on success.
    """
    if uploaded_video is None:
        return None, None, "No video uploaded."

    temp_input_path = None
    try:
        # Write uploaded bytes to a temp file so OpenCV can open it by file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            temp_input_path = tmp.name

        yolo_model, device = load_yolo_model()
        if yolo_model is None:
            return None, None, "Failed to load YOLO model."

        unique_id         = str(uuid.uuid4())[:8]
        output_video_path = f"tracked_video_{unique_id}.mp4"

        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            return None, None, "Could not open video file."

        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialise tracker — settings from sidebar are applied here
        tracker = PersonReIDTracker(
            device             = device,
            frame_size         = (width, height),
            fps                = fps,
            iou_thr            = settings.get("iou_thresh",         IOU_THR),
            reid_strict_thresh = settings.get("reid_strict_thresh", REID_STRICT_THRESH),
            max_age            = int(settings.get("max_age",        MAX_AGE)),
            min_hits           = int(settings.get("min_hits",       MIN_HITS_TO_CONFIRM)),
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            return None, None, "Could not create output video writer."

        high_conf_thr = settings.get("high_conf_thresh", HIGH_CONF_THR)
        low_conf_thr  = settings.get("low_conf_thresh",  LOW_CONF_THR)

        frame_count  = 0
        start_time   = time.time()
        progress_bar = st.progress(0)
        status_text  = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress     = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            elapsed      = time.time() - start_time
            spd          = frame_count / elapsed if elapsed > 0 else 0
            status_text.text(
                f"Frame {frame_count}/{total_frames}  |  Speed: {spd:.1f} fps"
            )

            # YOLO: detect people (class 0) with low confidence floor
            results = yolo_model.predict(
                source  = frame,
                classes = [0],
                device  = device,
                verbose = False,
                conf    = low_conf_thr,
                iou     = 0.7,
                half    = (device == "cuda"),
                imgsz   = 640,
            )

            show_frame = frame.copy()

            if len(results[0].boxes.data) == 0:
                out.write(show_frame)
                continue

            all_boxes      = results[0].boxes.data.cpu().numpy()
            high_conf_mask = all_boxes[:, 4] >= high_conf_thr
            high_conf_boxes = (
                all_boxes[high_conf_mask][:, :4]
                if high_conf_mask.any()
                else np.array([]).reshape(0, 4)
            )

            # Extract part embeddings for each high-confidence detection
            detections  = []
            parts_feats = []
            for bb in high_conf_boxes:
                box   = [float(b) for b in bb[:4]]
                parts = tracker.embedder.extract(frame, box)
                if parts:
                    detections.append(box)
                    parts_feats.append(parts)

            # Only call tracker if there are detections to process
            if detections:
                tracks = tracker.step(detections, parts_feats, frame_count)
            else:
                # No high-confidence detections — still advance Kalman predictions
                tracks = tracker.step([], [], frame_count)

            # Draw confirmed tracks
            live_confirmed = 0
            for tid, tr in tracks.items():
                if tr.retired:
                    continue

                # Generate a bright, saturated track color (avoid near-black)
                r_c = int((tid * 37) % 205) + 50
                g_c = int((tid * 97) % 205) + 50
                b_c = int((tid * 61) % 205) + 50
                color = (r_c, g_c, b_c)

                box             = [int(round(v)) for v in tr.smoothed_box]
                x1, y1, x2, y2 = box
                box_h = y2 - y1

                if tr.confirmed:
                    live_confirmed += 1
                    cv2.rectangle(show_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        show_frame,
                        f"ID:{tid}",
                        (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )

                    # Draw horizontal part division lines inside the box
                    # Only draw when the box is tall enough for lines to be visible
                    if box_h >= 40 and NUM_PARTS > 1:
                        part_h = box_h // NUM_PARTS
                        # Bright contrasting tint for part lines
                        line_color = (
                            min(255, r_c + 80),
                            min(255, g_c + 80),
                            min(255, b_c + 80),
                        )
                        for i in range(1, NUM_PARTS):
                            line_y = y1 + i * part_h
                            # Draw a thin dark outline + bright line for contrast
                            cv2.line(
                                show_frame,
                                (x1 + 1, line_y), (x2 - 1, line_y),
                                (0, 0, 0), 3,
                            )
                            cv2.line(
                                show_frame,
                                (x1 + 1, line_y), (x2 - 1, line_y),
                                line_color, 2,
                            )
                else:
                    # Unconfirmed track: thin gray box, no label
                    cv2.rectangle(show_frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

            # Frame-level info overlay
            cv2.putText(
                show_frame,
                f"Frame {frame_count}/{total_frames}  Confirmed: {live_confirmed}"
                f"  Gallery: {len(tracker.gallery)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 210), 2,
            )

            out.write(show_frame)

        cap.release()
        out.release()

        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)

        tracking_df   = _build_tracking_dataframe(tracker)
        elapsed_total = time.time() - start_time
        avg_fps       = frame_count / elapsed_total if elapsed_total > 0 else 0

        progress_bar.empty()
        status_text.empty()

        print(
            f"Tracking complete. {frame_count} frames in {elapsed_total:.1f}s "
            f"({avg_fps:.1f} fps avg). Gallery size: {len(tracker.gallery)}"
        )

        if os.path.exists(output_video_path):
            return output_video_path, tracking_df, None
        return None, tracking_df, "Output video not found after processing."

    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        print(f"Processing error: {e}")
        return None, None, f"Processing error: {e}"


def _build_tracking_dataframe(tracker: PersonReIDTracker) -> pd.DataFrame:
    """Build a summary DataFrame from all tracks in the tracker."""
    rows = []
    for tid, tr in tracker.tracks.items():
        entry_str = (
            tr.entry_time.strftime("%Y-%m-%d %H:%M:%S") if tr.entry_time else ""
        )
        if tr.exit_time is not None:
            exit_str = tr.exit_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = f"{(tr.exit_time - tr.entry_time).total_seconds():.1f}s"
            status   = "Completed"
        else:
            exit_str = "Active"
            duration = "Active"
            status   = "Active"

        rows.append({
            "ID"         : f"ID_{tid:03d}",
            "Entry_Time" : entry_str,
            "Exit_Time"  : exit_str,
            "Duration"   : duration,
            "Status"     : status,
        })
    return pd.DataFrame(rows)


def cleanup_temp_files():
    """Remove stale temporary files (temp_*.mp4, temp_*.csv) from working dir."""
    try:
        for f in os.listdir(os.getcwd()):
            if f.startswith("temp_") and (f.endswith(".mp4") or f.endswith(".csv")):
                if "tracked_video" not in f:
                    try:
                        os.unlink(f)
                    except Exception:
                        pass
    except Exception:
        pass


def cleanup_old_tracked_videos(max_age_seconds: int = 3600):
    """Remove tracked output videos older than max_age_seconds (default 1 hour)."""
    try:
        now = time.time()
        for f in os.listdir(os.getcwd()):
            if f.startswith("tracked_video_") and f.endswith(".mp4"):
                path = os.path.join(os.getcwd(), f)
                if now - os.path.getctime(path) > max_age_seconds:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
    except Exception:
        pass
