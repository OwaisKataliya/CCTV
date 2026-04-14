"""
track.py
--------
Represents a single tracked person across video frames.

Lifecycle:
    New -> (hits >= MIN_HITS_TO_CONFIRM) -> Confirmed -> Lost -> Retired

A track stores:
- Kalman filter state for motion prediction
- Sliding deque of recent part embeddings for ReID
- Smoothed bounding box for stable visualization
- Entry/exit timestamps for CSV analytics
"""

import math
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np

from tracker.config import (
    EMBED_CACHE_SIZE,
    MIN_HITS_TO_CONFIRM,
    SMOOTH_ALPHA_SLOW,
    SMOOTH_ALPHA_FAST,
    EXIT_MARGIN,
    STATIC_VELOCITY_THRESH,
    STATIC_DISPLACEMENT_THRESH,
    STATIC_FRAMES_REQUIRED,
)
from tracker.kalman import KalmanBox
from tracker.utils import (
    box_xyxy_to_xywh,
    box_xywh_to_xyxy,
    l2_normalize_vec,
    center_of,
)


class Track:
    """
    Holds all state for a single tracked person.

    Args:
        init_box:   Initial detection box [x1, y1, x2, y2].
        part_feats: Initial list of part embeddings from the detection.
        track_id:   Unique integer ID assigned to this track.
        frame_idx:  Frame index at creation.
        dt:         Time delta between frames (1 / fps).
        frame_size: (width, height) of the video frame.
    """

    def __init__(
        self,
        init_box: List[float],
        part_feats: List[np.ndarray],
        track_id: int,
        frame_idx: int,
        dt: float,
        frame_size: tuple,
    ):
        self.id                = track_id
        self.age               = 0
        self.hits              = 1
        self.time_since_update = 0
        self.confirmed         = False
        self.retired           = False
        self.first_frame       = frame_idx
        self.last_update_frame = frame_idx
        self.frame_size        = frame_size   # (width, height)
        self.status            = "New"
        self.misses_outside    = 0

        self.box          = list(map(float, init_box))
        self.smoothed_box = list(map(float, init_box))

        # Sliding cache of L2-normalized part embeddings
        self.embeds: deque = deque(maxlen=EMBED_CACHE_SIZE)
        for p in part_feats:
            self.embeds.append(p)

        # Kalman filter
        self.kf     = KalmanBox(dt=dt)
        xywh        = box_xyxy_to_xywh(self.box)
        self.x, self.P = self.kf.initiate(xywh)

        # Static object detection state
        init_center = center_of(init_box)
        self._center_history: deque = deque(maxlen=STATIC_FRAMES_REQUIRED)
        self._center_history.append(init_center)
        self._consecutive_static: int = 0
        self._marked_static: bool     = False

        # Timestamps for analytics
        self.entry_time: datetime        = datetime.now()
        self.exit_time: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def predict(self) -> List[float]:
        """
        Advance Kalman state by one frame and update predicted box.
        Increments misses_outside if predicted center exits the frame.
        """
        self.x, self.P = self.kf.predict(self.x, self.P)
        pred_box  = box_xywh_to_xyxy(self.x[0:4])
        self.box  = [float(v) for v in pred_box]

        # Check if predicted center has drifted outside the frame boundary
        cx, cy = center_of(self.box)
        w, h   = self.frame_size
        outside = (
            cx < -EXIT_MARGIN or cx > w + EXIT_MARGIN or
            cy < -EXIT_MARGIN or cy > h + EXIT_MARGIN
        )
        self.misses_outside = self.misses_outside + 1 if outside else 0
        return self.box

    def update(
        self,
        detected_box: List[float],
        detected_parts: List[np.ndarray],
        frame_idx: int,
    ):
        """
        Correct track state with a matched detection.
        Runs Kalman update and applies adaptive exponential smoothing.
        """
        z = box_xyxy_to_xywh(detected_box)
        self.x, self.P = self.kf.update(self.x, self.P, z)
        self.box = box_xywh_to_xyxy(self.x[0:4])

        # Store incoming part embeddings in the rolling cache
        for p in detected_parts:
            self.embeds.append(p)

        # Adaptive smoothing: high alpha = stable, low alpha = responsive
        alpha = self._adaptive_smoothing_alpha()
        self.smoothed_box = [
            alpha * self.smoothed_box[i] + (1.0 - alpha) * self.box[i]
            for i in range(4)
        ]

        self.time_since_update = 0
        self.misses_outside    = 0
        self.hits             += 1
        self.age              += 1
        self.last_update_frame = frame_idx

        # Record center for displacement-based static detection
        self._center_history.append(center_of(detected_box))
        self._update_static_state()

        if not self.confirmed and self.hits >= MIN_HITS_TO_CONFIRM:
            # Block confirmation for static objects (posters, signs, etc.)
            if not self._marked_static:
                self.confirmed = True

        # Status reflects current confirmed state, not just the update event
        self.status = "Confirmed" if self.confirmed else "Tracked"

    def mark_missed(self):
        """Called each frame where no detection is matched to this track."""
        self.time_since_update += 1
        self.age               += 1
        self.status             = "Lost"

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Return the simple mean of all cached embeddings, L2-normalized."""
        if not self.embeds:
            return None
        arr  = np.stack(list(self.embeds), axis=0)
        return l2_normalize_vec(np.mean(arr, axis=0))

    def get_weighted_embedding(
        self, conf_weights: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """
        Return a confidence-weighted mean of cached embeddings, L2-normalized.
        Intended for gallery storage — weights should reflect detection quality.
        Falls back to uniform mean if weights are not provided or mismatched.
        """
        if not self.embeds:
            return None

        embeds = list(self.embeds)
        arr    = np.stack(embeds, axis=0)

        if conf_weights is not None and len(conf_weights) == len(embeds):
            w    = np.array(conf_weights, dtype=np.float32)
            w   /= w.sum() + 1e-8
            mean = np.average(arr, axis=0, weights=w)
        else:
            mean = np.mean(arr, axis=0)

        return l2_normalize_vec(mean)

    # ------------------------------------------------------------------
    # Static object detection
    # ------------------------------------------------------------------

    @property
    def is_static(self) -> bool:
        """True if this track has been classified as a static object."""
        return self._marked_static

    def _update_static_state(self):
        """
        Hybrid static detection using both Kalman velocity and displacement.

        A track is considered instantaneously static when:
          1. Kalman velocity magnitude < STATIC_VELOCITY_THRESH, AND
          2. Displacement from oldest stored center < STATIC_DISPLACEMENT_THRESH.

        Once this holds for STATIC_FRAMES_REQUIRED consecutive updates,
        the track is permanently marked as static.
        """
        # Solution 2: Kalman velocity check
        vx    = abs(float(self.x[4]))
        vy    = abs(float(self.x[5]))
        speed = math.hypot(vx, vy)
        velocity_static = speed < STATIC_VELOCITY_THRESH

        # Solution 1: Displacement check over the observation window
        if len(self._center_history) >= 2:
            oldest = self._center_history[0]
            newest = self._center_history[-1]
            displacement = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1])
            displacement_static = displacement < STATIC_DISPLACEMENT_THRESH
        else:
            displacement_static = False

        if velocity_static and displacement_static:
            self._consecutive_static += 1
        else:
            self._consecutive_static = 0

        if self._consecutive_static >= STATIC_FRAMES_REQUIRED:
            self._marked_static = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_smoothing_alpha(self) -> float:
        """
        Choose smoothing alpha based on velocity relative to box size.
        Fast-moving tracks get a low alpha (box follows more closely).
        Slow-moving tracks get a high alpha (box stays stable).
        """
        vx    = abs(self.x[4])
        vy    = abs(self.x[5])
        speed = math.hypot(vx, vy)
        w     = max(1.0, self.x[2])
        h     = max(1.0, self.x[3])
        rel   = speed / (math.hypot(w, h) + 1e-6)
        return SMOOTH_ALPHA_FAST if rel > 0.12 else SMOOTH_ALPHA_SLOW
