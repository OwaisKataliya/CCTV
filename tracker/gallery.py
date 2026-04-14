"""
gallery.py
----------
Track Gallery for persistent person re-identification.

When a track is retired (due to MAX_AGE or frame boundary exit), its
appearance embedding and spatial metadata are saved here. When a new
detection cannot be matched to any active track, the gallery is queried
to check if this person was seen before.

Retrieval uses a 4-gate pipeline to avoid expensive cosine computation
on irrelevant entries:

    Gate 1 (TTL)     : Skip entries older than GALLERY_TTL_SECONDS.
    Gate 2 (Spatial) : Skip entries whose last known center is too far.
    Gate 3 (Area)    : Skip entries with too different a bounding box area.
    Gate 4 (Cosine)  : Compute cosine distance on surviving candidates only.

Gallery structure per entry:
    {
        'embedding'     : np.ndarray,   # 2048D L2-normalized vector
        'last_position' : (cx, cy),     # last known center in pixels
        'last_area'     : float,        # last known bounding box area
        'retired_frame' : int,          # frame index when retired
        'retired_time'  : float,        # wall-clock time (time.time())
        'hit_count'     : int,          # how many detections this track had
    }
"""

import math
import time
from typing import Dict, List, Optional

import numpy as np

from tracker.config import (
    GALLERY_COSINE_THRESH,
    GALLERY_TTL_SECONDS,
    GALLERY_MAX_SIZE,
    GALLERY_MIN_HITS,
    REID_CENTER_MAX_DIST_BASE,
    REID_AREA_RATIO_MAX,
)
from tracker.utils import cosine_distance_vec, center_of, box_area


class TrackGallery:
    """
    Stores appearance representations of retired tracks and provides
    gated retrieval for new detections.
    """

    def __init__(self):
        # Maps track_id -> entry dict
        self._entries: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def save(self, track, frame_idx: int) -> bool:
        """
        Save a retiring track to the gallery.

        Skips the track if it was too short-lived to be reliable.
        Evicts the oldest entry if the gallery is at capacity.

        Args:
            track:     Track object being retired.
            frame_idx: Current frame index.

        Returns:
            True if saved, False if skipped.
        """
        if track.hits < GALLERY_MIN_HITS:
            return False

        # Build gallery embedding: confidence-weighted mean of cached embeds
        embedding = track.get_weighted_embedding()
        if embedding is None:
            return False

        cx, cy = center_of(track.smoothed_box)
        area   = box_area(track.smoothed_box)

        entry = {
            'embedding'     : embedding,
            'last_position' : (cx, cy),
            'last_area'     : max(1.0, area),
            'retired_frame' : frame_idx,
            'retired_time'  : time.time(),
            'hit_count'     : track.hits,
        }

        # Evict oldest if gallery is full
        if len(self._entries) >= GALLERY_MAX_SIZE:
            self._evict_oldest()

        self._entries[track.id] = entry
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        det_box: List[float],
        det_embed: np.ndarray,
        current_time: float,
    ) -> Optional[int]:
        """
        Find the best matching gallery entry for a new detection.

        Applies 4 sequential gates before computing cosine distance.
        Only entries that pass all gates are compared by cosine distance.

        Args:
            det_box:      Detection box [x1, y1, x2, y2].
            det_embed:    L2-normalized mean embedding for the detection.
            current_time: Current wall-clock time (time.time()).

        Returns:
            Matching track ID from gallery, or None if no match found.
        """
        best_tid  = None
        best_dist = GALLERY_COSINE_THRESH   # only accept if below this threshold

        det_cx, det_cy = center_of(det_box)
        det_area       = max(1.0, box_area(det_box))

        for tid, entry in self._entries.items():

            # Gate 1: TTL — skip expired entries
            if current_time - entry['retired_time'] > GALLERY_TTL_SECONDS:
                continue

            # Gate 2: Spatial — skip if last known center is too far
            ecx, ecy = entry['last_position']
            if math.hypot(det_cx - ecx, det_cy - ecy) > REID_CENTER_MAX_DIST_BASE:
                continue

            # Gate 3: Area ratio — skip if box sizes are too different
            area_ratio = max(det_area, entry['last_area']) / min(det_area, entry['last_area'])
            if area_ratio > REID_AREA_RATIO_MAX:
                continue

            # Gate 4: Cosine distance — only reached for plausible candidates
            dist = cosine_distance_vec(entry['embedding'], det_embed)
            if dist < best_dist:
                best_dist = dist
                best_tid  = tid

        return best_tid

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def remove(self, tid: int):
        """Remove a gallery entry after it has been successfully re-attached."""
        self._entries.pop(tid, None)

    def evict_expired(self, current_time: float):
        """Remove all entries older than GALLERY_TTL_SECONDS."""
        expired_ids = [
            tid for tid, entry in self._entries.items()
            if current_time - entry['retired_time'] > GALLERY_TTL_SECONDS
        ]
        for tid in expired_ids:
            del self._entries[tid]

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self):
        """Remove the entry with the earliest retirement time."""
        if not self._entries:
            return
        oldest = min(self._entries, key=lambda k: self._entries[k]['retired_time'])
        del self._entries[oldest]
