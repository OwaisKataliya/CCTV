"""
reid_tracker.py
---------------
PersonReIDTracker — orchestrates the full per-frame tracking pipeline.

Pipeline per frame (in order):
  1.  Predict all active tracks (Kalman forward pass).
  2.  Mark all tracks as missed for this frame.
  3.  Retire tracks whose predicted center has exited the frame boundary.
      Retired tracks are saved to the gallery before deletion.
  4.  Build N x M cost matrix:
          cost[i, j] = 0.4 * (1 - IoU) + 0.6 * ReID_cosine_distance
  5.  Run Hungarian algorithm for globally optimal track-detection assignment.
  6.  Gate each accepted assignment:
          Accept if: IoU >= iou_thr  OR  (strict ReID + spatial gates pass)
  7.  Update all accepted (track, detection) pairs.
  8.  Fallback IoU attach for unmatched detections:
          Prevents premature new ID creation when overlap is moderate.
  9.  Gallery query for still-unmatched detections:
          Resurrect old track IDs if appearance matches a gallery entry.
  10. Create new tracks for truly unmatched detections.
  11. Retire tracks that have exceeded MAX_AGE. Save them to gallery.
  12. Periodically evict expired gallery entries (every GALLERY_EVICT_INTERVAL frames).
"""

import math
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.config import (
    DEVICE,
    NUM_PARTS,
    IOU_THR,
    REID_STRICT_THRESH,
    REID_REATTACH_MAX_AGE,
    REID_CENTER_MAX_DIST,
    REID_AREA_RATIO_MAX,
    MATCH_WEIGHT_IOU,
    MATCH_WEIGHT_REID,
    FALLBACK_IOU_FOR_NEW,
    FALLBACK_CENTER_MAX_DIST,
    MAX_FALLBACK_MISSES,
    MAX_AGE,
    MIN_HITS_TO_CONFIRM,
    EXIT_MIN_MISSES,
    HARD_RETIRE_ON_EXIT,
    GALLERY_EVICT_INTERVAL,
)
from tracker.embedder import PartEmbedder
from tracker.track import Track
from tracker.gallery import TrackGallery
from tracker.utils import iou, cosine_distance_vec, center_of, box_area, l2_normalize_vec


class PersonReIDTracker:
    """
    Full person tracking pipeline combining YOLO detections, part-based
    ReID embeddings, Kalman filtering, Hungarian assignment, and a
    persistent Track Gallery for long-term re-identification.

    Args:
        device:             'cuda' or 'cpu'.
        frame_size:         (width, height) of the video.
        fps:                Video frame rate.
        iou_thr:            Minimum IoU to directly accept a match.
        reid_strict_thresh: Max cosine distance for ReID re-attachment.
        max_age:            Frames before a lost track is retired.
        min_hits:           Detections before a track is confirmed.
    """

    def __init__(
        self,
        device: str = DEVICE,
        frame_size: tuple = (1920, 1080),
        fps: float = 30.0,
        iou_thr: float = IOU_THR,
        reid_strict_thresh: float = REID_STRICT_THRESH,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS_TO_CONFIRM,
    ):
        self.device             = device
        self.frame_size         = frame_size
        self.fps                = fps
        self.dt                 = 1.0 / (fps if fps > 0 else 30.0)

        # These thresholds can be overridden from the UI sidebar
        self.iou_thr            = iou_thr
        self.reid_strict_thresh = reid_strict_thresh
        self.max_age            = max_age
        self.min_hits           = min_hits

        self.embedder           = PartEmbedder(device=device, num_parts=NUM_PARTS)
        self.tracks: Dict[int, Track] = {}
        self.next_id            = 1
        self.gallery            = TrackGallery()
        self._frames_processed  = 0

    def reset(self):
        """Clear all tracks, gallery, and reset the ID counter."""
        self.tracks            = {}
        self.next_id           = 1
        self.gallery           = TrackGallery()
        self._frames_processed = 0

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------

    def step(
        self,
        detections: List[List[float]],
        parts_features: List[List[np.ndarray]],
        frame_idx: int,
    ) -> Dict[int, Track]:
        """
        Process one frame and update all tracks.

        Args:
            detections:     List of detection boxes [[x1,y1,x2,y2], ...].
            parts_features: Part embeddings per detection (same order).
            frame_idx:      Current frame number (1-indexed).

        Returns:
            Dict {track_id: Track} of all currently active tracks.
        """
        self._frames_processed += 1
        current_time = time.time()

        # ----------------------------------------------------------
        # Steps 1-3: Predict, mark missed, retire boundary exits
        # ----------------------------------------------------------
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            tr.predict()
            tr.mark_missed()

            should_retire = (
                (HARD_RETIRE_ON_EXIT and tr.misses_outside > 0) or
                (not HARD_RETIRE_ON_EXIT and tr.misses_outside >= EXIT_MIN_MISSES)
            )
            if should_retire:
                tr.exit_time = datetime.now()
                self.gallery.save(tr, frame_idx)
                del self.tracks[tid]

        track_ids = list(self.tracks.keys())
        N = len(track_ids)
        M = len(detections)

        if N == 0 and M == 0:
            return self.tracks

        # ----------------------------------------------------------
        # Step 4: Build cost matrix (N tracks x M detections)
        # ----------------------------------------------------------
        cost = np.ones((N, M), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            tr     = self.tracks[tid]
            tr_emb = tr.get_average_embedding()

            for j in range(M):
                det_box   = detections[j]
                det_parts = parts_features[j]
                iou_score = iou(tr.box, det_box)
                iou_cost  = 1.0 - iou_score

                reid_cost = 1.0
                if tr_emb is not None and det_parts:
                    dists     = [cosine_distance_vec(tr_emb, p) for p in det_parts]
                    reid_cost = float(min(dists))

                cost[i, j] = MATCH_WEIGHT_IOU * iou_cost + MATCH_WEIGHT_REID * reid_cost

        # ----------------------------------------------------------
        # Steps 5-6: Hungarian assignment + acceptance gating
        # ----------------------------------------------------------
        matches            = []
        unmatched_track_ii = list(range(N))
        unmatched_det_jj   = list(range(M))

        if N > 0 and M > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_tracks  = set()
            assigned_dets    = set()

            for r, c in zip(row_ind, col_ind):
                tid       = track_ids[r]
                tr        = self.tracks[tid]
                det_box   = detections[c]
                det_parts = parts_features[c]

                iou_score = iou(tr.box, det_box)
                tr_emb    = tr.get_average_embedding()
                best_reid = 1.0
                if tr_emb is not None and det_parts:
                    dists     = [cosine_distance_vec(tr_emb, p) for p in det_parts]
                    best_reid = float(min(dists))

                # Spatial gating for ReID-based re-attachment
                tcx, tcy      = center_of(tr.box)
                dcx, dcy      = center_of(det_box)
                center_dist   = math.hypot(tcx - dcx, tcy - dcy)
                tarea         = max(1.0, box_area(tr.box))
                darea         = max(1.0, box_area(det_box))
                area_ratio    = max(tarea / darea, darea / tarea)
                recent_enough = tr.time_since_update <= REID_REATTACH_MAX_AGE

                reid_ok = (
                    tr_emb is not None and
                    det_parts and
                    recent_enough and
                    best_reid   <= self.reid_strict_thresh and
                    center_dist <= REID_CENTER_MAX_DIST and
                    area_ratio  <= REID_AREA_RATIO_MAX
                )

                if iou_score >= self.iou_thr or reid_ok:
                    matches.append((r, c))
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            unmatched_track_ii = [i for i in range(N) if i not in assigned_tracks]
            unmatched_det_jj   = [j for j in range(M) if j not in assigned_dets]

        # ----------------------------------------------------------
        # Step 7: Update matched tracks
        # ----------------------------------------------------------
        for r, c in matches:
            self.tracks[track_ids[r]].update(detections[c], parts_features[c], frame_idx)

        # ----------------------------------------------------------
        # Step 8: Fallback IoU attach for unmatched detections
        # Prevents new ID creation when Hungarian rejected a moderate-IoU
        # pair due to a competing better assignment elsewhere.
        # ----------------------------------------------------------
        fallback_attached = set()
        for di in list(unmatched_det_jj):
            det_box  = detections[di]
            best_tid = None
            best_iou = 0.0

            for tid, tr in self.tracks.items():
                if tr.time_since_update == 0:
                    continue   # already updated this frame
                if tr.time_since_update > MAX_FALLBACK_MISSES:
                    continue
                if tr.retired:
                    continue
                i_score = iou(tr.smoothed_box, det_box)
                if i_score > best_iou:
                    best_iou = i_score
                    best_tid = tid

            if best_tid is not None and best_iou >= FALLBACK_IOU_FOR_NEW:
                tr       = self.tracks[best_tid]
                tcx, tcy = center_of(tr.smoothed_box)
                dcx, dcy = center_of(det_box)
                if math.hypot(tcx - dcx, tcy - dcy) <= FALLBACK_CENTER_MAX_DIST:
                    tr.update(det_box, parts_features[di], frame_idx)
                    fallback_attached.add(di)

        unmatched_det_jj = [d for d in unmatched_det_jj if d not in fallback_attached]

        # ----------------------------------------------------------
        # Step 9: Gallery query for still-unmatched detections
        # Resurrects the original track ID when appearance matches
        # a previously retired track in the gallery.
        # ----------------------------------------------------------
        gallery_revived = set()
        for di in list(unmatched_det_jj):
            det_box   = detections[di]
            det_parts = parts_features[di]

            if not det_parts:
                continue

            # Compute a single representative embedding for gallery comparison
            arr       = np.stack(det_parts, axis=0)
            det_embed = l2_normalize_vec(np.mean(arr, axis=0))

            matched_gallery_tid = self.gallery.query(det_box, det_embed, current_time)
            if matched_gallery_tid is not None:
                # Re-create the track with its original ID
                revived = Track(
                    init_box   = det_box,
                    part_feats = det_parts,
                    track_id   = matched_gallery_tid,
                    frame_idx  = frame_idx,
                    dt         = self.dt,
                    frame_size = self.frame_size,
                )
                # Mark as confirmed immediately so it renders with full style
                revived.hits      = self.min_hits
                revived.confirmed = True
                revived.status    = "Confirmed"

                self.tracks[matched_gallery_tid] = revived
                self.gallery.remove(matched_gallery_tid)
                gallery_revived.add(di)

        unmatched_det_jj = [d for d in unmatched_det_jj if d not in gallery_revived]

        # ----------------------------------------------------------
        # Step 10: Create new tracks for truly unmatched detections
        # ----------------------------------------------------------
        for di in unmatched_det_jj:
            det_box   = detections[di]
            det_parts = parts_features[di]

            # Skip if detection significantly overlaps a confirmed active track
            overlapping = any(
                tr.confirmed and iou(tr.smoothed_box, det_box) > 0.45
                for tr in self.tracks.values()
            )
            if overlapping:
                continue

            new_track = Track(
                init_box   = det_box,
                part_feats = det_parts,
                track_id   = self.next_id,
                frame_idx  = frame_idx,
                dt         = self.dt,
                frame_size = self.frame_size,
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        # ----------------------------------------------------------
        # Steps 11-12: Retire aged tracks, periodic gallery eviction
        # ----------------------------------------------------------
        for tid in [t for t, tr in list(self.tracks.items()) if tr.time_since_update > self.max_age]:
            tr           = self.tracks[tid]
            tr.exit_time = datetime.now()
            self.gallery.save(tr, frame_idx)
            del self.tracks[tid]

        if self._frames_processed % GALLERY_EVICT_INTERVAL == 0:
            self.gallery.evict_expired(current_time)

        return self.tracks
