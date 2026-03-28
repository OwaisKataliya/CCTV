"""
utils.py
--------
Pure stateless utility functions used across the tracker pipeline.
No imports from other tracker modules — safe to import anywhere.
"""

import math
import numpy as np


def iou(boxA, boxB) -> float:
    """Compute Intersection over Union between two xyxy boxes."""
    x1, y1, x2, y2 = boxA
    xa, ya, xb, yb = boxB
    xx1, yy1 = max(x1, xa), max(y1, ya)
    xx2, yy2 = min(x2, xb), min(y2, yb)
    iw    = max(0.0, xx2 - xx1)
    ih    = max(0.0, yy2 - yy1)
    inter = iw * ih
    a1    = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2    = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def box_xyxy_to_xywh(box) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = box
    w  = max(1.0, x2 - x1)
    h  = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)


def box_xywh_to_xyxy(xywh) -> list:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = xywh
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    """L2-normalize a numpy vector. Adds epsilon to avoid division by zero."""
    v    = v.astype(np.float32)
    norm = np.linalg.norm(v) + 1e-8
    return v / norm


def cosine_distance_vec(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance between two L2-normalized vectors.
    Returns 0 for identical vectors, 1 for orthogonal, 2 for opposite.
    """
    return 1.0 - float(np.dot(a, b))


def center_of(box) -> tuple:
    """Return (cx, cy) for an xyxy box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box) -> float:
    """Return area of an xyxy box."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
