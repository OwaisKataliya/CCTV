"""
config.py
---------
Central configuration for all tunable tracking parameters.
Modify values here to adjust behaviour without touching algorithm code.
"""

import torch

# ---- Device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Detection thresholds ----
HIGH_CONF_THR = 0.45      # Detections below this are not passed to the tracker
LOW_CONF_THR  = 0.12      # YOLO confidence floor for initial prediction

# ---- Part-based ReID ----
NUM_PARTS        = 3      # Number of horizontal strips per person crop
EMBED_CACHE_SIZE = 10     # Sliding window — max embeddings stored per active track

# ---- Matching cost weights ----
MATCH_WEIGHT_IOU  = 0.4
MATCH_WEIGHT_REID = 0.6

# ---- Active track matching thresholds ----
IOU_THR               = 0.30   # Minimum IoU to accept a match directly
REID_STRICT_THRESH    = 0.45   # Max cosine distance for ReID-based re-attachment
REID_REATTACH_MAX_AGE = 30     # Track must have been lost within this many frames
REID_CENTER_MAX_DIST  = 160    # Max pixel distance between track and detection centers
REID_AREA_RATIO_MAX   = 2.5    # Max area ratio between track box and detection box

# ---- Fallback IoU attach ----
# Prevents premature new ID creation when overlap is moderate
FALLBACK_IOU_FOR_NEW     = 0.18
FALLBACK_CENTER_MAX_DIST = 160
MAX_FALLBACK_MISSES      = 8    # Only consider tracks missed <= this many frames

# ---- Track lifecycle ----
MIN_HITS_TO_CONFIRM = 2     # Detections needed before a track is marked confirmed
MAX_AGE             = 150   # Frames without update before retirement

# ---- Adaptive exponential smoothing ----
SMOOTH_ALPHA_SLOW = 0.70    # High smoothing for slow-moving tracks (stable)
SMOOTH_ALPHA_FAST = 0.18    # Low smoothing for fast-moving tracks (responsive)

# ---- Kalman filter noise ----
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEAS_NOISE    = 1e-1

# ---- Exit / frame boundary detection ----
EXIT_MARGIN         = 40    # Pixels outside frame boundary before counting as exit
EXIT_MIN_MISSES     = 3     # Consecutive outside-boundary predictions to retire track
HARD_RETIRE_ON_EXIT = False # True = retire immediately on first outside-boundary prediction

# ---- Track Gallery (persistent re-identification) ----
GALLERY_COSINE_THRESH  = 0.38   # Max cosine distance for gallery re-attachment
GALLERY_TTL_SECONDS    = 600    # Gallery entry expires after this many seconds (10 min)
GALLERY_MAX_SIZE       = 100    # Maximum entries stored in gallery at any time
GALLERY_MIN_HITS       = 2      # Track must have at least this many hits to be saved
GALLERY_EVICT_INTERVAL = 300    # Run TTL eviction every N frames
