# Advanced CCTV Person Tracker

A modular, production-ready person tracking system built with YOLO11, part-based Re-Identification (ReID), Kalman filtering, Hungarian assignment, and a persistent Track Gallery. Deployed as an interactive Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tracking Pipeline](#tracking-pipeline)
- [Track Gallery](#track-gallery)
- [Project Structure](#project-structure)
- [Quick Start from GitHub](#quick-start-from-github)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Hardware Requirements](#hardware-requirements)

---

## Overview

Standard object trackers fail under real-world surveillance conditions. Occlusions, re-entries after long absences, crossing paths, and appearance changes all cause identity switches and fragmented tracks. This project implements a full multi-stage tracking pipeline to address these challenges.

**Core capabilities:**
- Consistent person IDs throughout the video, even across occlusions
- Re-identification of returning persons using a persistent Track Gallery
- All parameters tunable live via a Streamlit sidebar — no code changes needed
- Modular codebase: tracking logic and UI are fully separated

---

## Architecture

The system is divided into two packages with a strict dependency direction:

```
app.py  ->  ui/*  ->  tracker/*
```

`tracker/` contains all detection, tracking, and re-identification logic. It has no dependency on Streamlit or any UI library. `ui/` contains all Streamlit rendering code and calls into `tracker/`. This makes the core tracking engine independently testable and reusable.

### Component Overview

| Component | Role |
|---|---|
| `tracker/config.py` | Single source of truth for all tunable constants |
| `tracker/utils.py` | Pure stateless math utilities (IoU, box conversions, cosine distance) |
| `tracker/embedder.py` | Part-based ResNet50 feature extractor |
| `tracker/kalman.py` | Constant-velocity Kalman filter |
| `tracker/track.py` | Per-track state manager (Kalman state, embedding cache, lifecycle) |
| `tracker/gallery.py` | Persistent appearance gallery with gated retrieval |
| `tracker/reid_tracker.py` | Pipeline orchestrator — runs all steps per frame |
| `ui/sidebar.py` | Sidebar controls, returns settings dict |
| `ui/uploader.py` | Upload panel, preview, process button |
| `ui/results.py` | Results panel (4 tabs) |
| `ui/pipeline.py` | Video processing loop, YOLO inference, CSV export |

---

## Tracking Pipeline

Every frame is processed through the following 11 steps in order:

### Step 1 — YOLO11 Detection
YOLO11m scans the frame for all persons (class 0). Two confidence thresholds are used:
- **Low threshold** (default 0.12): passed to YOLO as the prediction floor.
- **High threshold** (default 0.45): gate applied after YOLO returns — only detections above this are passed to the tracker.

### Step 2 — Part-based Feature Extraction
Each detection bounding box is divided into 3 horizontal strips (head, torso, legs). Each strip is passed through **ResNet50** (ImageNet pretrained, classification head removed) to produce a **2048-dimensional L2-normalized feature vector**. Using part-level features instead of a single global embedding makes the system more robust to partial occlusion — if the head is hidden, the torso and legs still contribute to matching.

### Step 3 — Kalman Prediction
Each active track runs a **Kalman filter** with an 8-dimensional state vector:
```
state = [cx, cy, w, h, vx, vy, vw, vh]
```
Position is predicted from previous position + estimated velocity. This produces a predicted bounding box for each track before any detections are matched.

### Step 4 — Cost Matrix Construction
An N × M cost matrix is built for all (track, detection) pairs:
```
cost[i, j] = 0.4 × (1 - IoU) + 0.6 × ReID_cosine_distance
```
- **IoU** measures spatial overlap between predicted track box and detection box.
- **ReID cosine distance** is the minimum cosine distance between the track's average embedding and the detection's part embeddings.
- Weights (0.4 / 0.6) make appearance the dominant signal while IoU prevents absurd spatial assignments.

### Step 5 — Hungarian Assignment
The **Hungarian algorithm** (via `scipy.optimize.linear_sum_assignment`) solves the cost matrix globally. Each track and detection is assigned at most once. Global optimization means the system won't greedily assign a track to its locally best detection if doing so results in a worse total assignment.

### Step 6 — Acceptance Gating
Each proposed assignment from Step 5 is accepted only if one of these conditions is true:
- `IoU >= iou_thr` (direct spatial match), **OR**
- All of: ReID cosine distance below strict threshold + center distance within limit + area ratio within limit + track was lost recently enough

This two-path acceptance means IoU handles normal continuous tracking, while ReID handles recovery from missed detections.

### Step 7 — Track Update
Accepted matches trigger a **Kalman correction** (measurement update) on the track. The bounding box is further refined with **adaptive exponential smoothing**:
- Slow-moving tracks use high alpha (0.70) for stable, jitter-free boxes.
- Fast-moving tracks use low alpha (0.18) for responsive, accurate boxes.

### Step 8 — Fallback IoU Attach
Unmatched detections are compared against lost tracks using raw IoU on the **smoothed box** (not Kalman-predicted box). If overlap exceeds 0.18 and center distance is within 160px, the detection is attached to that track. This prevents new ID creation when Hungarian rejected a moderate-overlap pair because a better competing assignment existed.

### Step 9 — Gallery Query
Still-unmatched detections query the **Track Gallery** (see below). If a retired track matches by appearance and all spatial gates pass, it is **revived with its original ID**. This is what makes long-term re-identification possible.

### Step 10 — New Track Creation
Detections that survived all previous steps become new tracks. A detection that overlaps a confirmed active track by more than 0.45 IoU is ignored to prevent ghost tracks.

### Step 11 — Track Retirement
Tracks exceeding `MAX_AGE` (default 150) frames without a match are retired. Before deletion, they are saved to the Track Gallery. Boundary-exit retirement (when the predicted center is consistently outside the frame) is handled separately in Step 3.

---

## Track Gallery

The Track Gallery enables re-identification of persons returning after long absences (up to 10 minutes by default).

### Storage (on track retirement)

1. Tracks with fewer than `GALLERY_MIN_HITS` detections are not saved — they are too unreliable.
2. The track's cached embeddings are averaged into a single **2048D L2-normalized vector** (optionally confidence-weighted).
3. The vector is stored alongside: last known center position, last bounding box area, retirement timestamp, and hit count.
4. If the gallery is at capacity (100 entries), the oldest entry is evicted first.

### Retrieval (4-gate pipeline)

When a new detection cannot be matched to any active track, the gallery is queried. To avoid expensive 2048D cosine computation on every entry, three cheap gates run first:

| Gate | Check | Cost |
|---|---|---|
| 1. TTL | Entry older than 600 seconds? Skip. | O(1) |
| 2. Spatial | Last known center > 160px from detection center? Skip. | O(1) |
| 3. Area ratio | Box area differs by > 2.5x? Skip. | O(1) |
| 4. Cosine | Distance below `GALLERY_COSINE_THRESH` (0.38)? Match. | O(D) |

In typical scenes, gates 1–3 eliminate 90%+ of gallery entries. Cosine runs on only the small remaining set.

### Re-attachment

On a gallery match, a new `Track` object is created with the **original track ID** (not `next_id`). The track is marked confirmed immediately. The gallery entry is removed to prevent double-assignment.

---

## Project Structure

```
CCTV/
├── app.py                   Entry point — initialises session state and renders layout
├── requirements.txt         Python dependencies
├── README.md
├── .gitignore
│
├── tracker/
│   ├── __init__.py
│   ├── config.py            All tunable constants — edit here to change behaviour
│   ├── utils.py             Pure utility functions: iou, box ops, cosine distance
│   ├── embedder.py          PartEmbedder: ResNet50 per-body-part feature extraction
│   ├── kalman.py            KalmanBox: constant-velocity Kalman filter
│   ├── track.py             Track: per-track state, embedding cache, smoothing
│   ├── gallery.py           TrackGallery: save, gated query, TTL eviction
│   └── reid_tracker.py      PersonReIDTracker: 11-step per-frame pipeline
│
└── ui/
    ├── __init__.py
    ├── sidebar.py           render_sidebar() — returns settings dict
    ├── uploader.py          render_uploader() — upload panel and process button
    ├── results.py           render_results() — 4-tab results panel
    └── pipeline.py          run_tracking_pipeline() — YOLO + tracker video loop
```

---

## Quick Start from GitHub

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support:** For CUDA-accelerated inference, install the correct PyTorch build for your CUDA version **before** the above command. See https://pytorch.org/get-started/locally/

### 4. Download the YOLO model

Download `yolo11m.pt` from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place it in the project root directory.

Alternatively, the model will be downloaded automatically on first run if you have an internet connection.

### 5. Run the application

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Upload a video, adjust settings in the sidebar if needed, and click **Start Tracking**.

---

## Configuration

All parameters are defined in `tracker/config.py` and exposed as live controls in the Streamlit sidebar. Changes take effect on the next processing run — no code edits needed.

### Detection

| Parameter | Default | Description |
|---|---|---|
| `HIGH_CONF_THR` | 0.45 | Minimum YOLO confidence to pass a detection to the tracker |
| `LOW_CONF_THR` | 0.12 | YOLO prediction confidence floor |

### Matching

| Parameter | Default | Description |
|---|---|---|
| `IOU_THR` | 0.30 | Minimum IoU for direct track-detection match |
| `MATCH_WEIGHT_IOU` | 0.4 | Weight of spatial cost in assignment matrix |
| `MATCH_WEIGHT_REID` | 0.6 | Weight of appearance cost in assignment matrix |
| `REID_STRICT_THRESH` | 0.45 | Max cosine distance for ReID-based re-attachment |
| `REID_REATTACH_MAX_AGE` | 30 | Track must have been lost within this many frames for ReID re-attach |
| `REID_CENTER_MAX_DIST` | 160 | Max pixel distance for ReID spatial gate |
| `REID_AREA_RATIO_MAX` | 2.5 | Max area ratio for ReID spatial gate |

### Track Lifecycle

| Parameter | Default | Description |
|---|---|---|
| `MIN_HITS_TO_CONFIRM` | 2 | Detections before a track is confirmed |
| `MAX_AGE` | 150 | Frames without match before retirement |
| `FALLBACK_IOU_FOR_NEW` | 0.18 | IoU threshold for fallback attach (prevents duplicate IDs) |

### Track Gallery

| Parameter | Default | Description |
|---|---|---|
| `GALLERY_COSINE_THRESH` | 0.38 | Max cosine distance for gallery re-attachment |
| `GALLERY_TTL_SECONDS` | 600 | Gallery entry expires after this many seconds |
| `GALLERY_MAX_SIZE` | 100 | Maximum gallery entries |
| `GALLERY_MIN_HITS` | 2 | Minimum hits for a track to be saved to gallery |

---

## Outputs

### Annotated Video (MP4)

- Unique color-coded bounding box per confirmed track
- Track ID label above each box
- Horizontal lines dividing body parts (showing ReID feature regions)
- Top-left overlay: frame count, live confirmed track count, current gallery size

### Tracking Log (CSV)

| Column | Description |
|---|---|
| `ID` | Track identifier (e.g. `ID_001`) |
| `Entry_Time` | Wall-clock timestamp of first detection |
| `Exit_Time` | Wall-clock timestamp of retirement, or `Active` |
| `Duration` | Time in scene in seconds, or `Active` |
| `Status` | `Active` or `Completed` |

---

## Hardware Requirements

| Component | Recommended | Minimum |
|---|---|---|
| GPU | NVIDIA RTX 3060 8GB or higher | NVIDIA RTX 2060 6GB |
| RAM | 16 GB | 8 GB |
| Storage | SSD | HDD |

Processing is supported on CPU but will run significantly slower (under 5 fps) and is not suitable for real-time use. The GPU is used by both YOLO11 detection and ResNet50 feature extraction.

**Processing time estimate:**
```
Total Processing Time (seconds) = Total Video Frames / Inference FPS
```
Example: 5-minute video at 30 fps = 9000 frames. At 45 fps inference = ~200 seconds.