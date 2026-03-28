"""
embedder.py
-----------
Part-based appearance feature extractor using ResNet50.

Each person bounding box is split into NUM_PARTS horizontal strips.
Each strip is passed through ResNet50 (ImageNet pretrained, classification
head removed) to produce a 2048-dimensional feature vector.
Each vector is independently L2-normalized before returning.

Usage:
    embedder = PartEmbedder(device='cuda', num_parts=3)
    part_embeds = embedder.extract(bgr_frame, [x1, y1, x2, y2])
    gallery_embed = embedder.compute_weighted_mean(part_embeds, weights)
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
from typing import List, Optional

from tracker.config import NUM_PARTS
from tracker.utils import l2_normalize_vec


class PartEmbedder:
    """
    Extracts part-based appearance embeddings from a person crop.

    Attributes:
        available: False if ResNet50 failed to load. extract() will return
                   random vectors as a safe fallback in that case.
    """

    def __init__(self, device: str = 'cpu', num_parts: int = NUM_PARTS):
        self.device    = device
        self.num_parts = num_parts
        self.available = False

        try:
            r50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove the final FC classification layer — use 2048D pooled output
            self.model = nn.Sequential(*list(r50.children())[:-1]).to(self.device)
            self.model.eval()
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize((256, 128)),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                ),
            ])
            self.available = True
        except Exception as e:
            print(f"[PartEmbedder] ResNet50 load failed: {e}")

    def extract(self, frame: np.ndarray, box: List[float]) -> List[np.ndarray]:
        """
        Extract part-based embeddings for one detection.

        Args:
            frame: Full BGR frame (H x W x 3).
            box:   Detection box [x1, y1, x2, y2].

        Returns:
            List of L2-normalized 2048D numpy arrays (one per body part).
            Returns a single random vector on failure.
        """
        if not self.available:
            return [np.random.rand(2048).astype(np.float32)]

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)

            if (x2 - x1) < 8 or (y2 - y1) < 8:
                return [np.random.rand(2048).astype(np.float32)]

            box_h  = y2 - y1
            part_h = max(1, box_h // self.num_parts)
            parts  = []

            with torch.no_grad():
                for i in range(self.num_parts):
                    py1  = y1 + i * part_h
                    py2  = (y1 + (i + 1) * part_h) if i < self.num_parts - 1 else y2
                    crop = frame[py1:py2, x1:x2]

                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensor   = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                    feat     = self.model(tensor)
                    feat     = feat.view(feat.shape[0], -1).cpu().numpy().reshape(-1)

                    if feat.size == 0:
                        continue

                    parts.append(l2_normalize_vec(feat))

            return parts if parts else [np.random.rand(2048).astype(np.float32)]

        except Exception:
            return [np.random.rand(2048).astype(np.float32)]

    def compute_weighted_mean(
        self,
        embeds: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Compute a weighted mean of multiple embeddings and L2-normalize.
        Used when building the gallery representation for a retiring track.

        Args:
            embeds:  List of 2048D L2-normalized embeddings.
            weights: Optional confidence weights (same length as embeds).

        Returns:
            Single L2-normalized 2048D embedding.
        """
        if not embeds:
            return l2_normalize_vec(np.random.rand(2048).astype(np.float32))

        arr = np.stack(embeds, axis=0)  # shape: (N, 2048)

        if weights is not None and len(weights) == len(embeds):
            w    = np.array(weights, dtype=np.float32)
            w   /= (w.sum() + 1e-8)
            mean = np.average(arr, axis=0, weights=w)
        else:
            mean = np.mean(arr, axis=0)

        return l2_normalize_vec(mean)
