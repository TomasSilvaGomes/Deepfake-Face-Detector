"""
artifact_zones.py — versão de alta precisão (sobreposição estrita com H_isolamento)
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class ArtifactZone:
    name: str
    score: float
    bbox: tuple
    centroid: tuple
    mask_bin: np.ndarray
    description: str = ""


def extract_artifact_zones(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    region_masks: dict,
    region_scores: dict,
    bbox_padding: int = 12,
    local_percentile: float = 60,   
    min_floor: float = 0.20,        
) -> List[ArtifactZone]:

    h, w = img_rgb.shape[:2]
    if heatmap.shape != (512, 512):
        heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)

    if not region_scores:
        return []

    max_score = max(data["contrast"] for data in region_scores.values())
    if max_score < 0.05:
        return []

    candidates = [
        (name, data["contrast"])
        for name, data in region_scores.items()
        if data["contrast"] >= (max_score * 0.30)   
        and name not in ["pele", "cabelo", "pescoco"]
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    zones: List[ArtifactZone] = []
    h_h, w_h = heatmap.shape

    THIN_REGIONS = {"olho_esq", "olho_dir", "sobrancelha_esq", "sobrancelha_dir", "boca"}

    for name, score in candidates:
        mask_f = region_masks.get(name)
        if mask_f is None:
            continue

        mask_f_resized = cv2.resize(mask_f, (w_h, h_h), interpolation=cv2.INTER_NEAREST)

        # Dilatação adaptativa: maior para regiões finas, menor para regiões largas
        if name in THIN_REGIONS:
            kernel = np.ones((7, 7), np.uint8)
            iterations = 2
            percentile = 60
        else:
            kernel = np.ones((3, 3), np.uint8)
            iterations = 1
            percentile = 75

        region_bin = cv2.dilate(
            (mask_f_resized > 0.3).astype(np.uint8), kernel, iterations=iterations
        ).astype(bool)

        if not region_bin.any():
            continue

        region_heatmap_vals = heatmap[region_bin]
        local_thresh = max(
            np.percentile(region_heatmap_vals, percentile),
            min_floor
        )
        zone_specific_mask = (heatmap >= local_thresh) & region_bin

        if not zone_specific_mask.any():
            continue
        
        region_area = int(region_bin.sum())
        zone_area   = int(zone_specific_mask.sum())
        density     = zone_area / (region_area + 1e-6)

        # Regiões grandes (bochechas) precisam de densidade mínima maior
        min_density = 0.06 if region_area > 3000 else 0.02
        if density < min_density:
            continue

        ys, xs = np.where(zone_specific_mask)
        x1 = max(0, int(xs.min()) - bbox_padding)
        y1 = max(0, int(ys.min()) - bbox_padding)
        x2 = min(w, int(xs.max()) + bbox_padding)
        y2 = min(h, int(ys.max()) + bbox_padding)

        zones.append(ArtifactZone(
            name=name,
            score=score,
            bbox=(x1, y1, x2, y2),
            centroid=(int(xs.mean()), int(ys.mean())),
            mask_bin=zone_specific_mask,
            description=(
                f"Artefactos isolados na anatomia '{name}' (threshold local={local_thresh:.3f}). "
                f"Gravidade: {score:.3f}."
            ),
        ))

    return zones

def segment_zones_with_probability(
    heatmap_contrastive: np.ndarray,
    zones: List[ArtifactZone],
    prob_threshold: float = 0.30,
) -> dict:
    return {zone.name: zone.mask_bin for zone in zones}