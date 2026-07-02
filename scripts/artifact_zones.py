"""
artifact_zones.py — parâmetros optimizados por grid search
(otimizar_explicabilidade.py, N=60 imagens FAKE_EFS, score=0.6426)
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


# Regiões anatomicamente finas — precisam de dilatação maior
THIN_REGIONS = {"olho_esq", "olho_dir", "sobrancelha_esq", "sobrancelha_dir", "boca"}


def extract_artifact_zones(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    region_masks: dict,
    region_scores: dict,
    bbox_padding: int = 12,
) -> List[ArtifactZone]:

    h, w = img_rgb.shape[:2]
    if heatmap.shape != (512, 512):
        heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)

    if not region_scores:
        return []

    max_score = max(data["contrast"] for data in region_scores.values())
    if max_score < 0.05:
        return []

    cand_thresh = max(max_score * 0.40, 0.10)
    candidates = [
        (name, data["contrast"])
        for name, data in region_scores.items()
        if data["contrast"] >= cand_thresh
        and name not in ["pele", "cabelo", "pescoco"]
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    zones: List[ArtifactZone] = []
    h_h, w_h = heatmap.shape

    for name, score in candidates:
        mask_f = region_masks.get(name)
        if mask_f is None:
            continue

        mask_f_resized = cv2.resize(mask_f, (w_h, h_h), interpolation=cv2.INTER_NEAREST)

        # Dilatação adaptativa 
        if name in THIN_REGIONS:
            kernel     = np.ones((5, 5), np.uint8)   
            iterations = 3                            
            percentile = 70                            
        else:
            kernel     = np.ones((3, 3), np.uint8)    
            iterations = 1                             
            percentile = 85                            

        region_bin = cv2.dilate(
            (mask_f_resized > 0.3).astype(np.uint8), kernel, iterations=iterations
        ).astype(bool)

        if not region_bin.any():
            continue

        # Threshold adaptativo local com piso de 0.05 (era 0.20)
        region_heatmap_vals = heatmap[region_bin]
        local_thresh = max(
            np.percentile(region_heatmap_vals, percentile),
            0.05
        )
        zone_specific_mask = (heatmap >= local_thresh) & region_bin

        if not zone_specific_mask.any():
            continue

        # Filtro de densidade
        region_area = int(region_bin.sum())
        zone_area   = int(zone_specific_mask.sum())
        density     = zone_area / (region_area + 1e-6)

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
                f"Artefactos isolados na anatomia '{name}' "
                f"(threshold local={local_thresh:.3f}, "
                f"densidade={density:.3f}). "
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