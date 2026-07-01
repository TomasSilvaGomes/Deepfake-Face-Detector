import sys
from pathlib import Path

# 1. Injeta o caminho ANTES de qualquer import local ou externo
ROOT_DIR = Path(__file__).resolve().parent.parent
CLIP_SURGERY_PATH = str(ROOT_DIR / "CLIP_Surgery")

if CLIP_SURGERY_PATH not in sys.path:
    sys.path.insert(0, CLIP_SURGERY_PATH)

import clip as clip_surgery
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from PIL import Image
from segmenter import FaceSegmenter
from huggingface_hub import hf_hub_download

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    CLIP_MEAN,
    CLIP_STD,
    DEVICE,
    FAKE_PROMPT_KEYWORDS,
    REAL_PROMPTS,
    SURGERY_PROMPTS,
    SURGERY_RES,
)

# ════════════════════════════════════════════════════════════
# MEDIAPIPE — Máscara facial (API Tasks)
# ════════════════════════════════════════════════════════════

_landmarker = None
_segmenter_instance = None

def get_segmenter():
    global _segmenter_instance
    if _segmenter_instance is None:
        print(" A instanciar BiSeNet FaceSegmenter...")
        model_path = hf_hub_download(repo_id="liamu/Deepfake-Pesos", filename="79999_iter.pth")
        _segmenter_instance = FaceSegmenter(model_path=model_path, device=DEVICE)
        print(" BiSeNet pronto.")
    return _segmenter_instance

def get_region_masks(img_rgb: np.ndarray) -> dict:
    return get_segmenter().get_masks(img_rgb)

def get_landmarker():
    global _landmarker
    if _landmarker is None:
        # Caminho para o ficheiro .task na raiz
        model_path = hf_hub_download(
                repo_id="liamu/Deepfake-Pesos", 
                filename="face_landmarker.task")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker

def build_face_mask(img_rgb: np.ndarray) -> np.ndarray:
    try:
        h, w = img_rgb.shape[:2]
        if img_rgb is None or img_rgb.size == 0:
            return np.ones((h, w), dtype=np.float32)

        # Converter para formato do MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        landmarker = get_landmarker()
        detection_result = landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return np.ones((h, w), dtype=np.float32)

        lm = detection_result.face_landmarks[0]
        points = np.array(
            [(int(point.x * w), int(point.y * h)) for point in lm], dtype=np.int32
        )
        hull = cv2.convexHull(points)

        y_min = hull[:, 0, 1].min()
        y_max = hull[:, 0, 1].max()
        face_h = y_max - y_min

        # 10% padding para cima
        forehead_expansion = int(face_h * 0.10)
        hull_expanded = hull.copy()
        
        top_mask = hull_expanded[:, 0, 1] < (y_min + face_h * 0.35)
        hull_expanded[top_mask, 0, 1] = np.maximum(
            0, hull_expanded[top_mask, 0, 1] - forehead_expansion
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_expanded, 1)
        
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
        mask_f = mask_f / (mask_f.max() + 1e-8)
        return mask_f

    except Exception as e:
        print(f"[Aviso] Fallback build_face_mask: {e}")
        return np.ones((h, w), dtype=np.float32)

# ════════════════════════════════════════════════════════════
# CLIP SURGERY — Heatmaps visuais
# ════════════════════════════════════════════════════════════

_surgery_model = None
_surgery_preprocess = None

def get_surgery_model():
    global _surgery_model, _surgery_preprocess
    if _surgery_model is None:
        print(" A carregar CLIP Surgery CS-ViT-L/14...")
        _surgery_model, _ = clip_surgery.load("CS-ViT-L/14", device=DEVICE)
        _surgery_model.eval()
        _surgery_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (SURGERY_RES, SURGERY_RES),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
        print(" CLIP Surgery pronto.")
    return _surgery_model, _surgery_preprocess

def generate_heatmap(img_rgb, method: str = ""):
    prompts = SURGERY_PROMPTS
    sm, sp = get_surgery_model()
    h, w = img_rgb.shape[:2]
    tensor = sp(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feats = sm.encode_image(tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = clip_surgery.encode_text_with_prompt_ensemble(sm, prompts, DEVICE)
        similarity = clip_surgery.clip_feature_surgery(img_feats, txt_feats)
        sim_map = clip_surgery.get_similarity_map(similarity[:, 1:, :], (h, w))

    face_mask = build_face_mask(img_rgb)
    face_pixels = face_mask > 0.5
    sim_np = sim_map[0].cpu().numpy()

    fake_maps, real_maps, per_text, scores = [], [], {}, {}

    for n, text in enumerate(prompts):
        m = sim_np[:, :, n]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        m = m * face_mask
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        per_text[text] = m.astype(np.float32)
        scores[text] = float(m[face_pixels].mean()) if face_pixels.any() else 0.0

        if any(kw.lower() in text.lower() for kw in FAKE_PROMPT_KEYWORDS):
            fake_maps.append(m)
        elif text in REAL_PROMPTS:
            real_maps.append(m)

    manip_mean = np.mean(fake_maps, axis=0).astype(np.float32) if fake_maps else np.zeros((h, w), dtype=np.float32)
    real_mean = np.mean(real_maps, axis=0).astype(np.float32) if real_maps else np.zeros((h, w), dtype=np.float32)

    contrastive = np.clip(manip_mean - real_mean, 0, None)
    if contrastive.max() > 1e-8:
        contrastive = (contrastive / contrastive.max()).astype(np.float32)
    contrastive = contrastive * face_mask

    manip_scores = {t: s for t, s in scores.items() if any(kw.lower() in t.lower() for kw in FAKE_PROMPT_KEYWORDS)}
    top_prompt_name = max(manip_scores, key=manip_scores.get) if manip_scores else max(scores, key=scores.get)
    top_heatmap = per_text[top_prompt_name]

    return contrastive, per_text, scores, prompts, top_heatmap

def score_regions_manipulation(img_hires, heatmap, masks, scores):
    reg_scores = {}
    h, w = heatmap.shape[:2]
    for name, mask in masks.items():
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pixels = mask_resized > 0
        if not pixels.any():
            reg_scores[name] = {"contrast": 0.0}
            continue

        vals = heatmap[pixels]
        p95    = np.percentile(vals, 95)
        active = float((vals > 0.15).sum()) / (float(pixels.sum()) + 1e-6)
        reg_scores[name] = {"contrast": p95 * active}

    return reg_scores