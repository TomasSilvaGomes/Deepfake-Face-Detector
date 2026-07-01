"""
avaliar_DeepFakeFace.py
Avaliação externa do sistema (treinado em DF40 EFS+FE) sobre o dataset
DeepFakeFace (DFF) — Song et al., 2023 (arXiv:2309.02218).

Conjunto completamente externo:
  Reais  → IMDB-WIKI (wiki.zip)           — sem overlap com OpenFake ou DF40
  Fakes  → SD v1.5 text-to-image (text2img.zip) + SD Inpainting (inpainting.zip)
           — geradores não incluídos em OpenFake (Flux/DALL-E) nem em DF40

Não requer extracção para disco: lê imagens directamente do zip em memória.
Download feito via hf_hub_download — fica em cache após a primeira execução.
"""

import json
import random
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from models.models import (DF40CLIPModel, SwinV2Classifier,
                           get_clip_transform, get_swinv2_transform)
from scripts.config import DEVICE, FUSION_WEIGHTS, get_model_path
from scripts.explainability import (generate_heatmap, get_region_masks,
                                    score_regions_manipulation)

# ── Configuração ───────────────────────────────────────────────────────
MAX_PER_CLASS  = 1000    
SEED           = 42
FAKE_ZIPS      = ["text2img.zip", "inpainting.zip"]   # SD v1.5 + SD Inpainting
REAL_ZIP       = "wiki.zip"
HF_REPO        = "OpenRL/DeepFakeFace"
RESULTS_DIR    = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════
def load_models():
    print("A carregar modelos...")
    swin_path = get_model_path("model.safetensors")
    swin = SwinV2Classifier(str(swin_path)).to(DEVICE).eval()

    clip_path = get_model_path("clip_large.pth")
    state = torch.load(clip_path, map_location="cpu")
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        if nk.startswith("backbone.") and not nk.startswith("backbone.vision_model."):
            nk = nk.replace("backbone.", "backbone.vision_model.", 1)
        cleaned[nk] = v
    clip = DF40CLIPModel(num_labels=2).to(DEVICE)
    clip.load_state_dict(cleaned, strict=False)
    clip.eval()

    with open(FUSION_WEIGHTS) as f:
        w = json.load(f)

    print(f"  Modelos prontos | DEVICE={DEVICE}")
    return swin, clip, w


# ══════════════════════════════════════════════════════════════════════
# INFERÊNCIA — idêntica ao pipeline de produção em app.py
# ══════════════════════════════════════════════════════════════════════
def infer_single(img_pil, swin, clip, swin_tf, clip_tf, w):
    img_rgb   = np.array(img_pil.convert("RGB"))
    img_hires = cv2.resize(img_rgb, (512, 512))

    # Especialista 1: SwinV2
    t_swin = swin_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p_swin = float(torch.softmax(swin(t_swin), dim=1)[0, 1].item())

    # Especialista 2: CLIP DF-40
    t_clip = clip_tf(img_pil).unsqueeze(0).to(DEVICE)
    t_clip = t_clip.type(next(clip.parameters()).dtype)
    with torch.no_grad():
        p_df40 = float(torch.softmax(clip(t_clip), dim=1)[0, 1].item())

    # Especialista 3: Z-score (CLIP Surgery + BiSeNet)
    try:
        _, per_text, scores, _, _ = generate_heatmap(img_hires)
        fake_map = per_text.get("AI face manipulation", np.zeros((512, 512)))
        real_map = per_text.get("real human face",      np.zeros((512, 512)))
        contrast = np.clip(fake_map - real_map, 0, 1)
        masks    = get_region_masks(img_hires)
        reg      = score_regions_manipulation(img_hires, contrast, masks, scores)
        contrasts = [v["contrast"] for v in reg.values()]
        z = (max(contrasts) - np.mean(contrasts)) / (np.std(contrasts) + 1e-6) \
            if len(contrasts) > 1 else 0.0
    except Exception:
        z = 0.0

    # Fusão LR
    logit = (p_swin * w["weight_swin"] +
             p_df40 * w["weight_df40"] +
             z      * w["weight_z"]    +
             w["bias"])
    prob_final = float(1.0 / (1.0 + np.exp(-logit)))

    # High-confidence override
    if max(p_swin, p_df40) > 0.85:
        prob_final = max(prob_final, max(p_swin, p_df40))

    return prob_final, p_swin, p_df40


# ══════════════════════════════════════════════════════════════════════
# DOWNLOAD + LISTAGEM — fica em cache após primeira execução
# ══════════════════════════════════════════════════════════════════════
def download_zip(filename):
    """Descarrega o zip via hf_hub_download (cache automático)."""
    print(f"  A verificar/descarregar {filename} do HuggingFace...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset"
    )
    print(f"  Pronto: {Path(path).name}  ({Path(path).stat().st_size / 1e9:.2f} GB)")
    return path


def list_images_in_zip(zip_path):
    """Lista os caminhos internos de imagens válidas num zip."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    with zipfile.ZipFile(zip_path, "r") as z:
        entries = [
            e for e in z.namelist()
            if Path(e).suffix.lower() in valid_ext
            and not e.startswith("__MACOSX")
            and not Path(e).name.startswith(".")
        ]
    return entries


def read_image_from_zip(zip_path, internal_path):
    """Lê uma imagem directamente do zip para memória (sem extrair para disco)."""
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(internal_path) as f:
            data = f.read()
    img_array = np.frombuffer(data, np.uint8)
    img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ══════════════════════════════════════════════════════════════════════
# AVALIAÇÃO POR ZIP
# ══════════════════════════════════════════════════════════════════════
def evaluate_zip(zip_path, entries, label, n_max,
                 swin, clip, weights, swin_tf, clip_tf, desc):
    """Itera aleatoriamente pelas imagens do zip e faz inferência."""
    sample = random.sample(entries, min(n_max * 3, len(entries)))
    probs, labels = [], []

    pbar = tqdm(sample, desc=desc, unit="img")
    for internal_path in pbar:
        if len(probs) >= n_max:
            break
        try:
            img_pil = read_image_from_zip(zip_path, internal_path)
            if img_pil is None:
                continue
            # Ignorar imagens com menos de 64 px num dos lados
            if min(img_pil.size) < 64:
                continue
            prob, p_swin, p_df40 = infer_single(
                img_pil, swin, clip, swin_tf, clip_tf, weights
            )
            probs.append(prob)
            labels.append(label)
            pbar.set_postfix(n=len(probs), p_swin=f"{p_swin:.2f}", p_df40=f"{p_df40:.2f}")
        except Exception as e:
            continue

    return np.array(probs), np.array(labels)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    swin, clip_m, weights = load_models()
    swin_tf = get_swinv2_transform()
    clip_tf = get_clip_transform()

    # ── Descarregar zips (cache após 1ª vez) ──────────────────────────
    print(f"\nA preparar dataset DeepFakeFace (OpenRL/DeepFakeFace)...")
    real_zip_path  = download_zip(REAL_ZIP)
    fake_zip_paths = [download_zip(z) for z in FAKE_ZIPS]

    # ── Listar imagens dentro dos zips ────────────────────────────────
    print("\nA indexar imagens nos zips...")
    real_entries = list_images_in_zip(real_zip_path)
    fake_entries = []
    for fzp in fake_zip_paths:
        entries = list_images_in_zip(fzp)
        fake_entries.extend([(fzp, e) for e in entries])
    random.shuffle(real_entries)
    random.shuffle(fake_entries)

    print(f"  Reais disponíveis : {len(real_entries)}")
    print(f"  Fakes disponíveis : {len(fake_entries)}")

    # ── Avaliação — reais ─────────────────────────────────────────────
    print(f"\nA avaliar imagens REAIS (IMDB-WIKI, n={MAX_PER_CLASS})...")
    probs_real, labels_real = evaluate_zip(
        real_zip_path, real_entries, label=0, n_max=MAX_PER_CLASS,
        swin=swin, clip=clip_m, weights=weights,
        swin_tf=swin_tf, clip_tf=clip_tf,
        desc="Reais (IMDB-WIKI)"
    )

    # ── Avaliação — fakes (de vários zips em sequência) ───────────────
    print(f"\nA avaliar imagens FALSAS (SD v1.5 + Inpainting, n={MAX_PER_CLASS})...")
    probs_fake_list, labels_fake_list = [], []
    n_remaining = MAX_PER_CLASS

    for fzp in fake_zip_paths:
        if n_remaining <= 0:
            break
        entries_this = [e for (z, e) in fake_entries if z == fzp]
        n_this = min(n_remaining, MAX_PER_CLASS // len(FAKE_ZIPS) + 1)
        pf, lf = evaluate_zip(
            fzp, entries_this, label=1, n_max=n_this,
            swin=swin, clip=clip_m, weights=weights,
            swin_tf=swin_tf, clip_tf=clip_tf,
            desc=f"Fakes ({Path(fzp).name.replace('.zip','')})"
        )
        probs_fake_list.append(pf)
        labels_fake_list.append(lf)
        n_remaining -= len(pf)

    probs_fake  = np.concatenate(probs_fake_list)
    labels_fake = np.concatenate(labels_fake_list)

    # ── Métricas finais ───────────────────────────────────────────────
    all_probs  = np.concatenate([probs_real, probs_fake])
    all_labels = np.concatenate([labels_real, labels_fake])

    auc_ext = roc_auc_score(all_labels, all_probs)
    acc_ext = accuracy_score(
        all_labels,
        (all_probs >= weights["threshold_optimal"]).astype(int)
    )

    # AUC dos especialistas isolados (para mostrar que a fusão acrescenta valor)
    # (só disponível se tivéssemos guardado p_swin/p_df40 individualmente;
    #  para simplificar, reportamos apenas o sistema completo)

    print(f"\n{'='*60}")
    print("AVALIAÇÃO EXTERNA — DeepFakeFace (DFF)")
    print(f"{'='*60}")
    print(f"Dataset          : OpenRL/DeepFakeFace (Song et al., 2023)")
    print(f"Reais            : IMDB-WIKI  (n={len(probs_real)})")
    print(f"Fakes            : SD v1.5 + SD Inpainting  (n={len(probs_fake)})")
    print(f"Total avaliado   : {len(all_labels)}")
    print(f"─────────────────────────────────────────────────────────")
    print(f"AUC-ROC          : {auc_ext:.4f}")
    print(f"Accuracy         : {acc_ext*100:.2f}%")
    print(f"Threshold usado  : {weights['threshold_optimal']:.4f}")
    print(f"{'='*60}")
    print(f"\nNota metodológica:")
    print(f"  Treino LR        : DF40 EFS+FE (6608 imagens)")
    print(f"  Treino SwinV2    : OpenFake  ← diferente do DFF")
    print(f"  Treino CLIP DF40 : DF40      ← diferente do DFF")
    print(f"  Este é um teste genuinamente cross-dataset:")
    print(f"  nenhum componente do sistema foi treinado em DFF.")
    print(f"\n  Referência literatura:")
    print(f"  CLIP (DF40 Protocol-3, cross-domain): AUC = 0.802")

    result = {
        "dataset":           "DeepFakeFace (OpenRL/DeepFakeFace)",
        "referencia":        "Song et al., arXiv:2309.02218",
        "n_real":            int(len(probs_real)),
        "n_fake":            int(len(probs_fake)),
        "n_total":           int(len(all_labels)),
        "auc":               float(auc_ext),
        "accuracy":          float(acc_ext),
        "threshold":         float(weights["threshold_optimal"]),
        "reais_fonte":       "IMDB-WIKI (wiki.zip)",
        "fakes_metodo":      "SD v1.5 text-to-image + SD Inpainting",
        "nota_cross_dataset":"Nenhum componente treinado em DFF. "
                             "SwinV2→OpenFake, CLIP DF40→DF40, LR→DF40 EFS+FE.",
    }

    out = RESULTS_DIR / "evaluation_deepfakeface.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Resultado guardado: {out}")


if __name__ == "__main__":
    main()