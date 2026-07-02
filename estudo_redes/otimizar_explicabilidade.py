"""
otimizar_explicabilidade.py
─────────────────────────────────────────────────────────────────────────────
Grid search sobre os hiperparâmetros de artifact_zones.py.

"""

import sys
import argparse
import itertools
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from config import DEVICE
from explainability import (
    generate_heatmap,
    get_region_masks,
    score_regions_manipulation,
)


THIN_REGIONS = {"olho_esq", "olho_dir", "sobrancelha_esq", "sobrancelha_dir", "boca"}

def extract_zones_parametric(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    region_masks: dict,
    region_scores: dict,
    params: dict,
) -> np.ndarray:
    """
    Versão do extract_artifact_zones que aceita parâmetros externos.
    Devolve a máscara binária total (união de todas as zonas válidas).
    """
    h, w = img_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    if not region_scores:
        return np.zeros((h, w), dtype=bool)

    max_score = max(d["contrast"] for d in region_scores.values())
    if max_score < 0.05:
        return np.zeros((h, w), dtype=bool)

    cand_thresh = max(max_score * params["cand_pct"], params["cand_floor"])
    candidates = [
        (name, data["contrast"])
        for name, data in region_scores.items()
        if data["contrast"] >= cand_thresh
        and name not in ["pele", "cabelo", "pescoco"]
    ]

    union_mask = np.zeros((h, w), dtype=bool)

    for name, score in candidates:
        mask_f = region_masks.get(name)
        if mask_f is None:
            continue

        mask_f_r = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_NEAREST)

        if name in THIN_REGIONS:
            k_size     = params["kernel_thin"]
            iters      = params["iters_thin"]
            percentile = params["pct_thin"]
        else:
            k_size     = params["kernel_wide"]
            iters      = params["iters_wide"]
            percentile = params["pct_wide"]

        kernel     = np.ones((k_size, k_size), np.uint8)
        region_bin = cv2.dilate(
            (mask_f_r > 0.3).astype(np.uint8), kernel, iterations=iters
        ).astype(bool)

        if not region_bin.any():
            continue

        vals          = heatmap[region_bin]
        local_thresh  = max(np.percentile(vals, percentile), params["min_floor"])
        zone_mask     = (heatmap >= local_thresh) & region_bin

        if not zone_mask.any():
            continue

        # Filtro de densidade
        density   = zone_mask.sum() / (region_bin.sum() + 1e-6)
        min_dens  = params["min_dens_large"] if region_bin.sum() > 3000 \
                    else params["min_dens_small"]
        if density < min_dens:
            continue

        union_mask |= zone_mask

    return union_mask


def compute_metric(heatmap: np.ndarray, zone_mask: np.ndarray, alpha: float = 0.5) -> float:
    """
    score = α × coverage + (1-α) × precision

    coverage  = activação média dentro das zonas / activação média total
                (queremos que as zonas cubram as zonas quentes)
    precision = activação média dentro das zonas
                (queremos que as zonas não estejam em zonas frias)
    """
    if not zone_mask.any():
        return 0.0

    global_mean = float(heatmap.mean()) + 1e-8
    zone_mean   = float(heatmap[zone_mask].mean())

    coverage  = min(zone_mean / global_mean, 5.0) / 5.0   # normalizado [0,1]
    precision = zone_mean                                   # já em [0,1]

    return alpha * coverage + (1 - alpha) * precision


# ─────────────────────────────────────────────
# Grid de hiperparâmetros
# ─────────────────────────────────────────────

GRID = {
    "cand_pct":       [0.20, 0.30, 0.40],
    "cand_floor":     [0.10, 0.15],
    "kernel_thin":    [5, 7, 9],
    "iters_thin":     [1, 2, 3],
    "kernel_wide":    [3, 5],
    "iters_wide":     [1, 2],
    "pct_thin":       [50, 60, 70],
    "pct_wide":       [65, 75, 85],
    "min_floor":      [0.05, 0.08, 0.12],
    "min_dens_large": [0.04, 0.06, 0.10],
    "min_dens_small": [0.01, 0.02, 0.04],
}


def grid_combinations(grid: dict) -> List[dict]:
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pasta", required=True,
                        help="Pasta com imagens FAKE para optimização")
    parser.add_argument("--n",     type=int, default=50,
                        help="Número de imagens a amostrar (default: 50)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Peso entre coverage e precision (default: 0.5)")
    parser.add_argument("--out",   default="results/params_explicabilidade.json",
                        help="Ficheiro de saída com os melhores parâmetros")
    args = parser.parse_args()

    pasta = Path(args.pasta)
    imgs  = list(pasta.glob("*.jpg")) + list(pasta.glob("*.png"))
    if not imgs:
        print(f"[Erro] Nenhuma imagem encontrada em {pasta}")
        return

    # Amostragem aleatória
    rng  = np.random.default_rng(42)
    imgs = list(rng.choice(imgs, size=min(args.n, len(imgs)), replace=False))
    print(f"A usar {len(imgs)} imagens de {pasta.name}")

    # ── Pré-computar heatmaps e máscaras (independentes dos params) ──
    print("\n[1/2] A computar heatmaps e máscaras BiSeNet...")
    cache = []
    for img_path in tqdm(imgs):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_512  = cv2.resize(img_rgb, (512, 512))

        try:
            contrastive, per_text, scores, _, _ = generate_heatmap(img_512)
            masks      = get_region_masks(img_512)
            contrast_m = np.clip(
                per_text.get("AI face manipulation", np.zeros((512, 512))) -
                per_text.get("real human face",      np.zeros((512, 512))),
                0, 1
            )
            reg_scores = score_regions_manipulation(img_512, contrast_m, masks, scores)
            cache.append({
                "img_rgb":     img_512,
                "heatmap":     contrastive,
                "masks":       masks,
                "reg_scores":  reg_scores,
            })
        except Exception as e:
            print(f"  [Aviso] Erro em {img_path.name}: {e}")

    print(f"  {len(cache)} imagens processadas com sucesso.")

    # ── Grid search ──
    combos = grid_combinations(GRID)
    print(f"\n[2/2] Grid search: {len(combos)} combinações × {len(cache)} imagens...")

    best_score  = -1.0
    best_params = None
    scores_list = []

    for params in tqdm(combos):
        scores_per_img = []
        for entry in cache:
            zone_mask = extract_zones_parametric(
                entry["img_rgb"],
                entry["heatmap"],
                entry["masks"],
                entry["reg_scores"],
                params,
            )
            s = compute_metric(entry["heatmap"], zone_mask, alpha=args.alpha)
            scores_per_img.append(s)

        mean_score = float(np.mean(scores_per_img))
        scores_list.append((mean_score, params))

        if mean_score > best_score:
            best_score  = mean_score
            best_params = params

    # ── Top 5 ──
    scores_list.sort(key=lambda x: x[0], reverse=True)
    print(f"\n Melhor score médio: {best_score:.4f}")
    print("Melhores parâmetros:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nTop 5 combinações:")
    for rank, (s, p) in enumerate(scores_list[:5], 1):
        print(f"  #{rank}  score={s:.4f}  params={p}")

    # ── Guardar resultado ──
    out_path = ROOT_DIR / args.out
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "best_score":  best_score,
            "best_params": best_params,
            "top5": [{"score": s, "params": p} for s, p in scores_list[:5]],
            "n_images":    len(cache),
            "n_combos":    len(combos),
            "alpha":       args.alpha,
        }, f, indent=2)

    print(f"\nResultados guardados ")


if __name__ == "__main__":
    main()