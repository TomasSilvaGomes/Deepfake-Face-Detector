import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 1. Injeção da raiz no sys.path para garantir que os imports funcionam
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.bisnet import BiSeNet


class FaceSegmenter:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.n_classes = 19
        self.model = BiSeNet(n_classes=self.n_classes)

        # Carregamento robusto dos pesos
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def get_masks(self, img_rgb):
        h, w = img_rgb.shape[:2]
        img_tensor = (
            self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self.model(img_tensor)[0]
            # O output do BiSeNet é [1, 19, 512, 512]
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
            # Redimensionar a segmentação para o tamanho original da imagem
            parsing = cv2.resize(
                parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )

        masks = {
            "sobrancelha_esq": (parsing == 2).astype(np.float32),
            "sobrancelha_dir": (parsing == 3).astype(np.float32),
            "olho_esq": (parsing == 4).astype(np.float32),
            "olho_dir": (parsing == 5).astype(np.float32),
            "orelha_esq": (parsing == 7).astype(np.float32),
            "orelha_dir": (parsing == 8).astype(np.float32),
            "nariz": (parsing == 10).astype(np.float32),
            "boca": ((parsing == 11) | (parsing == 12) | (parsing == 13)).astype(
                np.float32
            ),
            "pescoco": (parsing == 14).astype(np.float32),
            "cabelo": (parsing == 17).astype(np.float32),
            "pele": (parsing == 1).astype(np.float32),
        }

        # ── Lógica de exclusão e zonas dinâmicas ──
        free_skin = (parsing == 1).astype(np.uint8)

        # Testa: acima das sobrancelhas
        brows = ((parsing == 2) | (parsing == 3)).astype(np.uint8)
        if brows.any():
            brow_top_y = int(np.where(brows)[0].min())
            testa = np.zeros((h, w), dtype=np.uint8)
            testa[:brow_top_y, :] = free_skin[:brow_top_y, :]
            masks["testa"] = testa.astype(np.float32)
        else:
            masks["testa"] = np.zeros((h, w), dtype=np.float32)

        # Bochechas: exclusão mútua usando coordenadas
        eye_y = max(
            np.where(parsing == 4)[0].max() if (parsing == 4).any() else 0,
            np.where(parsing == 5)[0].max() if (parsing == 5).any() else 0,
        )
        mouth_top = int(
            np.where((parsing >= 11) & (parsing <= 13))[0].min()
            if ((parsing >= 11) & (parsing <= 13)).any()
            else h
        )
        nose_min_x = int(
            np.where(parsing == 10)[1].min() if (parsing == 10).any() else w // 2
        )
        nose_max_x = int(
            np.where(parsing == 10)[1].max() if (parsing == 10).any() else w // 2
        )

        y_coords, x_coords = np.indices((h, w))

        # Bochecha Esquerda
        mask_cheek_l = (
            (free_skin == 1)
            & (y_coords > eye_y)
            & (y_coords < mouth_top)
            & (x_coords < nose_min_x)
        )
        # Erosão para não invadir o nariz/olhos
        kernel = np.ones((3, 3), np.uint8)
        mask_cheek_l = cv2.erode(mask_cheek_l.astype(np.uint8), kernel, iterations=1)
        masks["bochecha_esq"] = mask_cheek_l.astype(np.float32)

        # Bochecha Direita
        mask_cheek_r = (
            (free_skin == 1)
            & (y_coords > eye_y)
            & (y_coords < mouth_top)
            & (x_coords > nose_max_x)
        )
        mask_cheek_r = cv2.erode(mask_cheek_r.astype(np.uint8), kernel, iterations=1)
        masks["bochecha_dir"] = mask_cheek_r.astype(np.float32)

        return masks
