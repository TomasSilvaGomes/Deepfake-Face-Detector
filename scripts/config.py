"""
config.py — Configuracao Global do Pipeline SOTA
Centralizado e ordenado com sistema de Lazy Loading para Cloud.
"""

import os
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

# 1. Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if "mps" in str(torch.backends.mps.is_available()):
    DEVICE = "mps"

# 2. Diretorios base
ROOT_DIR = Path(__file__).resolve().parent.parent

# 3. Estrutura de Pastas
MODELS_DIR = ROOT_DIR / "models"
CONFIG_DIR = ROOT_DIR / "config"
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Garante que a pasta existe no servidor
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 4. Gestor de Download em Producao (Cloud-Native SOTA)
# Substitui pelo nome exato do Model que criaste no Passo 1
HF_REPO_ID = "liamu/Deepfake-Pesos"

def get_model_path(filename: str) -> Path:
    """
    Verifica se o peso existe localmente. Se nao, faz download seguro do HF Hub.
    O hf_hub_download usa cache, logo so descarrega a primeira vez.
    """
    local_path = MODELS_DIR / filename
    if not local_path.exists():
        print(f"A inicializar SOTA: A descarregar {filename} da infraestrutura cloud...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename=filename, 
                local_dir=MODELS_DIR
            )
            return Path(downloaded_path)
        except Exception as e:
            raise FileNotFoundError(f"Falha ao transferir {filename}. Verifica o nome do repositorio. Erro: {e}")
    return local_path


FUSION_WEIGHTS = CONFIG_DIR / "fusion_weights.json"

# 6. Constantes (CLIP Surgery)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SURGERY_RES = 512

MANIPULATION_PROMPTS = ["AI face manipulation"]
REAL_PROMPTS = ["real human face"]
SURGERY_PROMPTS = MANIPULATION_PROMPTS + REAL_PROMPTS
FAKE_PROMPT_KEYWORDS = {"AI", "manipulation"}