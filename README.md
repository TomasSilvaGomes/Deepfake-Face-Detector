# Deepfake Face Detector

> Sistema multimodal de deteção e explicação de manipulações faciais sintéticas, desenvolvido como Projeto de Licenciatura na Universidade da Beira Interior.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/liamu/DeepFakes-Detection)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.0-red)](https://streamlit.io/)

---

## Demonstração

Acede à demonstração pública em: [huggingface.co/spaces/liamu/Deepfake-Face-Detector](https://huggingface.co/spaces/liamu/Deepfake-Face-Detector)

---

## Sobre o Projeto

Este sistema deteta e explica manipulações faciais sintéticas (deepfakes) através de três fases arquiteturais:

**Fase 1 — Classificador de Textura**
Classificador baseado em SwinV2 para deteção de artefactos de textura, com extracção facial via MediaPipe.

**Fase 2 — Explicabilidade Espacial**
Mapas de calor contrastivos via CLIP Surgery e segmentação anatómica facial via BiSeNet, permitindo localizar espacialmente as anomalias detetadas.

**Fase 3 — Fusão Multimodal e Relatório Automático**
Fusão de três sinais independentes (SwinV2, CLIP DF-40, Z-score espacial) através de uma Regressão Logística treinada. Geração automática de relatórios em linguagem natural via API Gemini.




---

## Estrutura do Repositório

```
Deepfake-Face-Detector/
│
├── app.py                          # Launcher (raiz) — redireciona para scripts/app.py
│
├── scripts/
│   ├── app.py                      # Interface Streamlit principal
│   ├── config.py                   # Configuração global (caminhos, constantes, prompts)
│   ├── explainability.py           # CLIP Surgery, heatmaps, face mask, scoring
│   ├── segmenter.py                # BiSeNet — segmentação anatómica facial
│   ├── artifact_zones.py           # Extracção de zonas de artefacto (ArtifactZone)
│   └── interacao_LVM.py            # Orquestrador VLM (Gemini API)
│
├── models/
│   ├── models.py                   # SwinV2Classifier, DF40CLIPModel, transforms
│   ├── bisnet.py                   # Arquitectura BiSeNet
│   └── resnet.py                   # Backbone ResNet-18 do BiSeNet
│
├── CLIP_Surgery/                   # Módulo CLIP Surgery (fork local)
│   └── clip/
│       ├── clip.py
│       ├── clip_surgery_model.py
│       └── ...
│
├── config/
│   └── fusion_weights.json         # Pesos da Regressão Logística treinada
│
├── data_prep/
│   └── analise_zips.py             # Preparação do dataset DF40
│
├── estudo_redes/
│   ├── test_redes_treino_lr.py      # Treino do classificador de fusão
│   ├── comparacao_bench.py          # Comparação com DF40 Protocol-3
│   ├── otimizar_explicabilidade.py  # Otimizar os parâmetros da explicabilidade (Sobreposicao da segmentacao com mapas de calor)
│   └── avaliar_DeepFakeFace.py      # Avaliação externa (DeepFakeFace)
│
├── results/
│   ├── fusion_metrics_dashboard.png
│   ├── comparacao_Protocol_3.png
│   ├── comparison_summary.json
│   ├── params_explicabilidade.json
│   └── evaluation_deepfakeface.json
│
├── requirements.txt                # Dependências para desenvolvimento local
├── packages.txt                    # Dependências de sistema (HuggingFace Spaces)
├── .env.example                    # Template para variáveis de ambiente
├── .streamlit/
│   └── secrets.toml.example        # Template para Streamlit secrets
└── README.md
```

---

## Instalação e Configuração

### Pré-requisitos

- Python 3.11
- CUDA 13.0+ (recomendado) ou CPU
- Git

### 1. Clonar o repositório

```bash
git clone https://github.com/TomasSilvaGomes/Deepfake-Face-Detector
cd Deepfake-Face-Detector
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# ou
.venv\Scripts\activate           # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

Para GPU (ajusta a versão de CUDA conforme a tua instalação):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 4. Configurar a API Gemini

Obtém uma chave gratuita em [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**Opção A — Streamlit Secrets (recomendado para a app):**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edita o ficheiro e preenche com a tua chave
```

**Opção B — Variável de ambiente (para scripts de treino):**
```bash
cp .env.example .env
# Edita o .env e preenche com a tua chave
```

### 5. Pesos dos modelos

Os pesos são descarregados automaticamente do HuggingFace Hub (`liamu/Deepfake-Pesos`) na primeira execução. Não precisas de os descarregar manualmente.

Ficheiros descarregados automaticamente:
- `model.safetensors` — SwinV2 (afinado no OpenFake)
- `clip_large.pth` — CLIP DF-40
- `79999_iter.pth` — BiSeNet
- `face_landmarker.task` — MediaPipe Face Landmarker

---

## Executar a Aplicação

```bash
streamlit run scripts/app.py
```

A interface estará disponível em `http://localhost:8501`.

---

## Treinar o Classificador de Fusão

Se quiseres re-treinar o classificador de Regressão Logística com os teus próprios dados:

### 1. Preparar os dados (dataset DF40)

Coloca os ficheiros `.zip` do DF40 em `data_prep/zips/` e corre:

```bash
python data_prep/analise_zips.py
```

Isto cria a estrutura:
```
data_prep/dataset/
├── REAL/
├── FAKE_EFS/
└── FAKE_FE/
```

### 2. Treinar a Regressão Logística

```bash
python estudo_redes/test_redes_treino_lr.py
```

Os pesos resultantes são guardados em `config/fusion_weights.json` e o dashboard de métricas em `results/`.

### 3. Avaliar em benchmarks

```bash
# Comparação com DF40 Protocol-3
python estudo_redes/comparacao_bench.py

# Avaliação externa no DeepFakeFace
python estudo_redes/avaliar_DeepFakeFace.py
```

---

## Modelos Utilizados

| Componente | Modelo |
|------------|--------|
| Classificador de textura | SwinV2-Small  | 
| Classificador semântico | CLIP ViT-L/14  |
| Explicabilidade espacial | CLIP Surgery CS-ViT-L/14 | 
| Segmentação anatómica | BiSeNet  |
| Detecção facial | MediaPipe Face Landmarker |
| Fusão | Regressão Logística |
| Relatório automático | Gemini 2.5 Flash | 

