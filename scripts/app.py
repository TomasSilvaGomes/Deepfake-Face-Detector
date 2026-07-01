"""
app.py — Interface Streamlit do sistema forense de deteção de deepfakes.

Responsabilidades deste ficheiro:
  - Carregar modelos (SwinV2, CLIP DF-40) via cache
  - Gerir a UI (upload, sliders, botões, visualizações)
  - Orquestrar o pipeline de inferência chamando funções dos módulos especializados

NÃO duplica lógica de:
  explainability.py  → generate_heatmap, get_region_masks, score_regions_manipulation,
                       get_landmarker (singleton FaceLandmarker), build_face_mask
  artifact_zones.py  → extract_artifact_zones, segment_zones_with_probability
  interacao_LVM.py   → ForensicVLMOrchestrator
  models/models.py   → SwinV2Classifier, DF40CLIPModel, transforms
  config.py          → FUSION_WEIGHTS, ROOT_DIR, DEVICE
"""

import base64
import concurrent.futures
import json
import warnings
import gc
from io import BytesIO

import cv2
import mediapipe as mp          # apenas para mp.Image / mp.ImageFormat
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import transformers
from huggingface_hub import hf_hub_download
from PIL import Image

# ── Módulos do projecto ────────────────────────────────────────────────
from artifact_zones import extract_artifact_zones, segment_zones_with_probability
from config import DEVICE, FUSION_WEIGHTS, ROOT_DIR
from explainability import (
    generate_heatmap,
    get_landmarker,     # singleton reutilizado aqui e no explainability.py
    get_region_masks,
    score_regions_manipulation,
)
from models.models import DF40CLIPModel, SwinV2Classifier, get_clip_transform, get_swinv2_transform
from scripts.interacao_LVM import ForensicVLMOrchestrator

warnings.filterwarnings("ignore", category=UserWarning, message=".*sm_120.*")
transformers.logging.set_verbosity_error()

PADDING_FACE = 0.45

st.set_page_config(
    page_title="Segurança Visual", layout="wide", initial_sidebar_state="expanded"
)


# ══════════════════════════════════════════════════════════════════════
# CACHE DE RECURSOS — carregados uma única vez por sessão
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_all_models():
    """Carrega SwinV2 e CLIP DF-40 a partir do HuggingFace Hub (cache local)."""
    # SwinV2
    swin_path = hf_hub_download(repo_id="liamu/Deepfake-Pesos", filename="model.safetensors")
    swin = SwinV2Classifier(ckpt_path=swin_path).to("cpu").eval()

    # CLIP DF-40 — higienização do state_dict para remover prefixos de DataParallel
    clip_path = hf_hub_download(repo_id="liamu/Deepfake-Pesos", filename="clip_large.pth")
    state = torch.load(clip_path, map_location="cpu")
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        if nk.startswith("backbone.") and not nk.startswith("backbone.vision_model."):
            nk = nk.replace("backbone.", "backbone.vision_model.", 1)
        cleaned[nk] = v
    clip = DF40CLIPModel(num_labels=2).to("cpu")
    clip.load_state_dict(cleaned)
    clip.eval()

    return swin, clip


@st.cache_resource
def load_fusion_weights():
    """Lê o JSON de pesos da Regressão Logística uma única vez."""
    try:
        with open(FUSION_WEIGHTS) as f:
            cfg = json.load(f)
        return (
            float(cfg.get("weight_swin", 1.0)),
            float(cfg.get("weight_df40", 1.0)),
            float(cfg.get("weight_z",    1.0)),
            float(cfg.get("bias",        0.0)),
            float(cfg.get("threshold_optimal", 0.6877)),
        )
    except Exception:
        return 1.0, 1.0, 1.0, 0.0, 0.6877


@st.cache_resource
def get_transforms():
    """Cria os transforms de pré-processamento uma única vez."""
    return get_swinv2_transform(), get_clip_transform()


# ══════════════════════════════════════════════════════════════════════
# DETEÇÃO FACIAL — reutiliza o singleton get_landmarker() do explainability.py
# ══════════════════════════════════════════════════════════════════════

def extract_main_face(img_bgr, padding_ratio=PADDING_FACE):
    """
    Recorta a face dominante com padding adaptativo.
    Usa o FaceLandmarker (MediaPipe Tasks) já instanciado em explainability.py.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = get_landmarker().detect(mp_image)
    except Exception as e:
        return None, f"Erro na deteção facial: {e}"

    if not result.face_landmarks:
        return None, "Nenhuma face detetada. Submeta um retrato mais claro."

    lm = result.face_landmarks[0]
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_h = int((y_max - y_min) * padding_ratio)
    pad_w = int((x_max - x_min) * padding_ratio)

    y1 = max(0, y_min - pad_h);  y2 = min(h, y_max + pad_h)
    x1 = max(0, x_min - pad_w);  x2 = min(w, x_max + pad_w)

    return img_bgr[y1:y2, x1:x2], "OK"


# ══════════════════════════════════════════════════════════════════════
# FUNÇÕES DE RENDERIZAÇÃO UI (específicas do Streamlit — não duplicar noutros módulos)
# ══════════════════════════════════════════════════════════════════════

def inject_custom_css():
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #2563eb; color: white; border-radius: 6px;
            font-weight: bold; border: none; padding: 0.5rem 1rem; transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #1d4ed8; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        div[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 8px; background-color: #0f172a; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        </style>
    """, unsafe_allow_html=True)


def render_confidence_bar(prob_fake, threshold):
    is_fake   = prob_fake > threshold
    confianca = prob_fake if is_fake else (1.0 - prob_fake)
    color     = "#ef4444" if is_fake else "#22c55e"
    label     = "FALSA"   if is_fake else "REAL"
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
      <div style="display:flex;justify-content:space-between;margin-bottom:.25rem;">
        <span style="font-weight:bold;font-size:1.1rem;color:{color};">🎯 {label}</span>
        <span style="font-weight:bold;">{confianca*100:.1f}%</span>
      </div>
      <div style="width:100%;background-color:#334155;border-radius:4px;height:12px;overflow:hidden;">
        <div style="width:{confianca*100}%;background-color:{color};height:100%;transition:width 0.5s ease;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def visualize_heatmap(heatmap_array, colormap=cv2.COLORMAP_JET):
    hm = cv2.applyColorMap((heatmap_array * 255).astype(np.uint8), colormap)
    return Image.fromarray(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB))


def render_heat_card(img_pil, title, subtitle, color):
    buff = BytesIO()
    img_pil.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;width:100%;">
      <img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;box-shadow:0 4px 6px rgba(0,0,0,.3);">
      <p style="text-align:center;color:{color};font-size:14px;margin-top:12px;line-height:1.4;">
        {title}<br><b>{subtitle}</b>
      </p>
    </div>"""


def render_interactive_polygons(img_pil, zones, prob_masks):
    buff = BytesIO()
    img_pil.save(buff, format="JPEG")
    img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    width, height = img_pil.size

    def san(name):
        return name.lower().replace(" ", "-").replace("/", "-")

    svg_polygons = ""
    menu_items   = ""
    valid_zones  = []

    for zone in zones:
        z_name = zone.name.lower()
        mask   = prob_masks.get(z_name)
        if mask is None or not mask.any():
            continue
        valid_zones.append(z_name)
        zc = san(z_name)

        # ── Redimensionar máscara de 512×512 para o espaço da imagem exibida ──
        mask_u8 = cv2.resize(
            mask.astype(np.uint8) * 255,
            (width, height),          # dimensões reais da imagem no SVG
            interpolation=cv2.INTER_NEAREST
        )
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            approx = cv2.approxPolyDP(cnt, 0.002 * cv2.arcLength(cnt, True), True)
            pts    = " ".join(f"{pt[0][0]},{pt[0][1]}" for pt in approx)
            svg_polygons += f'<polygon class="poly-{zc}" points="{pts}" style="fill:rgba(239,68,68,.15);stroke:rgba(255,255,255,.2);stroke-width:1;transition:all .3s ease;pointer-events:none;"></polygon>'

    for name in sorted(set(valid_zones)):
        zc = san(name)
        hi  = f"document.querySelectorAll('.poly-{zc}').forEach(p=>{{p.style.fill='rgba(239,68,68,.7)';p.style.stroke='rgba(255,255,255,1)';p.style.strokeWidth='3';}});this.style.backgroundColor='#3b82f6';this.style.color='white';"
        ho  = f"document.querySelectorAll('.poly-{zc}').forEach(p=>{{p.style.fill='rgba(239,68,68,.15)';p.style.stroke='rgba(255,255,255,.2)';p.style.strokeWidth='1';}});this.style.backgroundColor='#1e293b';this.style.color='#cbd5e1';"
        menu_items += f'<div onmouseover="{hi}" onmouseout="{ho}" style="padding:10px 15px;background-color:#1e293b;color:#cbd5e1;border-radius:6px;cursor:pointer;font-size:13px;font-weight:bold;transition:all .2s ease;border:1px solid #334155;text-transform:uppercase;">{name}</div>'

    components.html(f"""<!DOCTYPE html><html><head><style>body{{margin:0;padding:0;background:transparent;font-family:sans-serif;}}</style></head><body>
    <div style="display:flex;gap:20px;width:100%;align-items:start;">
      <div style="flex:0 0 180px;display:flex;flex-direction:column;gap:8px;">
        <p style="margin:0 0 5px 0;color:#94a3b8;font-size:12px;font-weight:bold;text-transform:uppercase;">Anatomia Afetada</p>
        {menu_items}
      </div>
      <div style="flex:1;position:relative;display:flex;justify-content:center;">
        <svg viewBox="0 0 {width} {height}" style="width:100%;max-width:512px;height:auto;max-height:550px;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.3);display:block;" xmlns="http://www.w3.org/2000/svg">
          <image href="data:image/jpeg;base64,{img_b64}" width="{width}" height="{height}"/>
          {svg_polygons}
        </svg>
      </div>
    </div></body></html>""", height=500)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    inject_custom_css()

    # Carregar todos os recursos em cache
    swin, clip_df40          = get_all_models()
    w_swin, w_df40, w_z, bias, threshold_default = load_fusion_weights()
    swin_tf, clip_tf         = get_transforms()

    st.markdown(
        '<p style="font-size: 60px; font-weight: bold; color: #bdbbbb; text-align: center; margin-bottom: 10px;"> 🔍 Deepfake Face Detector </p>',
        unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:16px;margin-bottom:2rem;'>Deteção de criação sintética e manipulação em rostos humanos.</p>", unsafe_allow_html=True)

    with st.expander("Como funciona a plataforma", expanded=False):
        st.markdown("""
        O sistema analisa a imagem em três fases para determinar se foi gerada ou manipulada por Inteligência Artificial:

        1. **Análise de Superfície:** Procura artefactos microscópicos e falhas de textura invisíveis ao olho humano.
        2. **Coerência Biométrica:** Verifica se os traços faciais e a iluminação são consistentes, detetando trocas de rosto, edições.
        3. **Localização de Anomalias:** Isola e mapeia graficamente as áreas específicas onde a manipulação ocorreu.
    

        
        <div style='margin-top:20px;'>
          <p style='font-size:12px;font-family:monospace;background-color:#0f172a;padding:10px;border-radius:4px;color:#cbd5e1;'>
          <strong>INPUT:</strong> Imagem RGB da cara (A cores) <br>
          <strong>OUTPUT:</strong> Classificação → Explicação Visual (Explicador e Segmentador) → Relatório do Gemini
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚙️ Configuração da Análise")
    opcoes_input  = ["Sua Imagem", "Exemplo Falso", "Exemplo Real"]
    escolha_input = st.sidebar.selectbox("Fonte da imagem:", opcoes_input)
    threshold     = st.sidebar.slider("Rigor da Deteção", 0.10, 0.95, float(threshold_default), 0.01)
    
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='background-color:#0f172a;padding:15px;border-radius:8px;border-left:4px solid #3b82f6;font-size:14px;line-height:1.4;'>
      <p style='margin-top:0; margin-bottom:12px; font-weight:bold; color:#e2e8f0; font-size:15px;'>
        Em que consiste a APP
      </p>
      
      <p style='margin-bottom:4px; color:#94a3b8; font-size:12px; text-transform:uppercase; font-weight:bold;'>
        1. Deteção
      </p>
      <p style='margin-bottom:12px; font-weight:bold; color:#f8fafc; font-size:13px;'>
        SwinV2 & CLIP DF-40<br>
        <span style='font-weight:normal; color:#cbd5e1; font-size:12px;'>
          Analisam texturas microscópicas e traços faciais para calcular a probabilidade de a imagem ser falsa.
        </span>
      </p>
      
      <p style='margin-bottom:4px; color:#94a3b8; font-size:12px; text-transform:uppercase; font-weight:bold;'>
        2. Mapeamento
      </p>
      <p style='margin-bottom:12px; font-weight:bold; color:#f8fafc; font-size:13px;'>
        CLIP Surgery & BiSeNet<br>
        <span style='font-weight:normal; color:#cbd5e1; font-size:12px;'>
          Funcionam como um raio-X, isolando e destacando as zonas exatas do rosto que sofreram manipulação.
        </span>
      </p>

      <p style='margin-bottom:4px; color:#94a3b8; font-size:12px; text-transform:uppercase; font-weight:bold;'>
        3. Relatório
      </p>
      <p style='margin-bottom:0; font-weight:bold; color:#f8fafc; font-size:13px;'>
        Gemini (Google)<br>
        <span style='font-weight:normal; color:#cbd5e1; font-size:12px;'>
          Lê as anomalias detetadas nos passos anteriores e gera uma explicação consoante seja falsa ou real.
        </span>
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1.2], gap="large")
    img_bgr     = None
    raw_img_bgr = None

    # ── Coluna de Input ───────────────────────────────────────────────
    with col_input:
        st.markdown("#### Origem da Imagem")

        if escolha_input == "Sua Imagem":
            up = st.file_uploader("Arraste o ficheiro", type=["jpg", "png", "jpeg"],
                                  label_visibility="collapsed")
            if up:
                raw_img_bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            nome_base = "false" if "Falso" in escolha_input else "real"
            for ext in [".png", ".jpg", ".jpeg"]:
                p = ROOT_DIR / "exemplos" / f"{nome_base}{ext}"
                if p.exists():
                    raw_img_bgr = cv2.imread(str(p))
                    break

        analisar = False
        if raw_img_bgr is not None:
            with st.spinner("A detetar rosto na imagem..."):
                cropped, status = extract_main_face(raw_img_bgr)

            if cropped is None:
                st.error(status)
            else:
                img_bgr = cropped
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         caption="Área de análise isolada", width=350)
                label_btn = "Verificar Autenticidade" if escolha_input == "Sua Imagem" \
                            else f"Analisar Exemplo ({nome_base.upper()})"
                analisar = st.button(label_btn, use_container_width=True)

        

    # ── Coluna de Resultados ─────────────────────────────────────────
    with col_result:
        st.markdown("#### 📝 Resultados da Análise")

        if analisar and img_bgr is not None:
            # Limpar estado de análise anterior
            for key in ["contrastive_hm", "per_text_hm", "prompt_list", "reg_scores"]:
                st.session_state.pop(key, None)

            img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hires  = cv2.resize(img_rgb, (512, 512))
            img_pil    = Image.fromarray(img_rgb)
            raw_rgb    = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
            raw_pil    = Image.fromarray(raw_rgb)

            # Preparar tensores
            t_swin = swin_tf(raw_pil).unsqueeze(0).to("cpu")
            t_clip = (clip_tf(img_pil).unsqueeze(0).to("cpu")
                      .type(next(clip_df40.parameters()).dtype))

            # Inferência paralela dos três especialistas
            with st.spinner("A verificar autenticidade..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                    f_swin    = ex.submit(lambda t: float(torch.softmax(swin(t), dim=1)[0, 1].item()), t_swin)
                    f_clip    = ex.submit(lambda t: float(torch.softmax(clip_df40(t), dim=1)[0, 1].item()), t_clip)
                    f_surgery = ex.submit(generate_heatmap, img_hires)

                    prob_swin      = f_swin.result()
                    prob_clip_df40 = f_clip.result()
                    contrastive_hm, per_text_hm, scores, prompts, _ = f_surgery.result()

            # Z-score espacial
            masks        = get_region_masks(img_hires) # # 512×512 — mesma resolução do heatmap
            contrast_map = np.clip(
                per_text_hm.get("AI face manipulation", np.zeros((512, 512))) -
                per_text_hm.get("real human face",      np.zeros((512, 512))),
                0, 1
            )
            reg_scores = score_regions_manipulation(img_hires, contrast_map, masks, scores)
            contrasts  = [d["contrast"] for d in reg_scores.values()]
            z_anomaly  = ((max(contrasts) - np.mean(contrasts)) / (np.std(contrasts) + 1e-6)
                          if len(contrasts) > 1 else 0.0)

            # Fusão LR + High-Confidence Override
            logit      = prob_swin * w_swin + prob_clip_df40 * w_df40 + z_anomaly * w_z + bias
            prob_final = float(1.0 / (1.0 + np.exp(-logit)))
            if max(prob_swin, prob_clip_df40) > 0.85:
                prob_final = max(prob_final, max(prob_swin, prob_clip_df40))

            is_fake = prob_final > threshold

            # Guardar no session_state para o painel técnico
            st.session_state.update({
                "contrastive_hm": contrastive_hm,
                "per_text_hm":    per_text_hm,
                "prompt_list":    prompts,
                "reg_scores":     reg_scores,
            })
            
            render_confidence_bar(prob_final, threshold)

            if is_fake:
                zones = extract_artifact_zones(img_hires, contrastive_hm, masks, reg_scores)
                if zones:
                    prob_masks = segment_zones_with_probability(contrastive_hm, zones, prob_threshold=0.40)
                    render_interactive_polygons(Image.fromarray(img_rgb), zones, prob_masks)

            else:
                st.image(img_rgb, width=250, caption="Nenhuma anomalia detetada.")


    # ── Relatório  ────────────────────────
    if analisar and img_bgr is not None and "is_fake" in dir() and not is_fake:
        st.markdown("<hr style='border:1px solid #334155;margin:2rem 0;'>",
                    unsafe_allow_html=True)
        st.markdown("### 📋 Relatório")

        with st.spinner("A gerar relatório ..."):
            orchestrator = ForensicVLMOrchestrator(mode="api")
            stream = orchestrator.generate_real_justification(img_rgb, prob_final)

        ESTILO_REAL = ("background-color:#0f172a;padding:25px;border-radius:8px;"
                       "border-left:4px solid #22c55e;font-size:15px;color:#f8fafc;"
                       "line-height:1.7;box-shadow:0 4px 6px rgba(0,0,0,.2);margin-bottom:2rem;")
        box_real = st.empty()
        if isinstance(stream, str):
            box_real.markdown(f"<div style='{ESTILO_REAL}'>{stream}</div>", unsafe_allow_html=True)
        else:
            acumulado = ""
            for chunk in stream:
                if chunk:
                    acumulado += chunk
                    box_real.markdown(f"<div style='{ESTILO_REAL}'>{acumulado} ▌</div>",
                                 unsafe_allow_html=True)
            box_real.markdown(f"<div style='{ESTILO_REAL}'>{acumulado}</div>", unsafe_allow_html=True)

    # ── Relatório ) ──────────
    if analisar and img_bgr is not None and "is_fake" in dir() and is_fake and "zones" in dir() and zones:
        st.markdown("<hr style='border:1px solid #334155;margin:2rem 0;'>", unsafe_allow_html=True)
        st.markdown("### 📋 Relatório ")

        with st.spinner("A gerar relatório ..."):
            orchestrator = ForensicVLMOrchestrator(mode="api")
            global_bbox  = (
                min(z.bbox[0] for z in zones), min(z.bbox[1] for z in zones),
                max(z.bbox[2] for z in zones), max(z.bbox[3] for z in zones),
            )
            stream = orchestrator.generate_justification(
                img_rgb=img_hires,
                prob_final=prob_final,
                prob_swin=prob_swin,
                prob_clip=prob_clip_df40,
                zone_name=", ".join(z.name for z in zones),
                bbox=global_bbox,
            )

        ESTILO = ("background-color:#0f172a;padding:25px;border-radius:8px;"
                  "border-left:4px solid #FF0000;font-size:15px;color:#f8fafc;"
                  "line-height:1.7;box-shadow:0 4px 6px rgba(0,0,0,.2);margin-bottom:2rem;")
        box = st.empty()
        if isinstance(stream, str):
            box.markdown(f"<div style='{ESTILO}'>{stream}</div>", unsafe_allow_html=True)
        else:
            acumulado = ""
            for chunk in stream:
                if chunk:
                    acumulado += chunk
                    box.markdown(f"<div style='{ESTILO}'>{acumulado} ▌</div>", unsafe_allow_html=True)
            box.markdown(f"<div style='{ESTILO}'>{acumulado}</div>", unsafe_allow_html=True)

    # ── Painel Técnico: Matemática Contrastiva ────────────────────────
    if analisar and is_fake and st.session_state.get("contrastive_hm") is not None:
        with st.expander("Visão Detalhada: Como a Anomalia é Isolada", expanded=False):
            st.markdown("### Processo de Subtração Visual")
            st.markdown("O sistema analisa a imagem através de duas 'lentes' diferentes: uma programada para detetar sinais de manipulação gerada por IA e outra para reconhecer padrões orgânicos de um rosto humano natural. Ao subtrair a componente natural, o ruído visual desaparece, destacando apenas as áreas manipuladas.")
            st.markdown("<br>", unsafe_allow_html=True)

            top_prompt  = "AI face manipulation"
            real_prompt = "real human face"
            c1, cm, c2, ce, c3 = st.columns([1.5, .3, 1.5, .3, 1.5], vertical_alignment="center")

            with c1:
                st.markdown(render_heat_card(
                    visualize_heatmap(st.session_state["per_text_hm"][top_prompt]),
                    "Padrão Sintético", "Lente de Manipulação", "#ef4444"), unsafe_allow_html=True)
            with cm:
                st.markdown("<h1 style='text-align:center;color:#cbd5e1;'>-</h1>", unsafe_allow_html=True)
            with c2:
                if real_prompt in st.session_state["per_text_hm"]:
                    st.markdown(render_heat_card(
                        visualize_heatmap(st.session_state["per_text_hm"][real_prompt]),
                        "Padrão Orgânico", "Lente Natural", "#22c55e"), unsafe_allow_html=True)
            with ce:
                st.markdown("<h1 style='text-align:center;color:#cbd5e1;'>=</h1>", unsafe_allow_html=True)
            with c3:
                st.markdown(render_heat_card(
                    visualize_heatmap(st.session_state["contrastive_hm"]),
                    "Resultado Final", "Anomalia Destacada", "#3b82f6"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()