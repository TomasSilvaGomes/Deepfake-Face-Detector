import streamlit as st
import google.generativeai as genai
from PIL import Image
import os

class ForensicVLMOrchestrator:
    def __init__(self, mode="api"):
        self.mode = mode
        # Configura a API do Google Gemini
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _determine_typology(self, prob_swin, prob_clip):
        if prob_swin > prob_clip:
            return "uma imagem totalmente criada por computador (não existe na vida real)"
        else:
            return "uma fotografia verdadeira onde o rosto foi alterado ou trocado"

    def _crop_artifact_zone(self, img_rgb, bbox):
        if not bbox:
            return img_rgb
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return img_rgb
        return crop

    def build_forensic_prompt(self, prob_final, prob_swin, prob_clip, zones_list):
        tipo_fraude = self._determine_typology(prob_swin, prob_clip)

        prompt = (
            f"Gere um relatório pericial objetivo e acessível a um público leigo.\n"
            f"Analise o recorte da imagem.\n\n"
            f"--- DADOS DO SISTEMA ---\n"
            f"Certeza de fraude: {prob_final * 100:.1f}%\n"
            f"Tipo de fraude: {tipo_fraude}\n"
            f"Zonas com anomalias detetadas: {zones_list}\n"
            f"-----------------------\n\n"
            f"REGRAS OBRIGATÓRIAS:\n"
            f"1. Escreva exatamente 4 frases num único parágrafo, em Português de Portugal.\n"
            f"2. PROIBIDO usar a primeira pessoa (eu, vejo, noto, acho). Use exclusivamente voz impessoal e formal.\n"
            f"3. PROIBIDO usar calão ou jargão técnico (IA, algoritmo, síntese, CLIP, Swin, artefactos, píxeis).\n"
            f"4. FRASE 1: Identifique as zonas afetadas ({zones_list}) e descreva os defeitos visuais presentes. Utilize conceitos reais de análise de imagem.\n"
            f"5. FRASE 2: Indique que estas inconsistências visuais evidenciam tratar-se de {tipo_fraude}.\n"
            f"6. FRASE 3: Descreva características globais da imagem que não são condizentes com uma fotografia genuína.\n"
            f"7. FRASE 4: Conclua EXATAMENTE com: 'O sistema tem {prob_final * 100:.1f}% de certeza de que esta imagem é falsa.'\n"
        )
        return prompt


    def build_real_prompt(self, prob_final):
        """Prompt para imagens classificadas como genuínas."""
        certeza = (1.0 - prob_final) * 100
        return (
            f"Gere um relatório pericial objetivo e acessível a um público leigo.\n"
            f"Analise o recorte da imagem.\n\n"
            f"--- DADOS DO SISTEMA ---\n"
            f"Certeza de autenticidade: {certeza:.1f}%\n"
            f"-----------------------\n\n"
            f"REGRAS OBRIGATÓRIAS:\n"
            f"1. Escreva exatamente 3 frases num único parágrafo, em Português de Portugal.\n"
            f"2. PROIBIDO usar a primeira pessoa. Use exclusivamente voz impessoal e formal.\n"
            f"3. PROIBIDO usar jargão técnico (IA, algoritmo, CLIP, Swin, píxeis, artefactos).\n"
            f"4. FRASE 1: Descreva características visuais que evidenciam autenticidade "
            f"(ex: textura dérmica natural com imperfeições orgânicas, iluminação coerente, "
            f"transições faciais fluidas, ausência de marcas de sobreposição).\n"
            f"5. FRASE 2: Indique que não foram identificadas inconsistências visuais "
            f"características de manipulação digital ou síntese artificial.\n"
            f"6. FRASE 3: Conclua EXATAMENTE com: "
            f"'O sistema tem {certeza:.1f}% de certeza de que esta imagem é genuína.'\n"
        )

    def generate_real_justification(self, img_rgb, prob_final):
        """Gera relatório de autenticidade para imagens classificadas como reais."""
        if not hasattr(self, "model"):
            return "Erro: Google API Key não configurada."

        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((512, 512), Image.Resampling.BICUBIC)
        prompt = self.build_real_prompt(prob_final)

        try:
            response = self.model.generate_content(
                [prompt, pil_img],
                stream=True,
                generation_config={"temperature": 0.1}
            )
            def stream_generator():
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            return stream_generator()
        except Exception as e:
            return f"Erro na chamada à API Gemini: {str(e)}"

    def generate_justification(self, img_rgb, prob_final, prob_swin, prob_clip, zone_name, bbox):
        crop_rgb = self._crop_artifact_zone(img_rgb, bbox)
        prompt = self.build_forensic_prompt(prob_final, prob_swin, prob_clip, zone_name)

        if not hasattr(self, 'model'):
            return "Erro: Google API Key não configurada."

        # Preparar imagem (Thumb para otimizar token/latência)
        pil_img = Image.fromarray(crop_rgb)
        pil_img.thumbnail((512, 512), Image.Resampling.BICUBIC)

        try:
            # Geração com streaming nativo do SDK do Gemini
            response = self.model.generate_content(
                [prompt, pil_img],
                stream=True,
                generation_config={"temperature": 0.1}
            )
            
            def stream_generator():
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            
            return stream_generator()
            
        except Exception as e:
            return f"Erro na chamada à API Gemini: {str(e)}"