import os
import sys
import json
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from tqdm import tqdm

# ==========================================
# CONFIGURAÇÃO DE CAMINHOS E IMPORTAÇÕES
# ==========================================
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from scripts.config import DEVICE, FUSION_WEIGHTS, get_model_path
from models.models import SwinV2Classifier, DF40CLIPModel, get_swinv2_transform, get_clip_transform
from scripts.explainability import generate_heatmap, get_region_masks, score_regions_manipulation


def load_clip_df40():
    """Carrega o modelo DF-40 usando o gestor centralizado de assets."""
    try:
        clip_model = DF40CLIPModel(num_labels=2).to(DEVICE)
        clip_weights_path = get_model_path("clip_large.pth")

        state = torch.load(clip_weights_path, map_location="cpu")

        cleaned = {}
        for k, v in state.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            if nk.startswith("backbone.") and not nk.startswith("backbone.vision_model."):
                nk = nk.replace("backbone.", "backbone.vision_model.", 1)
            cleaned[nk] = v

        clip_model.load_state_dict(cleaned, strict=False)
        return clip_model.eval()
    except Exception as e:
        print(f"[Erro] Falha na inicialização do CLIP DF-40: {e}")
        return None


def set_deterministic_state(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_web_compression(img_bgr, min_quality=40, max_quality=70):
    quality = random.randint(min_quality, max_quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img_bgr, encode_param)
    return cv2.imdecode(encimg, 1)


def extract_features_to_csv(data_dir, csv_path, swin_model, clip_model, swin_tf, clip_tf):
    features = []
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    for folder_name in folders:
        folder_path = os.path.join(data_dir, folder_name)
        label = 1 if "fake" in folder_name.lower() else 0
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        print(f"\n--- Processando pasta: {folder_name} ({len(images)} imagens) [label={label}] ---")

        for img_name in tqdm(images, desc="Progresso", unit="img"):
            img_path = os.path.join(folder_path, img_name)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            if random.random() < 0.5:
                img_bgr = apply_web_compression(img_bgr)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_hires = cv2.resize(img_rgb, (512, 512))

            # ── Especialista 1: SwinV2 ────────────────────────────────────
            tensor_swin = swin_tf(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob_swin = float(torch.softmax(swin_model(tensor_swin), dim=1)[0, 1].item())

            # ── Especialista 2: CLIP DF-40 ────────────────────────────────
            tensor_clip = clip_tf(img_pil).unsqueeze(0).to(DEVICE)
            tensor_clip = tensor_clip.type(next(clip_model.parameters()).dtype)
            with torch.no_grad():
                prob_clip_df40 = float(torch.softmax(clip_model(tensor_clip), dim=1)[0, 1].item())

            # ── Especialista 3: Z-Score espacial (CLIP Surgery + BiSeNet) ─
            # IMPORTANTE: o contrast_map é calculado exactamente como em app.py
            # para garantir que os pesos do JSON são aplicados correctamente
            # em inferência (os valores de treino == os valores de produção).
            try:
                _, per_text_hm, scores, _, _ = generate_heatmap(img_hires)

                fake_map = per_text_hm.get("AI face manipulation", np.zeros((512, 512)))
                real_map = per_text_hm.get("real human face", np.zeros((512, 512)))
                contrast_map = np.clip(fake_map - real_map, 0, 1)

                masks = get_region_masks(img_hires)
                reg_scores = score_regions_manipulation(img_hires, contrast_map, masks, scores)

                contrasts = [data["contrast"] for data in reg_scores.values()]
                if len(contrasts) > 1:
                    z_anomaly = (max(contrasts) - np.mean(contrasts)) / (np.std(contrasts) + 1e-6)
                else:
                    z_anomaly = 0.0
            except Exception as e:
                print(f"  [AVISO] CLIP Surgery falhou em {img_name}: {e}")
                z_anomaly = 0.0

            features.append({
                "img_name": img_name,
                "folder": folder_name,
                "prob_swin": prob_swin,
                "prob_df40": prob_clip_df40,
                "z_score": z_anomaly,
                "label": label
            })

    df = pd.DataFrame(features)
    df.to_csv(csv_path, index=False)
    print(f"\n[+] CSV guardado: {csv_path}  ({len(df)} linhas)")
    return df


def generate_forensic_plots(y_true, y_probs, best_t, roc_auc, output_dir):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 5))

    ax1 = plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    ax1.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="#64748b", lw=2, linestyle="--")
    ax1.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel="FPR", ylabel="TPR", title="Curva ROC")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.2)

    ax2 = plt.subplot(1, 3, 2)
    sweep_t = np.linspace(0.05, 0.95, 200)
    accuracies = [accuracy_score(y_true, y_probs >= t) for t in sweep_t]
    ax2.plot(sweep_t, accuracies, color="#22c55e", lw=2)
    ax2.axvline(best_t, color="#ef4444", linestyle="--", label=f"Limiar Ótimo: {best_t:.3f}")
    ax2.set(xlabel="Limiar de Decisão", ylabel="Exatidão", title="Efeito do Limiar na Exatidão")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.2)

    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(y_probs[y_true == 0], bins=25, alpha=0.6, color="#22c55e", label="Reais", density=True)
    ax3.hist(y_probs[y_true == 1], bins=25, alpha=0.6, color="#ef4444", label="Falsas", density=True)
    ax3.axvline(best_t, color="white", linestyle="--", lw=2)
    ax3.set(xlabel="Probabilidade P(Fake)", ylabel="Densidade", title="Separação Latente (Real vs Fake)")
    ax3.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "fusion_metrics_dashboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def save_metrics_report(y_true, y_pred, best_acc, roc_auc, best_t,
                        w_swin, w_df40, w_z, bias, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "metricas_calibracao.txt")

    total_reais = int(np.sum(y_true == 0))
    total_fakes = int(np.sum(y_true == 1))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    report = f"""==================================================
RELATÓRIO DE CALIBRAÇÃO - CLASSIFICADOR DE FUSÃO 3D
==================================================

1. RESUMO DO CONJUNTO DE DADOS
--------------------------------------------------
Total de Amostras  : {len(y_true)}
Classe Real (0)    : {total_reais}
Classe Fake (1)    : {total_fakes}

2. DESEMPENHO GLOBAL DO MODELO
--------------------------------------------------
AUC (Área Sob ROC) : {roc_auc:.4f}
Accuracy Global    : {best_acc * 100:.2f}%
Limiar de Decisão  : {best_t:.4f}

3. ANÁLISE DE ERRO (Limiar Ótimo: {best_t:.4f})
--------------------------------------------------
Falsos Positivos   : {fp} (Reais classificados como Fakes)
Falsos Negativos   : {fn} (Fakes classificados como Reais)

4. PARÂMETROS DA REGRESSÃO LOGÍSTICA
--------------------------------------------------
Nota: pesos directos sobre features em escala natural
(prob_swin ∈ [0,1], prob_df40 ∈ [0,1], z_score ∈ ℝ)
Sem StandardScaler — compatível com app.py.

w_swin (SwinV2)    : {w_swin:.4f}
w_df40 (CLIP-DF40) : {w_df40:.4f}
w_z    (Z-Score)   : {w_z:.4f}
Bias / Interceção  : {bias:.4f}

Equação de Transferência Algébrica:
logit = (P_swin × {w_swin:.4f}) + (P_clip × {w_df40:.4f}) + (Z × {w_z:.4f}) + ({bias:.4f})
P(Fake) = sigmoid(logit) = 1 / (1 + e^-logit)
==================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report_path


def main():
    set_deterministic_state()

    data_dir = ROOT_DIR / "estudo_redes"
    results_dir = ROOT_DIR / "results"
    csv_path = Path(__file__).resolve().parent / "Teste_Redes.csv"

    if not data_dir.exists():
        print(f"[Erro] Diretoria de dados não encontrada: {data_dir}")
        return

    # ── Fase 1: Extração de features (com cache em CSV) ──────────────
    if not csv_path.exists():
        print(f"[!] A extrair features para {csv_path.name}...")

        swin_weights_path = get_model_path("model.safetensors")
        swin_model = SwinV2Classifier(str(swin_weights_path)).to(DEVICE).eval()
        swin_transform = get_swinv2_transform()

        clip_model = load_clip_df40()
        if clip_model is None:
            return
        clip_transform = get_clip_transform()

        df = extract_features_to_csv(
            str(data_dir), str(csv_path),
            swin_model, clip_model,
            swin_transform, clip_transform
        )

        del swin_model, clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[+] CSV em cache encontrado: {csv_path.name}")
        df = pd.read_csv(csv_path)

    print(f"\n[+] Dataset: {(df['label']==0).sum()} reais | {(df['label']==1).sum()} fakes")

    # ── Fase 2: Treino da Regressão Logística ────────────────────────
    # SEM StandardScaler — pesos directamente compatíveis com app.py
    X = df[["prob_swin", "prob_df40", "z_score"]].values
    y = df["label"].values

    clf = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000
    )
    clf.fit(X, y)

    # Extrair pesos directamente (sem pipeline)
    w_swin, w_df40, w_z = clf.coef_[0]
    bias = float(clf.intercept_[0])

    # ── Fase 3: Avaliação ────────────────────────────────────────────
    y_probs = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)

    # Limiar óptimo via sweep de accuracy (consistente com app.py)
    sweep_t = np.linspace(0.05, 0.95, 200)
    accs = [accuracy_score(y, y_probs >= t) for t in sweep_t]
    best_t = float(sweep_t[np.argmax(accs)])
    best_acc = float(max(accs))
    y_pred_optimal = (y_probs >= best_t).astype(int)

    # ── Fase 4: Exportar artefactos ──────────────────────────────────
    plot_file = generate_forensic_plots(y, y_probs, best_t, roc_auc, str(results_dir))
    report_file = save_metrics_report(
        y, y_pred_optimal, best_acc, roc_auc, best_t,
        w_swin, w_df40, w_z, bias, str(results_dir)
    )

    weights = {
        "weight_swin":       float(w_swin),
        "weight_df40":       float(w_df40),
        "weight_z":          float(w_z),
        "bias":              bias,
        "threshold_optimal": best_t,
        "accuracy":          best_acc,
        "auc":               float(roc_auc)
    }

    FUSION_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with open(FUSION_WEIGHTS, "w") as f:
        json.dump(weights, f, indent=4)

    print("\n" + "=" * 50)
    print("CALIBRAÇÃO DO ENSEMBLE CONCLUÍDA".center(50))
    print("=" * 50)
    print(f"Accuracy Global : {best_acc * 100:.2f}%")
    print(f"AUC             : {roc_auc:.4f}")
    print(f"Limiar de Corte : {best_t:.4f}")
    print(f"w_swin          : {w_swin:.4f}")
    print(f"w_df40          : {w_df40:.4f}")
    print(f"w_z             : {w_z:.4f}")
    print(f"bias            : {bias:.4f}")
    print(f"\n[!] Artefactos Exportados:")
    print(f"  -> JSON de Pesos : {FUSION_WEIGHTS.name}")
    print(f"  -> Gráfico ROC   : {Path(plot_file).name}")
    print(f"  -> Relatório Txt : {Path(report_file).name}")
    print("=" * 50)


if __name__ == "__main__":
    main()