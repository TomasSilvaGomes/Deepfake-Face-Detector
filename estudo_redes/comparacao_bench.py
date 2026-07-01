"""
comparacao_bench.py
Avaliação do sistema de com comparação directa à Table 5
(Protocol-3) do paper DF40 — linha EFS (FF).
Gera gráficos com fundo branco prontos para relatório académico.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

CSV_PATH    = Path("estudo_redes/Teste_Redes.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# MAPEAMENTO: prefixo do ficheiro → nome canónico da Table 5
# ══════════════════════════════════════════════════════════════════════
GENERATOR_MAP = {
    # Coincidências directas com Table 5
    "midjourney":     "MidJourney-6",
    "whichfaceisrea": "Whichisreal",
    "stargan":        "StarGAN",
    "starganv2":      "StarGAN2",
    "styleclip":      "StyleCLIP",
    "e4e":            "e4e",
    "collabdif":      "CollabDiff",
    # EFS adicionais (não na Table 5 — reportados em separado)
    "vqgan":          "VQGAN",
    "stylegan2":      "StyleGAN2",
    "stylegan3":      "StyleGAN3",
    "styleganxl":     "StyleGAN-XL",
    "sd2.1":          "SD-v2.1",
    "ddim":           "DDIM",
    "rddm":           "RDDM",
    "pixart":         "PixArt-α",
    "dit":            "DiT",
    "sit":            "SiT",
}

# Geradores que coincidem com Table 5 (os 7 disponíveis)
TABLE5_GENS = [
    "MidJourney-6", "Whichisreal",
    "StarGAN", "StarGAN2", "StyleCLIP", "e4e", "CollabDiff",
]

# Valores da Table 5, Protocol-3, linha EFS (FF) — paper DF40
LITERATURA_PER_GEN = {
    "MidJourney-6": {"Xception":0.472, "CLIP":0.534, "SRM":0.338, "SPSL":0.427, "RECCE":0.442, "RFM":0.551},
    "Whichisreal":  {"Xception":0.772, "CLIP":0.828, "SRM":0.794, "SPSL":0.694, "RECCE":0.753, "RFM":0.623},
    "StarGAN":      {"Xception":0.777, "CLIP":0.946, "SRM":0.769, "SPSL":0.699, "RECCE":0.769, "RFM":0.730},
    "StarGAN2":     {"Xception":0.677, "CLIP":0.823, "SRM":0.703, "SPSL":0.723, "RECCE":0.724, "RFM":0.636},
    "StyleCLIP":    {"Xception":0.984, "CLIP":0.929, "SRM":0.982, "SPSL":0.922, "RECCE":0.964, "RFM":0.966},
    "e4e":          {"Xception":0.611, "CLIP":0.923, "SRM":0.509, "SPSL":0.602, "RECCE":0.643, "RFM":0.665},
    "CollabDiff":   {"Xception":0.997, "CLIP":0.983, "SRM":0.997, "SPSL":0.967, "RECCE":0.979, "RFM":0.979},
}

# ══════════════════════════════════════════════════════════════════════
# CARREGAR CSV
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv(CSV_PATH)
X  = df[["prob_swin", "prob_df40", "z_score"]].values
y  = df["label"].values
print(f"Dataset: {(y==0).sum()} reais | {(y==1).sum()} fakes | total {len(y)}")

# ══════════════════════════════════════════════════════════════════════
# 1. SPLIT 80/20
# ══════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
clf = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000)
clf.fit(X_train, y_train)

y_probs_test = clf.predict_proba(X_test)[:, 1]
auc_test     = roc_auc_score(y_test, y_probs_test)
acc_test     = accuracy_score(y_test, (y_probs_test >= 0.5).astype(int))

print(f"\n── Split 80/20 ──")
print(f"AUC (teste)     : {auc_test:.4f}")
print(f"Accuracy (teste): {acc_test*100:.2f}%")
print(f"Train/Test      : {len(X_train)} / {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════
# 2. 5-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs_cv, accs_cv = [], []
for train_idx, val_idx in skf.split(X, y):
    clf_cv = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000)
    clf_cv.fit(X[train_idx], y[train_idx])
    probs = clf_cv.predict_proba(X[val_idx])[:, 1]
    aucs_cv.append(roc_auc_score(y[val_idx], probs))
    accs_cv.append(accuracy_score(y[val_idx], (probs >= 0.5).astype(int)))

auc_cv_mean = np.mean(aucs_cv)
auc_cv_std  = np.std(aucs_cv)
print(f"\n── 5-Fold Cross-Validation ──")
for i, (a, acc) in enumerate(zip(aucs_cv, accs_cv)):
    print(f"  Fold {i+1}: AUC={a:.4f}  Acc={acc*100:.2f}%")
print(f"  Média : AUC={auc_cv_mean:.4f} ± {auc_cv_std:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 3. AUC POR GERADOR (extraído do prefixo do nome do ficheiro)
# ══════════════════════════════════════════════════════════════════════
clf_full = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000)
clf_full.fit(X, y)
df["prob_pred"] = clf_full.predict_proba(X)[:, 1]

df["generator_raw"] = df["img_name"].apply(
    lambda n: str(n).split("_")[0].lower() if pd.notna(n) else "unknown"
)
df["generator"] = df["generator_raw"].map(GENERATOR_MAP).fillna(df["generator_raw"])

reais_probs = df.loc[df["label"] == 0, "prob_pred"].values
our_per_gen = {}

print(f"\n── AUC por Gerador ──")
for raw, gen_name in GENERATOR_MAP.items():
    mask = (df["label"] == 1) & (df["generator_raw"] == raw)
    if mask.sum() < 5:
        continue
    fake_probs = df.loc[mask, "prob_pred"].values
    y_bin      = np.array([0]*len(reais_probs) + [1]*len(fake_probs))
    p_bin      = np.concatenate([reais_probs, fake_probs])
    auc_gen    = roc_auc_score(y_bin, p_bin)
    our_per_gen[gen_name] = auc_gen
    marker = "← Table 5" if gen_name in TABLE5_GENS else ""
    print(f"  {gen_name:<18} AUC: {auc_gen:.4f}  (n={mask.sum()})  {marker}")

# Médias separadas
our_t5   = [our_per_gen[g] for g in TABLE5_GENS if g in our_per_gen]
our_add  = [v for k, v in our_per_gen.items() if k not in TABLE5_GENS]
avg_t5   = np.mean(our_t5)   if our_t5  else float("nan")
avg_add  = np.mean(our_add)  if our_add else float("nan")

print(f"\n  Avg. (7 gen. Table 5) : {avg_t5:.4f}")
print(f"  Avg. (adic. não T5)   : {avg_add:.4f}")

# ══════════════════════════════════════════════════════════════════════
# 4. TABELA IMPRESSA — comparação por gerador com Table 5
# ══════════════════════════════════════════════════════════════════════
# ── Tabela: linhas = modelos, colunas = geradores + Avg ──────────────
modelos_ordem = ["Xception", "CLIP", "SRM", "SPSL", "RECCE", "RFM", "Ours"]

# Cabeçalho — nomes curtos para caber
gen_labels = ["MidJou.", "Whichis.", "StarGAN", "StarGAN2", "StyleCLIP", "e4e", "CollabD.", "Avg."]

col_w = 10
header = f"{'Model':<12}" + "".join(f"{g:>{col_w}}" for g in gen_labels)
print(f"\n{'='*len(header)}")
print("PROTOCOL-3  DF40 Table 5 — EFS Test")
print(f"{'='*len(header)}")
print(header)
print(f"{'-'*len(header)}")

for modelo in modelos_ordem:
    if modelo == "Ours":
        vals = [our_per_gen.get(g, float("nan")) for g in TABLE5_GENS]
    else:
        vals = [LITERATURA_PER_GEN.get(g, {}).get(modelo, float("nan")) for g in TABLE5_GENS]

    avg = np.nanmean(vals)
    row = f"{'Ours':<12}" if modelo == "Ours" else f"{modelo:<12}"
    row += "".join(f"{v:>{col_w}.3f}" for v in vals)
    row += f"{avg:>{col_w}.3f}"
    print(row)

print(f"{'='*len(header)}")

# ══════════════════════════════════════════════════════════════════════
# 5. GRÁFICOS — FUNDO BRANCO, PRONTOS PARA RELATÓRIO
# ══════════════════════════════════════════════════════════════════════
AZUL   = "#1d4ed8"   # nosso sistema
CINZA  = "#94a3b8"   # literatura
VERDE  = "#16a34a"   # CV
LIGH   = "#e2e8f0"   # fundo de barras

fig = plt.figure(figsize=(18, 12), facecolor="white")
fig.suptitle(
    "Comparação Indicativa — DFFD vs DF40 EFS+FE "
    "(backbones SwinV2/CLIP congelados) vs. DF40 (Protocol-3)",
    fontsize=9, fontweight="bold", color="#0f172a", y=0.99
)

gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                      left=0.08, right=0.97, top=0.93, bottom=0.07)

ax_bar  = fig.add_subplot(gs[0, 0])   # barras horizontais avg
ax_roc  = fig.add_subplot(gs[0, 1])   # curva ROC
ax_gen  = fig.add_subplot(gs[1, :])   # por gerador (grouped bars)

# ─────────────────────────────────────────────────────────────────────
# Gráfico 1 — Barras horizontais: Avg. Table 5 vs Nosso sistema
# ─────────────────────────────────────────────────────────────────────
avg_lit_per_modelo = {
    m: np.mean([LITERATURA_PER_GEN[g][m] for g in TABLE5_GENS])
    for m in modelos_ordem[:-1]  # exclui "Ours"
}
avg_lit_per_modelo["Ours \n(80/20 test)"]  = auc_test
avg_lit_per_modelo["Ours \n(5-fold CV)"]   = auc_cv_mean

labels_bar = list(avg_lit_per_modelo.keys())
vals_bar   = list(avg_lit_per_modelo.values())
cores_bar  = [CINZA] * len(modelos_ordem[:-1]) + [AZUL, VERDE]

bars = ax_bar.barh(labels_bar, vals_bar, color=cores_bar,
                   edgecolor="white", linewidth=0.6, height=0.6)
ax_bar.axvline(0.5, color="#cbd5e1", linestyle="--", lw=1.2)
ax_bar.set_xlim(0.35, 1.02)
ax_bar.set_xlabel("AUC-ROC (média — 7 geradores)", fontsize=9, color="#334155")
ax_bar.set_title("Avg. AUC por Modelo\n(Table 5, Protocol-3, EFS Test)",
                 fontsize=10, fontweight="bold", color="#0f172a", pad=8)
ax_bar.set_facecolor("white")
ax_bar.tick_params(axis="y", labelsize=8.5)
ax_bar.tick_params(axis="x", labelsize=8)
ax_bar.xaxis.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
ax_bar.set_axisbelow(True)

for bar, val, cor in zip(bars, vals_bar, cores_bar):
    bold = cor in [AZUL, VERDE]
    ax_bar.text(val + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5,
                fontweight="bold" if bold else "normal",
                color=AZUL if cor == AZUL else (VERDE if cor == VERDE else "#334155"))

# ─────────────────────────────────────────────────────────────────────
# Gráfico 2 — Curva ROC (conjunto de teste 20%)
# ─────────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_probs_test)
ax_roc.plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1.2, zorder=1)
ax_roc.plot(fpr, tpr, color=AZUL, lw=2.5, zorder=3,
            label=f"Nosso Sistema  (AUC = {auc_test:.3f})")

# Referência textual ao melhor baseline do paper (não como linha no gráfico)
ax_roc.text(0.38, 0.12,
            f"Melhor baseline DF40 (CLIP): AUC = 0.802",
            fontsize=8, color="#475569",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9",
                      edgecolor="#cbd5e1", linewidth=0.8))

ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.05)
ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=9, color="#334155")
ax_roc.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=9, color="#334155")
ax_roc.set_title("Curva ROC — Conjunto de Teste (20%)",
                 fontsize=10, fontweight="bold", color="#0f172a", pad=8)
ax_roc.legend(fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#e2e8f0")
ax_roc.set_facecolor("white")
ax_roc.tick_params(labelsize=8)
ax_roc.grid(color="#e2e8f0", linewidth=0.8, zorder=0)
ax_roc.set_axisbelow(True)

# ─────────────────────────────────────────────────────────────────────
# Gráfico 3 — AUC por gerador: grouped bars (todos os modelos + nosso)
# ─────────────────────────────────────────────────────────────────────
modelos_plot = ["Xception", "CLIP", "SRM", "RECCE", "RFM", "Ours"]
palette = {
    "Xception": "#94a3b8",
    "CLIP":     "#64748b",
    "SRM":      "#a8a29e",
    "RECCE":    "#78716c",
    "RFM":      "#b0b8c8",
    "Ours": AZUL,
}

n_gen    = len(TABLE5_GENS)
n_mod    = len(modelos_plot)
x        = np.arange(n_gen)
width    = 0.13
offsets  = np.linspace(-(n_mod-1)/2, (n_mod-1)/2, n_mod) * width

for i, modelo in enumerate(modelos_plot):
    vals = []
    for gen in TABLE5_GENS:
        if modelo == "Ours":
            vals.append(our_per_gen.get(gen, 0.0))
        else:
            vals.append(LITERATURA_PER_GEN.get(gen, {}).get(modelo, 0.0))

    cor     = palette[modelo]
    ec      = AZUL if modelo == "Ours" else "white"
    lw      = 1.5  if modelo == "Ours" else 0.5
    zorder  = 4    if modelo == "Ours" else 2
    ax_gen.bar(x + offsets[i], vals, width,
               label=modelo, color=cor,
               edgecolor=ec, linewidth=lw, zorder=zorder)

ax_gen.set_xticks(x)
ax_gen.set_xticklabels(TABLE5_GENS, fontsize=9, rotation=15, ha="right")
ax_gen.set_ylabel("AUC-ROC", fontsize=9, color="#334155")
ax_gen.set_ylim(0.25, 1.08)
ax_gen.set_title(
    "AUC por Gerador — Comparação com Baselines (DF40 Table 5, Protocol-3, EFS Test)\n"
    "Os 7 geradores disponíveis que coincidem com os da tabela original",
    fontsize=10, fontweight="bold", color="#0f172a", pad=8
)
ax_gen.set_facecolor("white")
ax_gen.tick_params(axis="x", labelsize=8.5)
ax_gen.tick_params(axis="y", labelsize=8)
ax_gen.yaxis.grid(True, color="#e2e8f0", linewidth=0.8, zorder=0)
ax_gen.set_axisbelow(True)

# Linha da média do nosso sistema (só nos T5)
ax_gen.axhline(avg_t5, color=AZUL, linestyle="--", lw=1.5, alpha=0.7,
               label=f"Média Ours = {avg_t5:.3f}")
ax_gen.axhline(
    np.mean([np.mean(list(LITERATURA_PER_GEN[g].values())) for g in TABLE5_GENS]),
    color="#64748b", linestyle=":", lw=1.2, alpha=0.6,
    label="Média CLIP (best baseline)"
)

handles, lbls = ax_gen.get_legend_handles_labels()
ax_gen.legend(handles, lbls, fontsize=8, ncol=4,
              loc="upper left", framealpha=0.92, edgecolor="#e2e8f0")

# ─────────────────────────────────────────────────────────────────────
# Guardar
# ─────────────────────────────────────────────────────────────────────
out = RESULTS_DIR / "comparacao_Protocol_3.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n[+] Gráfico guardado: {out}")

# ══════════════════════════════════════════════════════════════════════
# 6. JSON SUMÁRIO
# ══════════════════════════════════════════════════════════════════════
summary = {
    "protocolo":            "DF40 Protocol-3 — Test only, 7 geradores Table 5",
    "auc_split_test":       float(auc_test),
    "acc_split_test":       float(acc_test),
    "auc_cv_mean":          float(auc_cv_mean),
    "auc_cv_std":           float(auc_cv_std),
    "auc_avg_table5_gens":  float(avg_t5),
    "n_total":              int(len(y)),
    "n_real":               int((y==0).sum()),
    "n_fake":               int((y==1).sum()),
    "per_generator": {
        "table5":       {k: float(v) for k, v in our_per_gen.items() if k in TABLE5_GENS},
        "additional":   {k: float(v) for k, v in our_per_gen.items() if k not in TABLE5_GENS},
    },
    "literatura_avg_table5": {
        m: float(np.mean([LITERATURA_PER_GEN[g][m] for g in TABLE5_GENS]))
        for m in modelos_ordem[:-1]  # exclui "Ours"
    },
}
with open(RESULTS_DIR / "comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)
print("[+] Sumário JSON guardado.")