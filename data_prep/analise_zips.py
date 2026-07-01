import os
import shutil
import zipfile
import random
import subprocess
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURAÇÃO DE ARQUITETURA
# ==========================================
DIR_ZIPS = "data_prep/DF-40"
OUT_BASE = "estudo_redes"
DIR_REAL = os.path.join(OUT_BASE, "REAL")
DIR_EFS = os.path.join(OUT_BASE, "FAKE_EFS")
DIR_FE = os.path.join(OUT_BASE, "FAKE_FE")

# Hiperparâmetros de Amostragem Balanceada (1:1)
SAMPLES_PER_FAKE_METHOD = 200
SAMPLES_PER_REAL_METHOD = 3400

# Mapeamento Semântico Rigoroso (Tudo em minúsculas)
EFS_METHODS = ['vqgan', 'stylegan2', 'stylegan3', 'styleganxl', 'sd2.1', 'ddim', 'rddm', 'pixart', 'dit', 'sit', 'midjourney', 'whichfaceisrea']
FE_METHODS = ['collabdif', 'e4e', 'stargan', 'starganv2', 'styleclip']
REAL_METHODS = ['real']


random.seed(42)

def setup_directories():
    for d in [DIR_REAL, DIR_EFS, DIR_FE]:
        os.makedirs(d, exist_ok=True)

def classify_method(name):
    name_lower = name.lower()
    
    # 1. Avaliar EFS primeiro (interceta 'wichfaceisreal')
    if any(m in name_lower for m in EFS_METHODS): 
        return "EFS"
        
    # 2. Avaliar FE em seguida
    if any(m in name_lower for m in FE_METHODS): 
        return "FE"
        
    # 3. Avaliar REAL apenas se falhar todas as condições de manipulação
    if any(m in name_lower for m in REAL_METHODS): 
        return "REAL"
        
    return "DESCONHECIDO"

def phase_1_analyze_zips(zip_files):
    print("\n" + "="*60)
    print("FASE 1: ANÁLISE DOS ZIPs")
    print("="*60)
    
    stats = {
        "REAL": defaultdict(int),
        "EFS": defaultdict(int),
        "FE": defaultdict(int),
        "DESCONHECIDO": defaultdict(int)
    }

    for zf_name in zip_files:
        method_name = zf_name.replace('.zip', '')
        category = classify_method(method_name)
        zf_path = os.path.join(DIR_ZIPS, zf_name)
        
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                img_count = sum(1 for f in zf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('/'))
                stats[category][method_name] += img_count
        except zipfile.BadZipFile:
            print(f"[ERRO FATAL] O ficheiro {zf_name} está corrompido.")

    summary_data = []
    for category, methods in stats.items():
        if not methods: continue
        counts = list(methods.values())
        summary_data.append({
            "Categoria": category,
            "Total Imagens": sum(counts),
            "Métodos Distintos": len(counts),
            "Min por Método": min(counts),
            "Max por Método": max(counts)
        })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    if stats["DESCONHECIDO"]:
        print("\n[AVISO CRÍTICO] Métodos ignorados na extração (Desconhecidos):")
        for m, c in stats["DESCONHECIDO"].items():
            print(f"  -> {m}: {c} imagens")
            
    return stats

def phase_2_extract_subset(zip_files, stats):
    print("\n" + "="*60)
    print("FASE 2: EXTRAÇÃO OTIMIZADA")
    print("="*60)
    
    extracted_counts = defaultdict(int)
    
    for zf_name in zip_files:
        method_name = zf_name.replace('.zip', '')
        category = classify_method(method_name)
        
        if category == "DESCONHECIDO": continue
        
        target_limit = SAMPLES_PER_REAL_METHOD if category == "REAL" else SAMPLES_PER_FAKE_METHOD
        if extracted_counts[method_name] >= target_limit: continue
        
        dest_dir = {"REAL": DIR_REAL, "EFS": DIR_EFS, "FE": DIR_FE}[category]
        zf_path = os.path.join(DIR_ZIPS, zf_name)
        
        # Flag para forçar ferramentas externas no caso do e4e
        force_external = (method_name == "e4e")
        
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                valid_files = [f for f in zf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('/')]
                random.shuffle(valid_files)
                
                pbar = tqdm(total=target_limit, desc=f"A extrair {method_name.ljust(15)}", initial=extracted_counts[method_name], leave=False)
                
                for file_path in valid_files:
                    if extracted_counts[method_name] >= target_limit: break

                    unique_name = file_path.replace("/", "_").replace("\\", "_")
                    output_path = os.path.join(dest_dir, f"{method_name}_{unique_name}")
                    
                    success = False
                    
                    # Tenta extração nativa APENAS se não for e4e e se o sistema suportar
                    if not force_external:
                        try:
                            with zf.open(file_path) as source, open(output_path, "wb") as target:
                                target.write(source.read())
                            success = True
                        except NotImplementedError:
                            force_external = True # Feedback: falhou, passa a usar externo
                    
                    # Fallback ou Extração Direta (via 7z ou unzip)
                    if not success:
                        try:
                            # Tenta 7z
                            subprocess.run(["7z", "e", "-y", f"-o{dest_dir}", zf_path, file_path], 
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            
                            # O 7z extrai para a pasta raiz, renomeia para o nome único esperado
                            extracted_filename = os.path.join(dest_dir, os.path.basename(file_path))
                            if os.path.exists(extracted_filename):
                                shutil.move(extracted_filename, output_path)
                                success = True
                        except:
                            pass # Ignora erros de extração individual
                            
                    if success:
                        extracted_counts[method_name] += 1
                        pbar.update(1)
                pbar.close()
        except zipfile.BadZipFile:
            print(f"[ERRO] ZIP corrompido: {zf_name}")

    # Resumo final
    print("\nRESUMO DA EXTRAÇÃO POR MÉTODO:")
    for m, c in extracted_counts.items(): print(f"  -> {m.ljust(20)}: {c} imagens")

    # Relatório Final de Verificação
    print("\nRESUMO DA EXTRAÇÃO POR MÉTODO:")
    for method, count in extracted_counts.items():
        print(f"  -> {method.ljust(20)}: {count} imagens")
    
    print("="*60)

if __name__ == "__main__":
    setup_directories()
    
    if not os.path.exists(DIR_ZIPS):
        print(f"[ERRO] Diretoria {DIR_ZIPS} não encontrada. Verifica o caminho.")
        exit(1)
        
    zips = [f for f in os.listdir(DIR_ZIPS) if f.lower().endswith('.zip')]
    if not zips:
        print(f"[ERRO] Nenhum ficheiro ZIP encontrado em {DIR_ZIPS}.")
        exit(1)
        
    stats_globais = phase_1_analyze_zips(zips)
    phase_2_extract_subset(zips, stats_globais)