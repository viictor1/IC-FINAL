"""
Extrai as deep features do ViT-large para o conjunto de validacao (Valid).

Faz a mesma extracao do ViT_large do extraicarac.py (media do last_hidden_state,
vetor de 1024-D), mas so com transformers, sem precisar de open_clip/timm.

O Train ja foi extraido em result_final_ViT_large.csv; aqui geramos o mesmo pro
Valid, porque o enunciado pede pra reportar teste e validacao.

Saida: result_final_ViT_large_valid.csv (95 amostras x 1024 features).
"""

import os
import re
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


def organizar_imagens_em_pastas(pasta_base):
    """Move imagens soltas (bart081.bmp ...) para subpastas por classe."""
    print("[-] A organizar imagens em subpastas...")
    movidos = 0
    for arquivo in os.listdir(pasta_base):
        if arquivo.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
            caminho = os.path.join(pasta_base, arquivo)
            if os.path.isfile(caminho):
                m = re.match(r"([a-zA-Z]+)", arquivo)
                if m:
                    destino = os.path.join(pasta_base, m.group(1).lower())
                    os.makedirs(destino, exist_ok=True)
                    shutil.move(caminho, os.path.join(destino, arquivo))
                    movidos += 1
    print(f"[+] {movidos} imagens organizadas." if movidos else
          "[!] Nenhuma imagem solta — assumindo já organizadas.")


def feature_extraction(image_path, model, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def features_to_df(folder_path, model, feature_extractor):
    data = []
    for subfolder in sorted(os.listdir(folder_path)):
        sub = os.path.join(folder_path, subfolder)
        if os.path.isdir(sub):
            for img in tqdm(sorted(os.listdir(sub)), desc=f"Processando {subfolder}"):
                if img.lower().endswith(("png", "jpg", "jpeg", "bmp")):
                    p = os.path.join(sub, img)
                    try:
                        data.append([p, *feature_extraction(p, model, feature_extractor)])
                    except Exception as e:
                        print(f"Erro em {p}: {e}")
    n = len(data[0]) - 1 if data else 0
    cols = ["image_path"] + [f"feature_{i}" for i in range(n)]
    return pd.DataFrame(data, columns=cols)


if __name__ == "__main__":
    valid_folder = os.path.abspath(os.path.join(HERE, "..", "Valid"))
    print(f"[-] Conjunto de validação: {valid_folder}")

    organizar_imagens_em_pastas(valid_folder)

    print("Carregando ViT-large (google/vit-large-patch16-224-in21k)...")
    model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k").eval().to(device)
    feat_ext = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")

    df = features_to_df(valid_folder, model, feat_ext)
    out = os.path.join(HERE, "result_final_ViT_large_valid.csv")
    df.to_csv(out, index=False)
    print(f"[✓] {len(df)} amostras de validação salvas em {out}")
