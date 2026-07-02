import os
import shutil
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import ViTModel, ViTImageProcessor, AutoModel, CLIPImageProcessor
import timm
import open_clip
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models(model_name):
    # Modelos baseados em ViT (transformers e auto model)
    if model_name == 'ViT_huge':
        model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-huge-patch14-224-in21k')
    elif model_name == 'ViT_large':
        model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
    elif model_name == 'ViT_base':
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    elif model_name == 'ViT_small':
        model = AutoModel.from_pretrained('WinKawaks/vit-small-patch16-224')
        feature_extractor = ViTImageProcessor.from_pretrained('WinKawaks/vit-small-patch16-224')
    elif model_name == 'VITAmin':
        model = timm.create_model('vitamin_large_384', pretrained=True, num_classes=0)
        model.eval()
        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    # Modelo openclip
    elif model_name == 'openclip_vitg14':
        model, _, feature_extractor = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    # Modelo mambaout 
    elif model_name == 'mambaout':
        model = timm.create_model('mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k', pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    # Novas arquiteturas do timm/huggingface
    elif model_name in ['resnet18', 'squeezenet1_0', 'resnet50', 'resnet101', 'efficientnet_b0', 
                        'inception_resnet_v2', 'nasnetalarge', 'inception_v3', 'xception', 'darknet53', 
                        'vit_so400m_patch14_siglip_378.webli_ft_in1k', 
                        'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k', 'mobilenetv4_hybrid_large.ix_e600_r384_in1k', 
                        'convnextv2_huge.fcmae_ft_in22k_in1k_384']:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(model)
        feature_extractor = timm.data.create_transform(**data_config, is_training=False)
    else:
        raise ValueError(f"Modelo {model_name} não suportado.")
    
    model.eval()
    return model.to(device), feature_extractor

def feature_extraction(image_path, model, feature_extractor, model_name):
    image = Image.open(image_path).convert('RGB')
    
    if model_name in ['ViT_huge', 'ViT_large', 'ViT_base', 'ViT_small']:
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    elif 'openclip' in model_name:
        image_tensor = feature_extractor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encode_image(image_tensor)
        features = feats.squeeze().cpu().numpy()
    
    elif hasattr(model, 'forward_features'):
        input_tensor = feature_extractor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.forward_features(input_tensor)
            feats = model.forward_head(feats, pre_logits=True)
        features = feats.squeeze().cpu().numpy()
    
    else:
        input_tensor = feature_extractor(image)
        if not torch.is_tensor(input_tensor):
            raise ValueError("A transformação do timm não retornou um tensor.")
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            feats = model(input_tensor)
        features = feats.squeeze().cpu().numpy()
    
    return features

def features_to_df(folder_path, model, feature_extractor, model_name):
    data = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for image_file in tqdm(os.listdir(subfolder_path), desc=f"Processando {subfolder}"):
                image_path = os.path.join(subfolder_path, image_file)
                if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')): # ADICIONEI 'bmp' AQUI PARA OS SIMPSONS
                    try:
                        feats = feature_extraction(image_path, model, feature_extractor, model_name)
                        data.append([image_path, *feats])
                    except Exception as e:
                        print(f"Erro ao processar a imagem {image_path}: {e}")
    if data:
        num_features = len(data[0]) - 1
    else:
        num_features = 0
    columns = ['image_path'] + [f'feature_{i}' for i in range(num_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

def save_dataframe_to_csv(df, save_path, file_name, model_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    full_path = os.path.join(save_path, file_name + '.csv')
    df.to_csv(full_path, index=False)
    print(f"Arquivos do modelo {model_name} salvos em: {full_path}")

# organiza as imagens soltas em subpastas por personagem antes de extrair
def organizar_imagens_em_pastas(pasta_base):
    print("[-] A organizar imagens em subpastas...")
    arquivos_movidos = 0
    for arquivo in os.listdir(pasta_base):
        # Verifica se é uma imagem solta (ex: bart001.bmp)
        if arquivo.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            caminho_antigo = os.path.join(pasta_base, arquivo)
            
            # Se for um ficheiro e não uma pasta
            if os.path.isfile(caminho_antigo):
                # Extrai as letras (a classe) ignorando os números
                match = re.match(r"([a-zA-Z]+)", arquivo)
                if match:
                    personagem = match.group(1).lower()
                    pasta_destino = os.path.join(pasta_base, personagem)
                    
                    # Cria a pasta se não existir
                    os.makedirs(pasta_destino, exist_ok=True)
                    
                    # Move a imagem
                    caminho_novo = os.path.join(pasta_destino, arquivo)
                    shutil.move(caminho_antigo, caminho_novo)
                    arquivos_movidos += 1
                    
    if arquivos_movidos > 0:
        print(f"[+] Organização concluída! {arquivos_movidos} imagens movidas para as respetivas pastas.")
    else:
        print("[!] Nenhuma imagem solta encontrada. Assumindo que já estão organizadas.")


if __name__ == "__main__":
    model_choices = [
        'ViT_large', 
    ]
    
    fonte_folder = './' 
    salvar_folder = './' # Onde queres que o CSV seja guardado
    
    organizar_imagens_em_pastas(fonte_folder)
    
    for model_choice in model_choices:
        try:
            print(f"Processando com o modelo: {model_choice}")
            model, feat_ext = load_models(model_choice)
            df = features_to_df(fonte_folder, model, feat_ext, model_choice)
            file_name = f'result_final_{model_choice}'
            save_dataframe_to_csv(df, salvar_folder, file_name, model_choice)
        except Exception as e:
            print(f"Erro ao processar o modelo {model_choice}: {e}")