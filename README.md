# Trabalho Final - Inteligencia Computacional (UTFPR-CM)

Classificacao de personagens de The Simpsons (5 classes: bart, homer, lisa,
maggie, marge). A ideia e extrair deep features com o ViT-large, rodar 20
classificadores e combinar eles por fusao estatica.

## Estrutura

```
IC-TRAB/
├── simpsons/simpsons/
│   ├── Train/          # 226 imagens por classe + os scripts
│   │   ├── extraicarac.py              # extrai as features do Train
│   │   ├── extrai_valid.py             # extrai as features do Valid (ViT-large)
│   │   ├── simpsons_classificacao.py   # os 20 classificadores (CV + holdout)
│   │   ├── classificadorInicial.py     # fusao dos classificadores (CV + holdout)
│   │   ├── result_final_ViT_large.csv        # features do Train (226 x 1024)
│   │   ├── result_final_ViT_large_valid.csv  # features do Valid (95 x 1024)
│   │   ├── resultados*.csv / resultados_fusao*.csv
│   │   └── fig*.png
│   └── Valid/          # 95 imagens (validacao)
└── ...
```

## Como rodar

Instala as dependencias:

```bash
pip install pandas scikit-learn matplotlib seaborn transformers timm tqdm Pillow
```

Depois, dentro de `simpsons/simpsons/Train/`, roda nesta ordem:

```bash
python extraicarac.py              # gera as features do Train
python extrai_valid.py             # gera as features do Valid
python simpsons_classificacao.py   # 20 classificadores -> resultados.csv, fig1..fig6
python classificadorInicial.py     # fusao -> resultados_fusao.csv, fig7..fig9
```

As features do Train (`result_final_ViT_large.csv`) ja foram geradas pelo
`extraicarac.py`. Se precisar refazer, esse script usa open_clip/timm alem do
transformers.
