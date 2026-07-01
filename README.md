# Trabalho Final — Inteligência Computacional (UTFPR-CM)

Classificação de personagens de *The Simpsons* (5 classes: bart, homer, lisa,
maggie, marge) usando **deep features** do ViT-large + **20 classificadores** com
**fusão estática**.

## Estrutura

```
IC-TRAB/
├── artigo/
│   ├── main.tex                  # artigo SBC (8 págs) — compilar no Overleaf
│   └── figs/                     # figuras usadas no artigo
├── simpsons/simpsons/
│   ├── Train/                    # 226 imagens (organizadas por classe) + scripts
│   │   ├── extraicarac.py        # extração de features (multi-modelo) — Train
│   │   ├── extrai_valid.py       # extração de features (ViT-large) — Valid
│   │   ├── simpsons_classificacao.py   # 20 classificadores: CV(Train) + holdout(Valid)
│   │   ├── classificadorInicial.py     # fusão avançada: CV(Train) + holdout(Valid)
│   │   ├── result_final_ViT_large.csv       # features Train (226 x 1024)
│   │   ├── result_final_ViT_large_valid.csv # features Valid (95 x 1024)
│   │   ├── resultados.csv / resultados_valid.csv
│   │   ├── resultados_fusao.csv / resultados_fusao_valid.csv
│   │   └── fig*.png
│   └── Valid/                    # 95 imagens (conjunto de validação)
└── modelosparapublicaodeartigos/ # template SBC original
```

## Ambiente

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn transformers timm tqdm Pillow
```

## Ordem de execução

Todos os comandos a partir de `simpsons/simpsons/Train/`:

1. **Extração de features** (já feita para o Train; rode para o Valid):
   ```bash
   python extrai_valid.py          # gera result_final_ViT_large_valid.csv
   # (Train: python extraicarac.py — requer open_clip/timm para os modelos extras)
   ```
2. **Classificadores individuais** (10-fold CV no Train + holdout no Valid):
   ```bash
   python simpsons_classificacao.py
   # -> resultados.csv, resultados_valid.csv, fig1..fig6, fig3b/4b/5b
   ```
3. **Fusão de classificadores** (5 métodos, CV + holdout):
   ```bash
   python classificadorInicial.py
   # -> resultados_fusao.csv, resultados_fusao_valid.csv, fig7..fig9, fig9b
   ```

## Artigo

`artigo/` é autossuficiente (já inclui o `sbc-template.sty` oficial). Compila com:

```bash
cd artigo && pdflatex main.tex && pdflatex main.tex   # 2 passadas p/ as referências
```

Também pode subir a pasta para o Overleaf. Tem **8 páginas**, sem abstract/resumo
(dispensado pelo enunciado). **Falta apenas preencher os nomes dos autores** (linha
`\author{...}` e o `\address{...}` de e-mail em `main.tex`).

## Protocolo de avaliação

- **Teste**: validação cruzada estratificada de 10 folds sobre o Train.
- **Validação**: treino no Train completo, avaliação no holdout Valid (divisão
  proposta nos arquivos da base).
- Métricas: acurácia, F1-macro e matriz de confusão (%) por classe.
