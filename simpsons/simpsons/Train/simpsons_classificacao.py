"""
Classificacao da base Simpsons (5 classes).

Usa as deep features do ViT-large (1024-D) geradas pelo extraicarac.py e roda
20 classificadores das 5 familias pedidas (k-NN, arvore, SVM, RF e MLP).

Faz as duas avaliacoes que o enunciado pede: 10-fold cross-validation no Train
e um holdout no Valid (treina no Train inteiro e testa no Valid).

Gera resultados.csv / resultados_valid.csv e as figuras fig1..fig6 (+ fig3b/4b/5b).
Rodar antes do classificadorInicial.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "result_final_ViT_large.csv")
VALID_CSV = os.path.join(HERE, "result_final_ViT_large_valid.csv")


# funcoes auxiliares
def carregar(csv_path, scaler=None, le=None, fit=False):
    """Carrega um CSV de features. Deriva a classe pelo nome da subpasta."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["image_path"]).values
    y_str = df["image_path"].apply(lambda p: os.path.basename(os.path.dirname(p))).values
    if fit:
        scaler = StandardScaler().fit(X)
        le = LabelEncoder().fit(y_str)
    X = scaler.transform(X)
    y = le.transform(y_str)
    return X, y, scaler, le


def construir_pool():
    """Pool de 20 classificadores (5 famílias)."""
    pool = []
    for k in [1, 3, 5, 7]:
        pool.append((f"kNN (k={k})", KNeighborsClassifier(n_neighbors=k)))
    for d in [5, 10, 15]:
        pool.append((f"DT (prof={d})", DecisionTreeClassifier(max_depth=d, random_state=42)))
    pool.append(("DT (entropy)", DecisionTreeClassifier(criterion="entropy", random_state=42)))
    for c in [0.1, 1.0, 10.0]:
        pool.append((f"SVM-RBF (C={c})", SVC(C=c, kernel="rbf", probability=True, random_state=42)))
    pool.append(("SVM-Linear", SVC(kernel="linear", probability=True, random_state=42)))
    for n in [10, 50, 100, 200]:
        pool.append((f"RF ({n} árvores)", RandomForestClassifier(n_estimators=n, random_state=42)))
    for topo, desc in zip([(50,), (100,), (50, 25), (100, 50)], ["(50)", "(100)", "(50,25)", "(100,50)"]):
        pool.append((f"MLP {desc}", MLPClassifier(hidden_layer_sizes=topo, max_iter=500, random_state=42)))
    return pool


def familia(nome):
    if nome.startswith("kNN"):
        return "k-NN"
    if nome.startswith("DT"):
        return "Árvore de Decisão"
    if nome.startswith("SVM"):
        return "SVM"
    if nome.startswith("RF"):
        return "Random Forest"
    if nome.startswith("MLP"):
        return "MLP"
    return "Fusão"


CORES_FAMILIA = {
    "k-NN": "#4C72B0",
    "Árvore de Decisão": "#DD8452",
    "SVM": "#55A868",
    "Random Forest": "#C44E52",
    "MLP": "#8172B3",
    "Fusão": "#000000",
}


def matriz_confusao(y_true, y_pred, classes, titulo, arquivo):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel("Classe Real", fontsize=11)
    ax.set_xlabel("Classe Predita", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, arquivo), dpi=300)
    plt.close()


def barras_horizontais(df, coluna, titulo, arquivo, ref=None):
    """Gráfico de barras horizontais colorido por família."""
    df = df.sort_values(["Família", coluna])
    cores = [CORES_FAMILIA[f] for f in df["Família"]]
    fig, ax = plt.subplots(figsize=(11, 8))
    barras = ax.barh(df["Classificador"], df[coluna], color=cores)
    for b, v in zip(barras, df[coluna]):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9)
    if ref is not None:
        ax.axvline(ref, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel(coluna, fontsize=11)
    ax.set_title(titulo, fontsize=13)
    ax.set_xlim(0, max(df[coluna]) + 8)
    # legenda por família
    from matplotlib.patches import Patch
    fams = [f for f in CORES_FAMILIA if f in set(df["Família"])]
    ax.legend(handles=[Patch(color=CORES_FAMILIA[f], label=f) for f in fams], loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, arquivo), dpi=300)
    plt.close()


# avaliacao
def avaliar_cv(pool, X, y, cv):
    """Protocolo A: 10-fold CV. Retorna DataFrame e predições de voting."""
    import time
    linhas = []
    print("\n[A] TESTE — 10-fold cross-validation (Train)")
    for nome, clf in pool:
        t0 = time.time()
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
        dt = time.time() - t0
        acc = accuracy_score(y, y_pred) * 100
        f1 = f1_score(y, y_pred, average="macro") * 100
        linhas.append({"Classificador": nome, "Acurácia (%)": acc,
                       "F1-Score Macro (%)": f1, "Tempo (s)": dt, "Família": familia(nome)})
        print(f"    {nome:<22} acc={acc:6.2f}%  f1={f1:6.2f}%")
    return pd.DataFrame(linhas)


def avaliar_holdout(pool, Xtr, ytr, Xte, yte):
    """Protocolo B: treina no Train, prediz no Valid."""
    linhas = []
    print("\n[B] VALIDAÇÃO — holdout (treina Train, testa Valid)")
    for nome, clf in pool:
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        acc = accuracy_score(yte, y_pred) * 100
        f1 = f1_score(yte, y_pred, average="macro") * 100
        linhas.append({"Classificador": nome, "Acurácia (%)": acc,
                       "F1-Score Macro (%)": f1, "Família": familia(nome)})
        print(f"    {nome:<22} acc={acc:6.2f}%  f1={f1:6.2f}%")
    return pd.DataFrame(linhas)


def main():
    print("=" * 60)
    print("  CLASSIFICAÇÃO SIMPSONS — ViT-large deep features")
    print("=" * 60)

    Xtr, ytr, scaler, le = carregar(TRAIN_CSV, fit=True)
    classes = le.classes_
    print(f"  Train: {Xtr.shape[0]} amostras, {Xtr.shape[1]} features, {len(classes)} classes")

    pool = construir_pool()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ---------- Protocolo A: 10-fold CV ----------
    df_cv = avaliar_cv(pool, Xtr, ytr, cv)

    # Hard / Soft voting (baselines de fusão) no CV
    ens_hard = VotingClassifier(estimators=pool, voting="hard")
    yph = cross_val_predict(ens_hard, Xtr, ytr, cv=cv, n_jobs=-1)
    acc_h, f1_h = accuracy_score(ytr, yph) * 100, f1_score(ytr, yph, average="macro") * 100
    ens_soft = VotingClassifier(estimators=pool, voting="soft")
    yps = cross_val_predict(ens_soft, Xtr, ytr, cv=cv, n_jobs=-1)
    acc_s, f1_s = accuracy_score(ytr, yps) * 100, f1_score(ytr, yps, average="macro") * 100
    print(f"\n    Hard Voting: acc={acc_h:.2f}%  f1={f1_h:.2f}%")
    print(f"    Soft Voting: acc={acc_s:.2f}%  f1={f1_s:.2f}%")

    # Salva resultados.csv (individuais + voting), formato igual ao original
    df_out = df_cv[["Classificador", "Acurácia (%)", "F1-Score Macro (%)", "Tempo (s)"]] \
        .sort_values("Acurácia (%)", ascending=False).copy()
    fusao_rows = pd.DataFrame([
        {"Classificador": "Fusão Hard Voting", "Acurácia (%)": acc_h, "F1-Score Macro (%)": f1_h, "Tempo (s)": np.nan},
        {"Classificador": "Fusão Soft Voting", "Acurácia (%)": acc_s, "F1-Score Macro (%)": f1_s, "Tempo (s)": np.nan},
    ])
    pd.concat([df_out, fusao_rows], ignore_index=True).to_csv(
        os.path.join(HERE, "resultados.csv"), index=False, float_format="%.2f")
    print("    -> resultados.csv")

    # ---------- Figuras 1, 2, 3, 4, 5, 6 (Protocolo A) ----------
    df_plot = df_cv.copy()
    # adiciona linhas de fusão para fig1/fig2
    df_plot = pd.concat([df_plot, pd.DataFrame([
        {"Classificador": "Soft Voting", "Acurácia (%)": acc_s, "F1-Score Macro (%)": f1_s, "Família": "Fusão"},
        {"Classificador": "Hard Voting", "Acurácia (%)": acc_h, "F1-Score Macro (%)": f1_h, "Família": "Fusão"},
    ])], ignore_index=True)
    melhor_ind_acc = df_cv["Acurácia (%)"].max()
    melhor_ind = df_cv.loc[df_cv["Acurácia (%)"].idxmax(), "Classificador"]

    barras_horizontais(df_plot, "Acurácia (%)",
                        "Acurácia dos Classificadores Individuais e Sistemas de Fusão",
                        "fig1_comparativo_acuracia.png", ref=melhor_ind_acc)
    barras_horizontais(df_plot, "F1-Score Macro (%)",
                        "F1-Score Macro dos Classificadores Individuais e Sistemas de Fusão",
                        "fig2_comparativo_f1.png", ref=df_cv["F1-Score Macro (%)"].max())
    print("    -> fig1_comparativo_acuracia.png / fig2_comparativo_f1.png")

    # Matriz do melhor individual (recalcula predição via CV)
    melhor_clf = dict(pool)[melhor_ind]
    yp_melhor = cross_val_predict(melhor_clf, Xtr, ytr, cv=cv, n_jobs=-1)
    matriz_confusao(ytr, yp_melhor, classes,
                    f"Matriz de Confusão (%) — Melhor Individual: {melhor_ind} ({melhor_ind_acc:.2f}%)",
                    "fig3_matriz_melhor_individual.png")
    matriz_confusao(ytr, yph, classes,
                    f"Matriz de Confusão (%) — Hard Voting ({acc_h:.2f}%)",
                    "fig4_matriz_hard_voting.png")
    matriz_confusao(ytr, yps, classes,
                    f"Matriz de Confusão (%) — Soft Voting ({acc_s:.2f}%)",
                    "fig5_matriz_soft_voting.png")
    print("    -> fig3/4/5 matrizes (CV)")

    # Média por família
    med = df_cv.groupby("Família")["Acurácia (%)"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(med.index, med.values, color=[CORES_FAMILIA[f] for f in med.index])
    for i, v in enumerate(med.values):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Acurácia média (%)")
    ax.set_title("Acurácia Média por Família de Classificador (10-fold CV)")
    ax.set_xlim(0, max(med.values) + 10)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig6_media_por_familia.png"), dpi=300)
    plt.close()
    print("    -> fig6_media_por_familia.png")

    # ---------- Protocolo B: Validação (holdout) ----------
    if os.path.exists(VALID_CSV):
        Xva, yva, _, _ = carregar(VALID_CSV, scaler=scaler, le=le, fit=False)
        print(f"\n  Valid: {Xva.shape[0]} amostras")
        df_va = avaliar_holdout(construir_pool(), Xtr, ytr, Xva, yva)

        # Hard/Soft voting no holdout
        eh = VotingClassifier(estimators=construir_pool(), voting="hard").fit(Xtr, ytr)
        es = VotingClassifier(estimators=construir_pool(), voting="soft").fit(Xtr, ytr)
        yph_v, yps_v = eh.predict(Xva), es.predict(Xva)
        ah, fh = accuracy_score(yva, yph_v) * 100, f1_score(yva, yph_v, average="macro") * 100
        as_, fs = accuracy_score(yva, yps_v) * 100, f1_score(yva, yps_v, average="macro") * 100
        print(f"\n    Hard Voting (Valid): acc={ah:.2f}%  f1={fh:.2f}%")
        print(f"    Soft Voting (Valid): acc={as_:.2f}%  f1={fs:.2f}%")

        df_va_out = df_va[["Classificador", "Acurácia (%)", "F1-Score Macro (%)"]] \
            .sort_values("Acurácia (%)", ascending=False)
        pd.concat([df_va_out, pd.DataFrame([
            {"Classificador": "Fusão Hard Voting", "Acurácia (%)": ah, "F1-Score Macro (%)": fh},
            {"Classificador": "Fusão Soft Voting", "Acurácia (%)": as_, "F1-Score Macro (%)": fs},
        ])], ignore_index=True).to_csv(
            os.path.join(HERE, "resultados_valid.csv"), index=False, float_format="%.2f")
        print("    -> resultados_valid.csv")

        melhor_va = df_va.loc[df_va["Acurácia (%)"].idxmax()]
        clf_va = dict(construir_pool())[melhor_va["Classificador"]].fit(Xtr, ytr)
        matriz_confusao(yva, clf_va.predict(Xva), classes,
                        f"Matriz de Confusão (%) Validação — Melhor Individual: "
                        f"{melhor_va['Classificador']} ({melhor_va['Acurácia (%)']:.2f}%)",
                        "fig3b_matriz_melhor_individual_valid.png")
        matriz_confusao(yva, yph_v, classes,
                        f"Matriz de Confusão (%) Validação — Hard Voting ({ah:.2f}%)",
                        "fig4b_matriz_hard_voting_valid.png")
        matriz_confusao(yva, yps_v, classes,
                        f"Matriz de Confusão (%) Validação — Soft Voting ({as_:.2f}%)",
                        "fig5b_matriz_soft_voting_valid.png")
        print("    -> fig3b/4b/5b matrizes (Validação)")
    else:
        print(f"\n  [!] {os.path.basename(VALID_CSV)} não encontrado — rode extrai_valid.py primeiro.")

    print("\n[✓] Concluído.")


if __name__ == "__main__":
    main()
