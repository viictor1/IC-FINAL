"""
Trabalho Final - Inteligência Computacional
UTFPR-CM | Prof. Dr. Diego Bertolini
Métodos Avançados de Fusão de Classificadores
  - Hard Voting (baseline)
  - Soft Voting (baseline)
  - Weighted Hard Voting (pesos = acurácia individual)
  - Top-K Ensemble (apenas os K melhores classificadores)
  - Stacking (meta-aprendizado com LogisticRegression)

Execute DEPOIS do script principal (simpsons_classificacao.py).
Depende do arquivo resultados.csv gerado pelo script principal.
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# ============================================================
# 1. RECARREGA OS DADOS (igual ao script principal)
# ============================================================
print("=" * 60)
print("  MÉTODOS AVANÇADOS DE FUSÃO — SIMPSONS")
print("=" * 60)
print("\n[1] Recarregando dados...")

df = pd.read_csv('result_final_ViT_large.csv')
X = df.drop(columns=['image_path']).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y_strings = df['image_path'].apply(lambda x: os.path.basename(os.path.dirname(x))).values
le = LabelEncoder()
y = le.fit_transform(y_strings)
print(f"    {X.shape[0]} amostras, {X.shape[1]} features, {len(le.classes_)} classes")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ============================================================
# 2. POOL (mesmo do script principal)
# ============================================================
pool = []
for k in [1, 3, 5, 7]:
    pool.append((f'kNN (k={k})', KNeighborsClassifier(n_neighbors=k)))
for d in [5, 10, 15]:
    pool.append((f'DT (prof={d})', DecisionTreeClassifier(max_depth=d, random_state=42)))
pool.append(('DT (entropy)', DecisionTreeClassifier(criterion='entropy', random_state=42)))
for c in [0.1, 1.0, 10.0]:
    pool.append((f'SVM-RBF (C={c})', SVC(C=c, kernel='rbf', probability=True, random_state=42)))
pool.append(('SVM-Linear', SVC(kernel='linear', probability=True, random_state=42)))
for n in [10, 50, 100, 200]:
    pool.append((f'RF ({n} árvores)', RandomForestClassifier(n_estimators=n, random_state=42)))
for topo, desc in zip([(50,), (100,), (50, 25), (100, 50)], ['(50)', '(100)', '(50,25)', '(100,50)']):
    pool.append((f'MLP {desc}', MLPClassifier(hidden_layer_sizes=topo, max_iter=500, random_state=42)))

# ============================================================
# 3. CARREGA RESULTADOS INDIVIDUAIS (do script anterior)
# ============================================================
print("\n[2] Carregando acurácias individuais de resultados.csv...")
df_res = pd.read_csv('resultados.csv').dropna(subset=['Tempo (s)'])  # exclui linhas dos ensembles
print(f"    {len(df_res)} classificadores carregados")

# Mapeia nome -> acurácia (para usar como peso)
acc_individual = dict(zip(df_res['Classificador'], df_res['Acurácia (%)'] / 100))

# ============================================================
# 4. BASELINES (referência rápida)
# ============================================================
print("\n[3] Calculando baselines (Hard/Soft Voting)...")

ens_hard = VotingClassifier(estimators=pool, voting='hard')
y_pred_hard = cross_val_predict(ens_hard, X, y, cv=cv, n_jobs=-1)
acc_hard = accuracy_score(y, y_pred_hard) * 100
f1_hard  = f1_score(y, y_pred_hard, average='macro') * 100
print(f"    Hard Voting:  Acc={acc_hard:.2f}%  F1={f1_hard:.2f}%")

ens_soft = VotingClassifier(estimators=pool, voting='soft')
y_pred_soft = cross_val_predict(ens_soft, X, y, cv=cv, n_jobs=-1)
acc_soft = accuracy_score(y, y_pred_soft) * 100
f1_soft  = f1_score(y, y_pred_soft, average='macro') * 100
print(f"    Soft Voting:  Acc={acc_soft:.2f}%  F1={f1_soft:.2f}%")

# ============================================================
# 5. WEIGHTED SOFT VOTING
#    Peso de cada classificador = sua acurácia no CV individual
#    Classificadores melhores têm mais influência na decisão final
# ============================================================
print("\n[4] Weighted Soft Voting (peso = acurácia individual)...")

pesos = [acc_individual.get(nome, 0.5) for nome, _ in pool]
print("    Pesos atribuídos:")
for (nome, _), p in zip(pool, pesos):
    print(f"      {nome:<24} peso={p:.4f}")

ens_weighted = VotingClassifier(estimators=pool, voting='soft', weights=pesos)
y_pred_weighted = cross_val_predict(ens_weighted, X, y, cv=cv, n_jobs=-1)
acc_weighted = accuracy_score(y, y_pred_weighted) * 100
f1_weighted  = f1_score(y, y_pred_weighted, average='macro') * 100
print(f"\n    Weighted Soft Voting:  Acc={acc_weighted:.2f}%  F1={f1_weighted:.2f}%")

# ============================================================
# 6. TOP-K ENSEMBLES
#    Seleção dos K melhores classificadores individuais
#    Testa K = 5, 10, 15 para encontrar o ponto ótimo
# ============================================================
print("\n[5] Top-K Ensemble (seleciona apenas os K melhores)...")

df_sorted = df_res.sort_values('Acurácia (%)', ascending=False)
pool_dict  = dict(pool)

resultados_topk = []
for k in [5, 8, 10, 12, 15]:
    top_nomes = df_sorted['Classificador'].head(k).tolist()
    top_pool  = [(n, pool_dict[n]) for n in top_nomes if n in pool_dict]

    ens_topk = VotingClassifier(estimators=top_pool, voting='soft')
    y_pred_k  = cross_val_predict(ens_topk, X, y, cv=cv, n_jobs=-1)
    acc_k     = accuracy_score(y, y_pred_k) * 100
    f1_k      = f1_score(y, y_pred_k, average='macro') * 100
    resultados_topk.append({'K': k, 'Acurácia (%)': acc_k, 'F1-Score Macro (%)': f1_k})
    print(f"    Top-{k:02d}: Acc={acc_k:.2f}%  F1={f1_k:.2f}%")

df_topk = pd.DataFrame(resultados_topk)
melhor_topk = df_topk.loc[df_topk['Acurácia (%)'].idxmax()]

# ============================================================
# 7. STACKING (meta-aprendizado)
#    Nível 0: os 20 classificadores
#    Nível 1: Regressão Logística treinada nas predições do nível 0
#    O meta-learner aprende QUAIS classificadores confiar mais
# ============================================================
print("\n[6] Stacking com Regressão Logística como meta-learner...")

# Para stacking, usamos apenas os classificadores com probability=True
# e que não são repetitivos demais. Usamos o pool completo (scikit-learn
# cuida do passthrough automaticamente).
meta_learner = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

stacking = StackingClassifier(
    estimators=pool,
    final_estimator=meta_learner,
    cv=5,           # CV interno para gerar as meta-features
    passthrough=True,  # inclui X original junto das predições como features do meta-learner
    n_jobs=-1
)

y_pred_stack = cross_val_predict(stacking, X, y, cv=cv, n_jobs=-1)
acc_stack = accuracy_score(y, y_pred_stack) * 100
f1_stack  = f1_score(y, y_pred_stack, average='macro') * 100
print(f"    Stacking (LR meta):  Acc={acc_stack:.2f}%  F1={f1_stack:.2f}%")

# ============================================================
# 8. TABELA COMPARATIVA FINAL
# ============================================================
melhor_ind_acc = df_res['Acurácia (%)'].max()
melhor_ind_nome = df_res.loc[df_res['Acurácia (%)'].idxmax(), 'Classificador']

print("\n" + "=" * 65)
print("  COMPARATIVO FINAL — TODOS OS MÉTODOS DE FUSÃO")
print("=" * 65)
print(f"  {'Método':<35} {'Acurácia (%)':>12} {'F1-Score (%)':>12}")
print("  " + "-" * 60)
print(f"  {'Melhor individual: '+melhor_ind_nome:<35} {melhor_ind_acc:>12.2f} {'—':>12}")
print("  " + "-" * 60)
metodos = [
    ("Hard Voting (20 clf)", acc_hard, f1_hard),
    ("Soft Voting (20 clf)", acc_soft, f1_soft),
    ("Weighted Soft Voting (20 clf)", acc_weighted, f1_weighted),
    (f"Top-{int(melhor_topk['K'])} Ensemble (Soft)", melhor_topk['Acurácia (%)'], melhor_topk['F1-Score Macro (%)']),
    ("Stacking (LR meta-learner)", acc_stack, f1_stack),
]
for nome, acc, f1 in metodos:
    delta = acc - melhor_ind_acc
    print(f"  {nome:<35} {acc:>12.2f} {f1:>12.2f}   (Δ{delta:+.2f}%)")
print("=" * 65)

# ============================================================
# 9. GRÁFICOS
# ============================================================
print("\n[7] Gerando gráficos comparativos...")

# --- 9.1: Comparativo de todos os métodos de fusão ---
nomes_fusao = [
    f'Melhor Individual\n({melhor_ind_nome})',
    'Hard Voting\n(20 clf)',
    'Soft Voting\n(20 clf)',
    'Weighted\nSoft Voting',
    f'Top-{int(melhor_topk["K"])}\nEnsemble',
    'Stacking\n(LR meta)',
]
accs_fusao = [
    melhor_ind_acc, acc_hard, acc_soft,
    acc_weighted, melhor_topk['Acurácia (%)'], acc_stack
]
cores_fusao = ['#9E9E9E', '#4C72B0', '#4C72B0', '#DD8452', '#55A868', '#C44E52']
alphas = [0.6, 0.7, 1.0, 1.0, 1.0, 1.0]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(nomes_fusao, accs_fusao, color=cores_fusao, edgecolor='white', linewidth=0.5)
for bar, val, alpha in zip(bars, accs_fusao, alphas):
    bar.set_alpha(alpha)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(melhor_ind_acc, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
           label=f'Melhor individual ({melhor_ind_acc:.1f}%)')
ax.set_ylabel('Acurácia (%)', fontsize=11)
ax.set_title('Comparativo dos Métodos de Fusão de Classificadores', fontsize=13)
ax.set_ylim(0, max(accs_fusao) + 8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig7_comparativo_fusao.png', dpi=300)
plt.close()
print("    -> fig7_comparativo_fusao.png")

# --- 9.2: Efeito do K no Top-K Ensemble ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df_topk['K'], df_topk['Acurácia (%)'], marker='o', color='#55A868',
        linewidth=2, markersize=8, label='Acurácia')
ax.plot(df_topk['K'], df_topk['F1-Score Macro (%)'], marker='s', color='#4C72B0',
        linewidth=2, markersize=8, label='F1-Score Macro', linestyle='--')
ax.axhline(melhor_ind_acc, color='gray', linestyle=':', alpha=0.7, label=f'Melhor individual')
ax.axhline(acc_hard, color='#4C72B0', linestyle=':', alpha=0.5, label='Hard Voting (20)')
ax.set_xlabel('K (número de classificadores no ensemble)', fontsize=11)
ax.set_ylabel('Taxa (%)', fontsize=11)
ax.set_title('Impacto do Tamanho do Top-K Ensemble na Acurácia', fontsize=12)
ax.set_xticks(df_topk['K'])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig8_topk_ensemble.png', dpi=300)
plt.close()
print("    -> fig8_topk_ensemble.png")

# --- 9.3: Matriz de confusão do melhor método de fusão ---
accs_todos = [acc_hard, acc_soft, acc_weighted, melhor_topk['Acurácia (%)'], acc_stack]
y_preds    = [y_pred_hard, y_pred_soft, y_pred_weighted, None, y_pred_stack]
nomes_todos = ['Hard Voting', 'Soft Voting', 'Weighted Soft Voting', f'Top-{int(melhor_topk["K"])} Ensemble', 'Stacking']

idx_melhor = np.argmax(accs_todos)
if y_preds[idx_melhor] is not None:
    y_pred_melhor_fusao = y_preds[idx_melhor]
else:
    # Recalcula Top-K se for o melhor
    k_melhor = int(melhor_topk['K'])
    top_nomes = df_sorted['Classificador'].head(k_melhor).tolist()
    top_pool  = [(n, pool_dict[n]) for n in top_nomes if n in pool_dict]
    ens_best  = VotingClassifier(estimators=top_pool, voting='soft')
    y_pred_melhor_fusao = cross_val_predict(ens_best, X, y, cv=cv, n_jobs=-1)

cm_best = confusion_matrix(y, y_pred_melhor_fusao)
cm_best_pct = cm_best.astype(float) / cm_best.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_best_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            linewidths=0.5, linecolor='gray')
ax.set_title(f'Matriz de Confusão (%) — {nomes_todos[idx_melhor]}\n(Melhor método de fusão: {accs_todos[idx_melhor]:.2f}%)', fontsize=11)
ax.set_ylabel('Classe Real', fontsize=11)
ax.set_xlabel('Classe Predita', fontsize=11)
plt.tight_layout()
plt.savefig('fig9_matriz_melhor_fusao.png', dpi=300)
plt.close()
print(f"    -> fig9_matriz_melhor_fusao.png  ({nomes_todos[idx_melhor]})")

# ============================================================
# 10. EXPORTAÇÃO
# ============================================================
df_fusao = pd.DataFrame([
    {'Método': 'Hard Voting (20 clf)',         'Acurácia (%)': acc_hard,     'F1-Score Macro (%)': f1_hard},
    {'Método': 'Soft Voting (20 clf)',          'Acurácia (%)': acc_soft,     'F1-Score Macro (%)': f1_soft},
    {'Método': 'Weighted Soft Voting (20 clf)', 'Acurácia (%)': acc_weighted, 'F1-Score Macro (%)': f1_weighted},
    {'Método': f'Top-{int(melhor_topk["K"])} Ensemble (Soft)', 'Acurácia (%)': melhor_topk['Acurácia (%)'], 'F1-Score Macro (%)': melhor_topk['F1-Score Macro (%)']},
    {'Método': 'Stacking (LR meta-learner)',    'Acurácia (%)': acc_stack,    'F1-Score Macro (%)': f1_stack},
])
df_fusao.to_csv('resultados_fusao.csv', index=False, float_format='%.2f')

print("\n" + "=" * 60)
print("  ARQUIVOS GERADOS")
print("=" * 60)
print("    resultados_fusao.csv         (tabela de fusão para o artigo)")
print("    fig7_comparativo_fusao.png")
print("    fig8_topk_ensemble.png")
print("    fig9_matriz_melhor_fusao.png")
print("=" * 60)

# ============================================================
# 11. PROTOCOLO B — VALIDAÇÃO (holdout)
#     Treina os ensembles no Train completo e avalia no Valid.
#     Exigido pelo enunciado: reportar teste E validação.
# ============================================================
VALID_CSV = 'result_final_ViT_large_valid.csv'
if os.path.exists(VALID_CSV):
    print("\n[8] Protocolo B — Validação (holdout)...")
    df_va = pd.read_csv(VALID_CSV)
    Xva = scaler.transform(df_va.drop(columns=['image_path']).values)
    yva_str = df_va['image_path'].apply(lambda p: os.path.basename(os.path.dirname(p))).values
    yva = le.transform(yva_str)
    print(f"    Valid: {Xva.shape[0]} amostras")

    res_va = []

    def _aval_va(nome, estimador):
        estimador.fit(X, y)
        yp = estimador.predict(Xva)
        acc = accuracy_score(yva, yp) * 100
        f1 = f1_score(yva, yp, average='macro') * 100
        res_va.append({'Método': nome, 'Acurácia (%)': acc, 'F1-Score Macro (%)': f1})
        print(f"    {nome:<32} acc={acc:6.2f}%  f1={f1:6.2f}%")
        return yp

    yph_va = _aval_va('Hard Voting (20 clf)', VotingClassifier(estimators=pool, voting='hard'))
    yps_va = _aval_va('Soft Voting (20 clf)', VotingClassifier(estimators=pool, voting='soft'))
    _aval_va('Weighted Soft Voting (20 clf)', VotingClassifier(estimators=pool, voting='soft', weights=pesos))

    # Top-K na validação: escolhe o melhor K (mesmo conjunto testado no CV)
    melhor_topk_va = None
    for k in [5, 8, 10, 12, 15]:
        top_nomes = df_sorted['Classificador'].head(k).tolist()
        top_pool = [(n, pool_dict[n]) for n in top_nomes if n in pool_dict]
        ek = VotingClassifier(estimators=top_pool, voting='soft').fit(X, y)
        yp = ek.predict(Xva)
        acc = accuracy_score(yva, yp) * 100
        if melhor_topk_va is None or acc > melhor_topk_va[1]:
            melhor_topk_va = (k, acc, f1_score(yva, yp, average='macro') * 100)
    res_va.append({'Método': f'Top-{melhor_topk_va[0]} Ensemble (Soft)',
                   'Acurácia (%)': melhor_topk_va[1], 'F1-Score Macro (%)': melhor_topk_va[2]})
    print(f"    {'Top-'+str(melhor_topk_va[0])+' Ensemble (Soft)':<32} acc={melhor_topk_va[1]:6.2f}%  f1={melhor_topk_va[2]:6.2f}%")

    _aval_va('Stacking (LR meta-learner)',
             StackingClassifier(estimators=pool,
                                final_estimator=LogisticRegression(max_iter=1000, random_state=42, C=1.0),
                                cv=5, passthrough=True, n_jobs=-1))

    df_fusao_va = pd.DataFrame(res_va)
    df_fusao_va.to_csv('resultados_fusao_valid.csv', index=False, float_format='%.2f')
    print("    -> resultados_fusao_valid.csv")

    # Matriz de confusão do melhor método de fusão na validação
    idx_best_va = df_fusao_va['Acurácia (%)'].idxmax()
    nome_best_va = df_fusao_va.loc[idx_best_va, 'Método']
    # recalcula predição do melhor (cobre Hard/Soft direto; Top-K/Stacking refazem)
    if nome_best_va.startswith('Hard'):
        yp_best_va = yph_va
    elif nome_best_va.startswith('Soft'):
        yp_best_va = yps_va
    elif nome_best_va.startswith('Top-'):
        k = melhor_topk_va[0]
        top_nomes = df_sorted['Classificador'].head(k).tolist()
        top_pool = [(n, pool_dict[n]) for n in top_nomes if n in pool_dict]
        yp_best_va = VotingClassifier(estimators=top_pool, voting='soft').fit(X, y).predict(Xva)
    elif nome_best_va.startswith('Weighted'):
        yp_best_va = VotingClassifier(estimators=pool, voting='soft', weights=pesos).fit(X, y).predict(Xva)
    else:
        yp_best_va = StackingClassifier(
            estimators=pool, final_estimator=LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            cv=5, passthrough=True, n_jobs=-1).fit(X, y).predict(Xva)

    cm_va = confusion_matrix(yva, yp_best_va).astype(float)
    cm_va = cm_va / cm_va.sum(axis=1)[:, np.newaxis] * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_va, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title(f'Matriz de Confusão (%) Validação — {nome_best_va}\n'
                 f'({df_fusao_va.loc[idx_best_va, "Acurácia (%)"]:.2f}%)', fontsize=11)
    ax.set_ylabel('Classe Real'); ax.set_xlabel('Classe Predita')
    plt.tight_layout()
    plt.savefig('fig9b_matriz_melhor_fusao_valid.png', dpi=300)
    plt.close()
    print("    -> fig9b_matriz_melhor_fusao_valid.png")
else:
    print(f"\n[!] {VALID_CSV} não encontrado — rode extrai_valid.py e simpsons_classificacao.py antes.")

print("\n[✓] Concluído.")