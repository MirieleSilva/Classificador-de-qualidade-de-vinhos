import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.base import clone


# O OBJETIVO DESDE PROJETO É CLASSIFICAR A QUALIDADE DO VINHO COMO BOM (1) OU RUIM (0) COM BASE NAS CARACTERÍSTICAS QUÍMICAS DISPONÍVEIS, E ENCONTRAR O MELHOR MODELO PARA FAZER ISSO


# PREPARAÇÃO DA BASE DE DADOS

data = pd.read_csv('winequalityN.csv')

estatisticas = data.describe().transpose()[['mean', 'std']]
estatisticas.columns = ['Média', 'Desvio Padrão']
print(estatisticas)

data = data.rename(columns={'quality': 'opinion'}) # RENOMEANDO COLUNA QUALITY PARA OPINION
data['opinion'] = data['opinion'].apply(lambda r: 0 if r <= 5 else 1) # CLASSIFICANDO OS VALORES DE QUALITY PARA OPINION
data = data[data['type'] == 'white']
data['type'] = data['type'].map({'white': 1})

# TRATAMENTO DE VALORES NULL
nan_values = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'pH', 'sulphates']
data[nan_values] = data[nan_values].fillna(data[nan_values].mean())

# SEPARAÇÃO DAS VARIAVEIS X E Y

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# CÓDIGOS REUTILIZAVEIS

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

def calcular_metricas(scores):
    return {
        'Média da Acurácia': np.mean(scores['test_accuracy']),
        'Desvio da Acurácia': np.std(scores['test_accuracy']),
        'Média da Precisão': np.mean(scores['test_precision']),
        'Desvio da Precisão': np.std(scores['test_precision']),
        'Média do Recall': np.mean(scores['test_recall']),
        'Desvio do Recall': np.std(scores['test_recall']),
        'Média do F1': np.mean(scores['test_f1']),
        'Desvio do F1': np.std(scores['test_f1'])
    }

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# REGRESSÃO LOGISTICA

# PIPELINE COM SCALER PARA PADRONIZAR OS DADOS E O MODELO DE REGRESSÃO LOGÍSTICA
pipelineRg = Pipeline([('scaler', StandardScaler()),
                      ('model', LogisticRegression(max_iter=1000, random_state=42))])

# VALIDAÇÃO CRUZADA PARA REGRESSÃO LOGISTICA USANDO OS DADOS (X, Y)
# OS DADOS SÃO DIVIDIDOS EM VÁRIAS PARTES (CV=KF) 
# PIPELINE É APLICADO EM TODOS OS DADOS
# O SCORING É USADO PARA MEDIR A ACURÁCIA, PRECISÃO, RECALL E F1
# A VALIDAÇÃO CRUZADA TESTA SE O MODELO REALMENTE FUNCIONA BEM EM DIFERENTES CONJUNTOS DE DADOS.
scoresLog = cross_validate(pipelineRg, x, y, cv=kf, scoring=scoring)

# TREINAMENTO DO MODELO COM TODOS OS DADOS
pipelineRg.fit(x, y)



# ARVORE DE DECISÃO

# PIPELINE COM O MODELO DE ARVORE DE DECISÃO
pipelineTd = Pipeline([
    ('model', DecisionTreeClassifier(random_state=42))
])

# BUSCA OS MELHORES PARÂMETROS PARA A ÁRVORE DE DECISÃO
# ESSA PARTE DO CÓDIGO TESTA DIFERENTES CONFIGURAÇÕES PARA ENCONTRAR O MELHOR MODELO.
params_grid = {
    'model__max_depth': range(2, 8),
    'model__criterion': ['gini', 'entropy'],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5]
}

search = GridSearchCV(
    estimator=pipelineTd, # USA O MODELO "pipelineTd"
    param_grid=params_grid,# USA O CONJUNTO DE PARÂMETROS DEFINIDO ACIMA
    scoring='accuracy',# AVALIA OS MODELOS COM BASE NA PRECISÃO
    cv=kf, # FAZ VALIDAÇÃO CRUZADA PARA TESTAR DIFERENTES PARTES DOS DADOS
    refit=True,# APÓS A BUSCA, USA OS MELHORES PARÂMETROS PARA TREINAR O MODELO FINAL
    error_score=0,# SE DER ALGUM ERRO, ATRIBUI 0 COMO PONTUAÇÃO
    verbose=10 # MOSTRA MAIS DETALHES NO TERMINAL DURANTE A EXECUÇÃO
)

# TREINAMENTO DA ARVORE DE DECISÃO
search.fit(x, y)

# VALIDAÇÃO CRUZADA PARA ARVORE DE DECISÃO USANDO OS DADOS (X, Y)
# OS DADOS SÃO DIVIDIDOS EM VÁRIAS PARTES (CV=KF)
# O SCORING É USADO PARA MEDIR A ACURÁCIA, PRECISÃO, RECALL E F1 
# n_jobs=-1 FAZ UTILIZAR TODOS OS PROCESSADORES DISPONÍVEIS
# A VALIDAÇÃO CRUZADA TESTA SE O MODELO REALMENTE FUNCIONA BEM EM DIFERENTES CONJUNTOS DE DADOS.
scoresTd = cross_validate(search.best_estimator_, x, y, cv=kf, scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1)




# SVM

# PIPELINE COM SCALER PARA PADRONIZAR OS DADOS E O MODELO DE SVM
pipelinesvm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=2.0, probability=True))
])

# VALIDAÇÃO CRUZADA PARA SVM USANDO OS DADOS (X, Y)
# OS DADOS SÃO DIVIDIDOS EM VÁRIAS PARTES (CV=KF) 
# PIPELINE É APLICADO EM TODOS OS DADOS
# O SCORING É USADO PARA MEDIR A ACURÁCIA, PRECISÃO, RECALL E F1
# A VALIDAÇÃO CRUZADA TESTA SE O MODELO REALMENTE FUNCIONA BEM EM DIFERENTES CONJUNTOS DE DADOS.
scoresSvm = cross_validate(pipelinesvm, x, y, cv=kf, scoring=scoring)
pipelinesvm.fit(x, y)



# MÉTRICAS DO 3 MODELOS


# ITERAÇÃO PARA MOSTRAR AS MÉTRICAS DO MODELO DE REGRESSÃO LOGÍSTICA
print("REGRESSÃO LOGISTICA")
for key, value in calcular_metricas(scoresLog).items():
    print(f'{key}: {value}')

# CONTAGEMAS DAS PREVISÕES DOS VINHOS BONS E RUINS    
y_pred_white_log = pipelineRg.predict(x)
num_bons_white_log = (y_pred_white_log == 1).sum()
num_ruins_white_log = (y_pred_white_log == 0).sum()
print(f"Total de Vinhos Bons: {num_bons_white_log}")
print(f"Total de Vinhos Ruins: {num_ruins_white_log}")

# ITERAÇÃO PARA MOSTRAR AS MÉTRICAS DO MODELO DE ARVORE DE DECISÃO
print("ARVORE DE DECISÃO")
for key, value in calcular_metricas(scoresTd).items():
    print(f'{key}: {value}')

 # CONTAGEMAS DAS PREVISÕES DOS VINHOS BONS E RUINS    
y_pred_white_tree = search.best_estimator_.predict(x)
num_bons_white_tree = (y_pred_white_tree == 1).sum()
num_ruins_white_tree = (y_pred_white_tree == 0).sum()
print(f"Total de Vinhos Bons: {num_bons_white_tree}")
print(f"Total de Vinhos Ruins: {num_ruins_white_tree}")

# ITERAÇÃO PARA MOSTRAR AS MÉTRICAS DO MODELO DE SVM
print("SVM")
for key, value in calcular_metricas(scoresSvm).items():
    print(f'{key}: {value}')

# CONTAGEMAS DAS PREVISÕES DOS VINHOS BONS E RUINS   
y_pred_white_svm = pipelinesvm.predict(x)
num_bons_white_svm = (y_pred_white_svm == 1).sum()
num_ruins_white_svm = (y_pred_white_svm == 0).sum()
print(f"Total de Vinhos Bons: {num_bons_white_svm}")
print(f"Total de Vinhos Ruins: {num_ruins_white_svm}")


modelos = {
    "Regressão Logística": pipelineRg,
    "Árvore de Decisão": search.best_estimator_,
    "SVM": pipelinesvm
}


# MÉDIA CURVA ROC PARA OS 3 MODELOS


plt.figure(figsize=(10, 7))

# ITERAÇÃO PARA PLOTAR A CURVA ROC DOS MODELOS
for nome_modelo, modelo in modelos.items():
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    
# VALIDAÇÃO CRUZADA, SEPARANDO OS DADOS EM TREINO E TESTE EM DIFERENTES RODADAS
    for train_idx, test_idx in kf.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# CLONA O MODELO PARA QUE ELE SEJA TREINADO DO ZERO EM CADA RODADA   
        model = clone(modelo)
        model.fit(x_train, y_train)

# PREVÊ AS PROBABILIDADES PARA A CLASSE POSITIVA (1) NOS DADOS DE TESTE          
        y_probs = model.predict_proba(x_test)[:, 1]

# CALCULA A CURVA ROC: TAXA DE FALSOS POSITIVOS (FP) E TAXA DE VERDADEIROS POSITIVOS (TP)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        
        tprs.append(np.interp(mean_fpr, fpr, tpr))

# CALCULA A MÉDIA DA TAXA DE VERDADEIROS POSITIVOS, MÉDIA DA ÁREA SOB A CURVA (AUC) E DESVIO PADRÃO DA AUC   
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

# PLOT DA MÉDIA DA CURVA ROC PARA OS 3 MODELOS   
    plt.plot(mean_fpr, mean_tpr, label=f'{nome_modelo} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')


plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC Média para os Modelos')
plt.legend()
plt.grid()
plt.show()


# INFERENCIA DO VINHO TINTO COM MODELO SVM


data_red = pd.read_csv('winequalityN.csv')

# TRATAMENTO DOS DADOS PARA O VINHO TINTO
data_red = data_red.rename(columns={'quality': 'opinion'})
data_red['opinion'] = data_red['opinion'].apply(lambda r: 0 if r <= 5 else 1)
data_red = data_red[data_red['type'] == 'red']
data_red['type'] = data_red['type'].map({'red': 0})
data_red[nan_values] = data_red[nan_values].fillna(data_red[nan_values].mean())


x_red = data_red.iloc[:, :-1]
y_red = data_red.iloc[:, -1]


print("\nInferência nos Vinhos Tintos")


y_pred_red = pipelinesvm.predict(x_red)

print("\nInferência nos Vinhos Tintos (Usando o Melhor Modelo: SVM)")
print(classification_report(y_red, y_pred_red, target_names=['Ruim', 'Bom']))

num_bons = (y_pred_red == 1).sum()
num_ruins = (y_pred_red == 0).sum()
print(f"Total de Vinhos Bons: {num_bons}")
print(f"Total de Vinhos Ruins: {num_ruins}")


# MÉDIA CURVA ROC PARA O VINHO TINTO


y_probs_red = pipelinesvm.predict_proba(x_red)[:, 1]  
fpr_red, tpr_red, _ = roc_curve(y_red, y_probs_red)  
roc_auc_red = auc(fpr_red, tpr_red)  


plt.figure(figsize=(10, 7))
plt.plot(fpr_red, tpr_red, label=f'SVM - Vinhos Tintos (AUC = {roc_auc_red:.3f})', color='red')


plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC para o Modelo SVM nos Vinhos Tintos')
plt.legend()
plt.grid()
plt.show()

# TREINAMENTO FINAL COM VINHOS BRANCOS E TINTOS

print("\nTREINANDO O MODELO COM TODOS OS VINHOS (BRANCOS E TINTOS)")

# CARREGAMENTO DOS DADOS DOS DOIS TIPOS DE VINHOS
data_all = pd.read_csv('winequalityN.csv')

# RENOMEANDO QUALITY PARA OPINION E CONVERTENDO EM CLASSIFICAÇÃO BINÁRIA
data_all = data_all.rename(columns={'quality': 'opinion'})
data_all['opinion'] = data_all['opinion'].apply(lambda r: 0 if r <= 5 else 1)

# CONVERTENDO O TIPO DE VINHO EM VARIÁVEL NUMÉRICA (WHITE = 1, RED = 0)
data_all['type'] = data_all['type'].map({'white': 1, 'red': 0})

# TRATAMENTO DE VALORES NULOS
data_all[nan_values] = data_all[nan_values].fillna(data_all[nan_values].mean())

# SEPARAÇÃO DAS VARIÁVEIS X E Y
x_all = data_all.drop(columns=['opinion'])  # Inclui todas as features, incluindo 'type'
y_all = data_all['opinion']

# VALIDANDO NOVAMENTE O PIPELINE PARA OS DOIS VINHOS

# REGRESSÃO LOGÍSTICA
scoresLog_all = cross_validate(pipelineRg, x_all, y_all, cv=kf, scoring=scoring)
pipelineRg.fit(x_all, y_all)

# ÁRVORE DE DECISÃO
search.fit(x_all, y_all)
scoresTd_all = cross_validate(search.best_estimator_, x_all, y_all, cv=kf, scoring=scoring, n_jobs=-1)

# SVM
scoresSvm_all = cross_validate(pipelinesvm, x_all, y_all, cv=kf, scoring=scoring)
pipelinesvm.fit(x_all, y_all)

# EXIBINDO RESULTADOS PARA OS DOIS TIPOS DE VINHO
print("\nMÉTRICAS DO MODELO TREINADO COM TODOS OS VINHOS (BRANCOS E TINTOS)")

# REGRESSÃO LOGÍSTICA
print("\nREGRESSÃO LOGÍSTICA:")
for key, value in calcular_metricas(scoresLog_all).items():
    print(f'{key}: {value}')

# ÁRVORE DE DECISÃO
print("\nÁRVORE DE DECISÃO:")
for key, value in calcular_metricas(scoresTd_all).items():
    print(f'{key}: {value}')

# SVM
print("\nSVM:")
for key, value in calcular_metricas(scoresSvm_all).items():
    print(f'{key}: {value}')

# CONTAGEM DE PREVISÕES PARA TODOS OS VINHOS
y_pred_all_svm = pipelinesvm.predict(x_all)
num_bons_all = (y_pred_all_svm == 1).sum()
num_ruins_all = (y_pred_all_svm == 0).sum()
print(f"\nTotal de Vinhos Bons (Brancos e Tintos): {num_bons_all}")
print(f"Total de Vinhos Ruins (Brancos e Tintos): {num_ruins_all}")

# PLOTAGEM DA CURVA ROC PARA OS DOIS TIPOS DE VINHOS
plt.figure(figsize=(10, 7))

# CURVA ROC PARA O MODELO SVM TREINADO COM TODOS OS VINHOS
y_probs_all = pipelinesvm.predict_proba(x_all)[:, 1]
fpr_all, tpr_all, _ = roc_curve(y_all, y_probs_all)
roc_auc_all = auc(fpr_all, tpr_all)

plt.plot(fpr_all, tpr_all, label=f'SVM - Vinhos Brancos e Tintos (AUC = {roc_auc_all:.3f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC para o Modelo SVM com Todos os Vinhos')
plt.legend()
plt.grid()
plt.show()
