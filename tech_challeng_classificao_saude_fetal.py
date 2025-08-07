import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from imblearn.over_sampling import SMOTE
import shap

df = pd.read_csv('fetal_health.csv')
df.head()

# Tratamento dos dados nulos
print("Dados nulos:")
print(df.isnull().sum())

# Tratamento dos dados duplicados
print("Dados duplicados:")
print(df.duplicated().sum())
df.shape

# Remover dados duplicados
df = df.drop_duplicates()

print("Dados duplicados:")
print(df.duplicated().sum())

df.shape

# Renomeia a coluna fetal_health -> target
df = df.rename(columns={'fetal_health': 'target'})

df.head()

# Verificação da quantidade de linhas e colunas do dataset, e dos tipos de dados presente nele;
# Como não há nenhum dado string, não é necessário alterar para valores inteiros
print("Shape (linhas, colunas):", df.shape)
print("\nTipos de dados:\n", df.dtypes)

df.describe()

df.info()

# Distribuição da variável Target
df['target'].value_counts().plot(kind='bar')
plt.title("Distribuição da variável Target")
plt.xlabel("Classifição")
plt.ylabel("Quantidade")
plt.show()

# A partir da distribuição da variável target, é possível perceber que esse dataset é desbalenceado. Já que a maior parte dos dados são classificados como Normal.

correlation_matrix = df.corr()

plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlação entre variáveis')
plt.tight_layout()
plt.show()

# As variáveis com maior correlação
# - abnormal_short_term_variability:	0.62
# - mean_value_of_short_term_variability:	–0.53
# - prolongued_decelerations:	0.49
# - histogram_mean:	–0.39
# - histogram_median:	–0.36

# Selecionar as variáveis mais correlacionadas com 'target'
correlation_cols = [
    'abnormal_short_term_variability',
    'mean_value_of_short_term_variability',
    'prolongued_decelerations',
    'histogram_mean',
    'histogram_median',
    'target'
]

# Gerar o pairplot
sns.pairplot(df[correlation_cols], hue='target', diag_kind='hist')
plt.suptitle("Pairplot")
plt.tight_layout()
plt.show()

# Boxplot da 'abnormal_short_term_variability'
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='target', y='abnormal_short_term_variability')
plt.title("Boxplot")
plt.grid(True)
plt.show()

# Interpretação:
# Valores da abnormal_short_term_variability e a distribuição geral aumentam conforme o target vai de 1 → 2 → 3.
# Isso sugere que maior variabilidade anormal de curto prazo está associada a maior risco fetal, apesar de existir alguns outliers

# Boxplot da 'prolongued_decelerations'
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='target', y='prolongued_decelerations')
plt.title("Boxplot")
plt.grid(True)
plt.show()


# Gráfico de contagem de valores de 'prolongued_decelerations' por classe 'target'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='prolongued_decelerations', hue='target')
plt.title("Contagem de valores de prolongued_decelerations por target")
plt.grid(True)
plt.show()

# Interpretação Boxplot + Contagem
# A maioria dos valores é zero, especialmente nas classes 1 e 2. Já a classe 3 apresenta uma maior frequência de valores diferentes de zero.
# A classe 3 tende a apresentar maior incidência de desacelerações prolongadas, o que faz sentido do ponto de vista clínico: desacelerações prolongadas estão associadas a sofrimento fetal.

# Separar features e target
X = df.drop("target", axis=1)
y = df["target"]

# Pipeline de pré-processamento dos dados
# Explicação: Para evitar que atributos com valores grandes dominem a modelagem, foi usado o StandardScaler() para padronizar os dados com média 0 e desvio padrão 1.

x_coluns = X.columns.tolist()

# Pipeline de escalonamento
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), x_coluns)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Distribuição das calsses após a aplicação do SMOTE, é possível ter um dataset balanceado.
contagem = y_resampled.value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(contagem.index.astype(str), contagem.values)
plt.title('Distribuição das Classes após SMOTE')
plt.tight_layout()
plt.show()

# Pipeline final: transforma os dados com `transformer` e aplica o classificador

random_forest_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# Treina com os dados balanceados
random_forest_pipeline.fit(X_resampled, y_resampled)

#  Avalia predicao no conjunto de teste original
y_pred_random_forest = random_forest_pipeline.predict(X_test)

# Accuracy mede o percentual de classificações corretas sobre o total.
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
print("Acurácia com Árvore de decisão: \n", accuracy_random_forest)

# Recall (taxa de verdadeiro positivo) representa a média do recall para cada classe, importante quando há desequilíbrio.
recall_random_forest = recall_score(y_test, y_pred_random_forest, average='macro')
print("Taxa de verdadeiro positivo com Árvore de decisão: \n", recall_random_forest)

f1_random_forest = f1_score(y_test, y_pred_random_forest, average='macro')
report = classification_report(y_test, y_pred_random_forest, output_dict=True)
print("F1-score com Árvore de decisão: \n", f1_random_forest)


result_random_forest = permutation_importance(random_forest_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Cria dicionário com os nomes das colunas e suas importâncias
feature_importances_random_forest = dict(zip(X.columns, result_random_forest.importances_mean))

df_importance_random_forest = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result_random_forest.importances_mean
}).sort_values(by='Importance', ascending=False)

print(df_importance_random_forest.head(5))

# Selecionar apenas as 5 variáveis mais importantes
top5_random_forest = df_importance_random_forest.head(5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top5_random_forest["Feature"], top5_random_forest["Importance"])
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# SHAP

model = random_forest_pipeline.named_steps['classifier']
X_test_transformed = random_forest_pipeline.named_steps['preprocessor'].transform(X_test)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

plt.figure()
shap.summary_plot(
    shap_values[1] if isinstance(shap_values, list) else shap_values,
    features=X_test_transformed,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.show()

# Funcao para definir o K
error = []

for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_resampled , y_resampled)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# Pipeline final: transforma os dados com preprocessor e aplica o classificador

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# Treina com os dados balanceados
knn_pipeline.fit(X_resampled, y_resampled)

#  Avalia predicao no conjunto de teste original
y_pred_knn = knn_pipeline.predict(X_test)

# Accuracy mede o percentual de classificações corretas sobre o total.
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Acurácia com KNN: \n", accuracy_knn)

# Recall (taxa de verdadeiro positivo) representa a média do recall para cada classe, importante quando há desequilíbrio.
recall_knn = recall_score(y_test, y_pred_knn, average='macro')
print("Taxa de verdadeiro positivo com KNN: \n", recall_knn)

f1_knn = f1_score(y_test, y_pred_knn, average='macro')
print("F1-score com KNN: \n", f1_knn)


# SHAP

X_test_transformed_knn = knn_pipeline.named_steps['preprocessor'].transform(X_test)
model_knn = knn_pipeline.named_steps['classifier']
predict_fn = lambda x: model_knn.predict_proba(x)

background = shap.sample(X_test_transformed_knn, 100)
explainer_knn = shap.KernelExplainer(predict_fn, background)

shap_values_knn = explainer_knn.shap_values(X_test_transformed_knn, nsamples=100)

# Plot do gráfico SHAP
plt.figure()
shap.summary_plot(
    shap_values_knn[1] if isinstance(shap_values_knn, list) else shap_values_knn,
    features=X_test_transformed_knn,
    feature_names=X.columns,
    show=False
)
plt.tight_layout()
plt.show()

rf_report = classification_report(y_test, y_pred_random_forest)
knn_report = classification_report(y_test, y_pred_knn)

cm_rf = confusion_matrix(y_test, y_pred_random_forest)
cm_knn = confusion_matrix(y_test, y_pred_knn)
labels_confusion_matrix = ['Normal', 'Suspect', 'Pathological']

print("Random Forest: \n", rf_report)
print(f'\n  Confusion Matrix:')
plt.figure(figsize=(8,6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=labels_confusion_matrix, yticklabels=labels_confusion_matrix)
plt.xlabel('Class Predit')
plt.ylabel('Class Real')
plt.title('Confusion Matrix')
plt.show()

print()
print("KNN: \n", knn_report)
print(f'\n  Confusion Matrix:')
plt.figure(figsize=(8,6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=labels_confusion_matrix, yticklabels=labels_confusion_matrix)
plt.xlabel('Class Predit')
plt.ylabel('Class Real')
plt.title('Confusion Matrix')
plt.show()
