import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import warnings

# Ignorar warnings do numpy RNG
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Carregando os dados
df = pd.read_csv("fetal_health.csv")

# 2. Análise exploratória
sns.countplot(data=df, x="fetal_health")
plt.title("Distribuição das classes")
plt.show()

# 3. Separar features e target
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"].map({1.0: 0, 2.0: 1, 3.0: 2})

# 4. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transformar de volta em DataFrame para manter nomes das colunas
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# 6. Comparar modelos
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Acurácia média = {scores.mean():.4f}")

# 7. Treinar melhor modelo (XGBoost)
best_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 8. Otimização de hiperparâmetros (Random Forest)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy'
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nMelhor Random Forest: {grid_search.best_params_}")
print(f"Acurácia: {grid_search.best_score_:.4f}")

# 9. Explicabilidade com TreeExplainer (XGBoost)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled_df)

classes = ["Normal", "Suspeito", "Patológico"]

for i, class_name in enumerate(classes):
    print(f"SHAP para a classe: {class_name}")
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap.summary_plot(shap_values[:, :, i], X_test_scaled_df, feature_names=X.columns, plot_type="bar")
    else:
        shap.summary_plot(shap_values[i], X_test_scaled_df, feature_names=X.columns, plot_type="bar")
