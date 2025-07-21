# Fetal Health Classification

Este projeto aplica técnicas de Machine Learning para prever o estado de saúde fetal com base em dados de cardiotocografia (CTG). O conjunto de dados é proveniente do [Kaggle - Fetal Health Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification).

---

## 📊 Objetivo

Classificar a saúde fetal em três categorias:

- **1.0**: Normal
- **2.0**: Suspeito
- **3.0**: Patológico

---

## 🧱 Estrutura do Projeto

```
fetal-health-classification/
├── fetal_health.csv
├── main.py                # Script principal com o pipeline completo
├── README.md              # Este arquivo
└── requirements.txt       # Dependências do projeto
```

---

## 🔍 Etapas do pipeline

1. Carregamento e visualização dos dados
2. Análise exploratória (EDA)
3. Pré-processamento (normalização, divisão treino/teste)
4. Comparação e treinamento de modelos (Random Forest, XGBoost, Logistic Regression)
5. Otimização de hiperparâmetros
6. Avaliação dos modelos (matriz de confusão, relatório de classificação)
7. Explicabilidade com SHAP (TreeExplainer)

---

## 🚀 Como executar

1. Clone o repositório:

```bash
git clone https://github.com/seuusuario/fetal-health-classification.git
cd fetal-health-classification
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o script principal:

```bash
python main.py
```

---

## 📚 Fonte dos Dados

- Kaggle: [Fetal Health Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

---

## 👨‍💻 Autor

Felipe Zanella

---

## 📌 Licença

Este projeto é apenas para fins educacionais. Dados disponíveis sob a licença do Kaggle.
