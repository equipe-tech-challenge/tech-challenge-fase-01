# Tech Challenge - Classificação de Saúde Fetal

Este projeto tem como objetivo desenvolver um modelo de Machine Learning para classificar a saúde fetal com base em sinais extraídos de exames cardiotocográficos. A classificação pode ajudar na triagem de casos e suporte à decisão clínica, sempre lembrando que a palavra final deve ser do médico responsável.

Link do dataset: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

## Dados

Este conjunto de dados contém 2.126 registros de características extraídas de exames de cardiotocograma, que foram então classificados por três obstetras especialistas em três categorias:

**Descrição dos Atributos (Colunas):**

1. Baseline value – frequência cardíaca basal (batimentos por minuto).
2. Accelerations – acelerações por segundo
3. Fetal movement – movimentos fetais por segundo
4. Uterine contractions – contrações uterinas por segundo
5. Light decelerations – desacelerações leves por segundo
6. Severe decelerations – desacelerações severas por segundo
7. Prolonged decelerations – desacelerações prolongadas por segundo
8. Abnormal short‑term variability – % de tempo com variabilidade de curto prazo anormal
9. Mean value of short‑term variability – valor médio da variabilidade de curto prazo
10. % abnormal long‑term variability – % de tempo com variabilidade de longo prazo anormal
11. Mean value of long‑term variability – valor médio da variabilidade de longo prazo
12. Histogram width – largura do histograma de FHR
13. Histogram min – valor mínimo no histograma de FHR
14. Histogram max – valor máximo no histograma de FHR
15. Histogram number of peaks – número de picos no histograma
16. Histogram number of zeros – número de zeros no histograma
17. Histogram mode – valor modal do histograma
18. Histogram mean – média do histograma
19. Histogram median – mediana do histograma
20. Histogram variance – variância do histograma
21. Histogram tendency – tendência (skewness ou direção) do histograma de FHR
22. Fetal health – label da classe: 1.0 → Normal | 2.0 → Suspect | 3.0 → Pathologica

## Estrutura do Projeto

- `tech_challeng_classificao_saude_fetal.ipynb`: Notebook principal contendo:
  - Análise exploratória dos dados
  - Pré-processamento (normalização, balanceamento)
  - Treinamento de modelos (Random Forest, KNN)
  - Avaliação dos resultados
  - Interpretação dos modelos com SHAP e feature importance

## Libs Utilizadas

- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP
- SMOTE (para balanceamento)

## Modelos Avaliados

- **Random Forest**
- **KNN**

## Como executar

1. Crie o ambiente virtual: ``python -m venv venv``
2. Ative o ambiente vitual: ``venv/Scripts/activate`` após rodar o comando será possivel ver (venv) na linha de comando terminal.
3. Instale as dependencias: ``pip install -r requirements.txt``
4. Rode o código: ``python tech_challeng_classificao_saude_fetal.py``

## Usando o Docker
1. Construa a imagem Docker: ``docker build -t fetal-health-app .``
2. Rode o container: ``docker run --rm fetal-health-app``

## Grupo:

1.  Bruna Cardoso Andrade - RM366295
2.  Felipe de Siqueira Zanella - RM365834
3.  Vinicius de Souza Medeiros -
4.  Horacy Lopes da Silva Junior - RM365525
5.  Gabriel Luiz Santana -
