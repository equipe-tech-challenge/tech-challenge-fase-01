import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import os
from preprocessamento_dados import verificar_e_processar_dados

class FetalHealthKNN:
    def __init__(self, n_vizinhos=3, metrica='euclidean', peso='uniform'):
        self.n_vizinhos = n_vizinhos
        self.metrica = metrica
        self.peso = peso

        self.modelo = None
        self.X_treino = None
        self.X_teste = None
        self.y_treino = None
        self.y_teste = None
        self.y_predicao = None
        self.metricas_desempenho = {}

        self._verificar_e_carregar_dados()

    def _verificar_e_carregar_dados(self):
        verificar_e_processar_dados()
        self._carregar_dados()

    def _carregar_dados(self):
        arquivo_treino = 'fetal_health_treino_smote.csv'
        arquivo_teste = 'fetal_health_teste.csv'

        if not os.path.exists(arquivo_treino) or not os.path.exists(arquivo_teste):
            preprocessador = verificar_e_processar_dados.PreprocessadorDados()
            preprocessador.executar_preprocessamento()

            if not os.path.exists(arquivo_treino) or not os.path.exists(arquivo_teste):
                raise FileNotFoundError("Erro ao gerar arquivos processados")

        treino = pd.read_csv(arquivo_treino)
        self.X_treino = treino.drop('target', axis=1)
        self.y_treino = treino['target']

        teste = pd.read_csv(arquivo_teste)
        self.X_teste = teste.drop('target', axis=1)
        self.y_teste = teste['target']

    def criar_modelo(self):
        self.modelo = KNeighborsClassifier(
            n_neighbors=self.n_vizinhos,
            metric=self.metrica,
            weights=self.peso
        )
        return self.modelo

    def treinar_modelo(self):
        if self.modelo is None:
            self.criar_modelo()

        self.modelo.fit(self.X_treino, self.y_treino)
        return self.modelo

    def realizar_predicao(self):
        if self.modelo is None:
            raise ValueError("Modelo não foi treinado. Execute treinar_modelo() primeiro.")

        self.y_predicao = self.modelo.predict(self.X_teste)
        return self.y_predicao

    def calcular_metricas(self):
        if self.y_predicao is None:
            self.realizar_predicao()

        self.metricas_desempenho = {
            'acuracia': accuracy_score(self.y_teste, self.y_predicao),
            'recall': recall_score(self.y_teste, self.y_predicao, average='macro'),
            'f1_score': f1_score(self.y_teste, self.y_predicao, average='macro')
        }

        return self.metricas_desempenho

    def obter_matriz_confusao(self):
        if self.y_predicao is None:
            self.realizar_predicao()

        return confusion_matrix(self.y_teste, self.y_predicao)

    def obter_relatorio_classificacao(self):
        if self.y_predicao is None:
            self.realizar_predicao()

        return classification_report(self.y_teste, self.y_predicao)

    def calcular_fitness(self, peso_acuracia=0.5, peso_f1=0.3, peso_recall=0.2, penalizar_complexidade=True):
        if not self.metricas_desempenho:
            self.calcular_metricas()

        fitness = (
            self.metricas_desempenho['acuracia'] * peso_acuracia +
            self.metricas_desempenho['f1_score'] * peso_f1 +
            self.metricas_desempenho['recall'] * peso_recall
        )

        # Penaliza modelos com muitos vizinhos (0-5% de redução)
        if penalizar_complexidade:
            penalidade = (self.n_vizinhos - 1) / 200  # máx 9.5% para k=20
            fitness *= (1 - penalidade)

        return fitness

    def calcular_fitness_ponderado(self, peso_acuracia=0.5, peso_f1=0.3, peso_recall=0.2):
        return self.calcular_fitness(peso_acuracia, peso_f1, peso_recall, penalizar_complexidade=False)

    def executar_pipeline_completo(self):
        self.treinar_modelo()
        self.realizar_predicao()

        metricas = self.calcular_metricas()
        matriz = self.obter_matriz_confusao()
        relatorio = self.obter_relatorio_classificacao()

        return {
            'metricas': metricas,
            'matriz_confusao': matriz,
            'relatorio_classificacao': relatorio,
            'fitness': self.calcular_fitness(),
            'parametros': {
                'n_vizinhos': self.n_vizinhos,
                'metrica': self.metrica,
                'peso': self.peso
            }
        }

    def exibir_resultados(self):
        """
        Summary:
            Exibe os resultados do modelo de forma formatada.
        """
        resultados = self.executar_pipeline_completo()

        print(f"Acurácia: {resultados['metricas']['acuracia']:.4f}, Recall: {resultados['metricas']['recall']:.4f}, F1-Score: {resultados['metricas']['f1_score']:.4f}")

        return resultados

    def atualizar_parametros(self, n_vizinhos=None, metrica=None, peso=None):
        if n_vizinhos is not None:
            self.n_vizinhos = n_vizinhos

        if metrica is not None:
            self.metrica = metrica

        if peso is not None:
            self.peso = peso

        self.criar_modelo()


if __name__ == "__main__":
    modelo_knn = FetalHealthKNN(n_vizinhos=3, metrica='minkowski', peso='uniform')
    modelo_knn.exibir_resultados()

    configuracoes = [
        {'n_vizinhos': 4, 'metrica': 'manhattan', 'peso': 'distance'},
    ]

    for i, config in enumerate(configuracoes, 1):
        modelo = FetalHealthKNN(**config)
        modelo.treinar_modelo()
        fitness = modelo.calcular_fitness()
