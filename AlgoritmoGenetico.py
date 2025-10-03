import random
import numpy as np
from fetal_health_KNN import FetalHealthKNN


class AlgoritmoGenetico:
    def __init__(self, tamanho_populacao=100, taxa_mutacao=0.7, taxa_crossover=0.8,
                 geracoes=500, elitismo=2):
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.geracoes = geracoes
        self.elitismo = elitismo

        self.populacao = []
        self.melhor_individuo = None
        self.melhor_fitness = 0
        self.historico_fitness = []

        self.ranges = {
            'n_vizinhos': (1, 20),
            'metrica': ['euclidean', 'manhattan', 'minkowski'],
            'peso': ['uniform', 'distance']
        }

    def gerar_individuo(self):
        return {
            'n_vizinhos': random.randint(*self.ranges['n_vizinhos']),
            'metrica': random.choice(self.ranges['metrica']),
            'peso': random.choice(self.ranges['peso'])
        }

    def inicializar_populacao(self):
        self.populacao = [self.gerar_individuo() for _ in range(self.tamanho_populacao)]

    def fitness(self, individuo):
        modelo = FetalHealthKNN(
            n_vizinhos=individuo['n_vizinhos'],
            metrica=individuo['metrica'],
            peso=individuo['peso']
        )
        modelo.treinar_modelo()
        return modelo.calcular_fitness(penalizar_complexidade=True)

    def avaliar_populacao(self):
        fitness_scores = []
        for individuo in self.populacao:
            score = self.fitness(individuo)
            fitness_scores.append((individuo, score))

        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        if fitness_scores[0][1] > self.melhor_fitness:
            self.melhor_individuo = fitness_scores[0][0]
            self.melhor_fitness = fitness_scores[0][1]

        return fitness_scores

    def selecao_torneio(self, fitness_scores, k=3):
        torneio = random.sample(fitness_scores, k)
        torneio.sort(key=lambda x: x[1], reverse=True)
        return torneio[0][0]

    def crossover(self, pai1, pai2):
        if random.random() > self.taxa_crossover:
            return pai1.copy(), pai2.copy()

        filho1 = {}
        filho2 = {}

        for gene in pai1.keys():
            if random.random() < 0.5:
                filho1[gene] = pai1[gene]
                filho2[gene] = pai2[gene]
            else:
                filho1[gene] = pai2[gene]
                filho2[gene] = pai1[gene]

        return filho1, filho2

    def mutacao(self, individuo):
        individuo_mutado = individuo.copy()
        for gene in individuo_mutado.keys():
            if random.random() < self.taxa_mutacao: 
                if gene == 'n_vizinhos':
                    individuo_mutado[gene] = random.randint(*self.ranges['n_vizinhos'])
                else:
                    individuo_mutado[gene] = random.choice(self.ranges[gene])
        return individuo_mutado

    def evoluir(self):
        self.inicializar_populacao()

        for geracao in range(self.geracoes):
            fitness_scores = self.avaliar_populacao()

            self.historico_fitness.append({
                'geracao': geracao + 1,
                'melhor_fitness': fitness_scores[0][1],
                'fitness_medio': np.mean([score for _, score in fitness_scores]),
                'melhor_individuo': fitness_scores[0][0]
            })

            print(f"Geração {geracao + 1}/{self.geracoes} - "
                  f"Melhor Fitness: {fitness_scores[0][1]:.4f} - "
                  f"Fitness Médio: {self.historico_fitness[-1]['fitness_medio']:.4f}")

            nova_populacao = []

            elite = [ind for ind, _ in fitness_scores[:self.elitismo]]
            nova_populacao.extend(elite)

            while len(nova_populacao) < self.tamanho_populacao:
                pai1 = self.selecao_torneio(fitness_scores)
                pai2 = self.selecao_torneio(fitness_scores)

                filho1, filho2 = self.crossover(pai1, pai2)

                filho1 = self.mutacao(filho1)
                filho2 = self.mutacao(filho2)

                nova_populacao.append(filho1)
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(filho2)

            self.populacao = nova_populacao

        return self.melhor_individuo, self.melhor_fitness

    def exibir_resultados(self):
        print("\n" + "=" * 60)
        print("RESULTADOS DO ALGORITMO GENÉTICO")
        print("=" * 60)
        print(f"\nMelhor Indivíduo Encontrado:")
        print(f"  Número de vizinhos: {self.melhor_individuo['n_vizinhos']}")
        print(f"  Métrica: {self.melhor_individuo['metrica']}")
        print(f"  Peso: {self.melhor_individuo['peso']}")
        print(f"\nMelhor Fitness (Acurácia): {self.melhor_fitness:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    print("Executando Algoritmo Genético para Otimização de Parâmetros KNN\n")

    ag = AlgoritmoGenetico(
        tamanho_populacao=20,
        taxa_mutacao=0.5,
        taxa_crossover=0.8,
        geracoes=50,
        elitismo=2
    )

    melhor_individuo, melhor_fitness = ag.evoluir()
    ag.exibir_resultados()
