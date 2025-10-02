import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os


class PreprocessadorDados:
    def __init__(self, arquivo_entrada='fetal_health.csv',
                 tamanho_teste=0.2,
                 semente_aleatoria=42):
        self.arquivo_entrada = arquivo_entrada
        self.tamanho_teste = tamanho_teste
        self.semente_aleatoria = semente_aleatoria
        self.scaler = StandardScaler()

    def carregar_dados(self):
        if not os.path.exists(self.arquivo_entrada):
            raise FileNotFoundError(f"Arquivo {self.arquivo_entrada} não encontrado")

        dados = pd.read_csv(self.arquivo_entrada)
        print(f"Dados carregados: {dados.shape[0]} linhas, {dados.shape[1]} colunas")
        return dados

    def verificar_dados_nulos(self, dados):
        print("Dados nulos:")
        print(dados.isnull().sum())

    def remover_duplicados(self, dados):
        print(f"\nDados duplicados: {dados.duplicated().sum()}")

        if dados.duplicated().sum() > 0:
            dados = dados.drop_duplicates()
            print(f"Dados duplicados removidos. Shape atual: {dados.shape}")

        return dados

    def renomear_colunas(self, dados):
        dados = dados.rename(columns={'fetal_health': 'target'})
        print("Coluna 'fetal_health' renomeada para 'target'")
        return dados

    def aplicar_standard_scaler(self, dados):
        X = dados.drop('target', axis=1)
        y = dados['target']

        colunas_features = X.columns.tolist()

        X_normalizado = self.scaler.fit_transform(X)

        dados_normalizados = pd.DataFrame(
            X_normalizado,
            columns=colunas_features,
            index=dados.index
        )
        dados_normalizados['target'] = y.values

        print(f"\nStandardScaler aplicado em {len(colunas_features)} features")
        return dados_normalizados

    def dividir_dados(self, dados):
        X = dados.drop('target', axis=1)
        y = dados['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=self.tamanho_teste,
            random_state=self.semente_aleatoria
        )

        print(f"\nDivisão dos dados:")
        print(f"  Treino: {X_train.shape[0]} amostras ({(1-self.tamanho_teste)*100:.0f}%)")
        print(f"  Teste: {X_test.shape[0]} amostras ({self.tamanho_teste*100:.0f}%)")

        return X_train, X_test, y_train, y_test

    def aplicar_smote(self, X_train, y_train):
        print("\nDistribuição antes do SMOTE (treino):")
        print(y_train.value_counts().sort_index())

        smote = SMOTE(random_state=self.semente_aleatoria)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print("\nDistribuição após SMOTE (treino):")
        print(y_resampled.value_counts().sort_index())

        return X_resampled, y_resampled

    def salvar_dados_processados(self, X_train_smote, y_train_smote, X_test, y_test):
        treino = X_train_smote.copy()
        treino['target'] = y_train_smote.values
        treino.to_csv('fetal_health_treino_smote.csv', index=False)

        teste = X_test.copy()
        teste['target'] = y_test.values
        teste.to_csv('fetal_health_teste.csv', index=False)

        print(f"\nArquivos salvos:")
        print(f"  Treino (com SMOTE): fetal_health_treino_smote.csv - {treino.shape}")
        print(f"  Teste (original): fetal_health_teste.csv - {teste.shape}")

    def executar_preprocessamento(self):
        print("=" * 60)
        print("Iniciando pré-processamento dos dados")
        print("=" * 60)

        dados = self.carregar_dados()
        self.verificar_dados_nulos(dados)
        dados = self.remover_duplicados(dados)
        dados = self.renomear_colunas(dados)
        dados = self.aplicar_standard_scaler(dados)

        X_train, X_test, y_train, y_test = self.dividir_dados(dados)
        X_train_smote, y_train_smote = self.aplicar_smote(X_train, y_train)

        self.salvar_dados_processados(X_train_smote, y_train_smote, X_test, y_test)

        print("=" * 60)
        print("Pré-processamento concluído com sucesso!")
        print("=" * 60)

        return {
            'X_train_smote': X_train_smote,
            'y_train_smote': y_train_smote,
            'X_test': X_test,
            'y_test': y_test
        }


def verificar_e_processar_dados():
    arquivos_necessarios = [
        'fetal_health_treino_smote.csv',
        'fetal_health_teste.csv'
    ]

    if all(os.path.exists(arq) for arq in arquivos_necessarios):
        return True

    print("Arquivos processados não encontrados. Executando pré-processamento...")
    preprocessador = PreprocessadorDados()
    preprocessador.executar_preprocessamento()

    return all(os.path.exists(arq) for arq in arquivos_necessarios)


if __name__ == "__main__":
    preprocessador = PreprocessadorDados()
    preprocessador.executar_preprocessamento()
