"""
Módulo responsável pelo pré-processamento dos dados de saúde fetal.
Aplica StandardScaler, divide em treino/teste e aplica SMOTE apenas no treino.
Gera arquivos separados conforme arquivo original.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os


class PreprocessadorDados:
    """Classe para tratamento e normalização dos dados de saúde fetal."""

    def __init__(self, arquivo_entrada='fetal_health.csv',
                 tamanho_teste=0.2,
                 semente_aleatoria=42):
        """
        Summary:
            Inicializa o preprocessador com configurações de arquivos e seed.

        Args:
            arquivo_entrada: Caminho do arquivo CSV original
            tamanho_teste: Proporção para teste (padrão 0.2 = 20%)
            semente_aleatoria: Seed para reprodutibilidade
        """
        self.arquivo_entrada = arquivo_entrada
        self.tamanho_teste = tamanho_teste
        self.semente_aleatoria = semente_aleatoria
        self.scaler = StandardScaler()

    def carregar_dados(self):
        """
        Summary:
            Carrega o dataset original do arquivo CSV.

        Returns:
            DataFrame com os dados carregados
        """
        if not os.path.exists(self.arquivo_entrada):
            raise FileNotFoundError(f"Arquivo {self.arquivo_entrada} não encontrado")

        dados = pd.read_csv(self.arquivo_entrada)
        print(f"Dados carregados: {dados.shape[0]} linhas, {dados.shape[1]} colunas")
        return dados

    def verificar_dados_nulos(self, dados):
        """
        Summary:
            Exibe informações sobre valores nulos no dataset.

        Args:
            dados: DataFrame com os dados
        """
        print("Dados nulos:")
        print(dados.isnull().sum())

    def remover_duplicados(self, dados):
        """
        Summary:
            Remove registros duplicados do dataset.

        Args:
            dados: DataFrame com os dados

        Returns:
            DataFrame sem duplicados
        """
        print(f"\nDados duplicados: {dados.duplicated().sum()}")

        if dados.duplicated().sum() > 0:
            dados = dados.drop_duplicates()
            print(f"Dados duplicados removidos. Shape atual: {dados.shape}")

        return dados

    def renomear_colunas(self, dados):
        """
        Summary:
            Renomeia a coluna fetal_health para target conforme arquivo original.

        Args:
            dados: DataFrame com os dados

        Returns:
            DataFrame com coluna renomeada
        """
        dados = dados.rename(columns={'fetal_health': 'target'})
        print("Coluna 'fetal_health' renomeada para 'target'")
        return dados

    def aplicar_standard_scaler(self, dados):
        """
        Summary:
            Aplica StandardScaler para normalizar features antes da divisão.

        Args:
            dados: DataFrame com os dados

        Returns:
            DataFrame com features normalizadas
        """
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
        """
        Summary:
            Divide dados em treino (80%) e teste (20%) com estratificação.

        Args:
            dados: DataFrame com os dados

        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
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
        """
        Summary:
            Aplica SMOTE apenas nos dados de treino para balanceamento.

        Args:
            X_train: Features de treino
            y_train: Target de treino

        Returns:
            Tupla (X_resampled, y_resampled)
        """
        print("\nDistribuição antes do SMOTE (treino):")
        print(y_train.value_counts().sort_index())

        smote = SMOTE(random_state=self.semente_aleatoria)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print("\nDistribuição após SMOTE (treino):")
        print(y_resampled.value_counts().sort_index())

        return X_resampled, y_resampled

    def salvar_dados_processados(self, X_train_smote, y_train_smote, X_test, y_test):
        """
        Summary:
            Salva dados de treino (com SMOTE) e teste em arquivos separados.

        Args:
            X_train_smote: Features de treino com SMOTE
            y_train_smote: Target de treino com SMOTE
            X_test: Features de teste (original)
            y_test: Target de teste (original)
        """
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
        """
        Summary:
            Executa pipeline completo conforme arquivo original.

        Returns:
            Dicionário com dados processados
        """
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
    """
    Summary:
        Valida existência dos arquivos processados ou executa preprocessamento.

    Returns:
        True se os arquivos existem ou foram criados com sucesso
    """
    arquivos_necessarios = [
        'fetal_health_treino_smote.csv',
        'fetal_health_teste.csv'
    ]

    if all(os.path.exists(arq) for arq in arquivos_necessarios):
        print("Arquivos processados encontrados. Pulando pré-processamento.")
        return True

    print("Arquivos processados não encontrados. Executando pré-processamento...")
    preprocessador = PreprocessadorDados()
    preprocessador.executar_preprocessamento()

    return all(os.path.exists(arq) for arq in arquivos_necessarios)


if __name__ == "__main__":
    preprocessador = PreprocessadorDados()
    preprocessador.executar_preprocessamento()
