# Usar imagem oficial do Python 3.8 slim
FROM python:3.8-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar arquivos de requisitos e instalar dependências
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código para dentro do container
COPY . .

# Comando para rodar o script principal
CMD ["python", "main.py"]
