# Imagem base
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos para o container
COPY tech_challeng_classificao_saude_fetal.py /app/
COPY requirements.txt /app/
COPY fetal_health.csv /app/

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["python", "tech_challeng_classificao_saude_fetal.py"]