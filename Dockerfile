# Usa la imagen base de Python 3.12
FROM python:3.12.6

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY requirements.txt .
COPY app.py .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto que Streamlit usará
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py"]
