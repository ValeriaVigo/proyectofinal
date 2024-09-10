import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import sqlite3
import os
from sqlalchemy import create_engine

DATABASE_URL = 'sqlite:///librosfinal.db'

# Lista de palabras clave para detectar versiones alternativas
keywords = ["guide", "version", "edition", "abridged", "unabridged", "adaptation", "annotated", "bloom's", "summary"]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def has_common_keywords(title1, title2):
    title1 = title1.lower()
    title2 = title2.lower()
    
    for keyword in keywords:
        if keyword in title1 and keyword in title2:
            return True
    return False

def get_title_similarity(input_title, all_titles):
    similarities = [similar(input_title, title) for title in all_titles]
    return np.array(similarities)

def get_book_recommendations(input_book, df, numerical_features, num_recommendations=20):
    try:
        idx = df[df['Name'] == input_book].index[0]
    except IndexError:
        st.write(f"Libro '{input_book}' no encontrado en el DataFrame.")
        return pd.DataFrame(), []

    input_title = df['Name'].iloc[idx]
    title_similarities = get_title_similarity(input_title, df['Name'].values)
    
    input_features = numerical_features[idx].reshape(1, -1)
    numerical_similarities = cosine_similarity(input_features, numerical_features)[0]

    # Peso de la similitud de títulos y características numéricas
    total_similarity = (0.7 * title_similarities) + (0.3 * numerical_similarities)
    
    sim_scores = list(enumerate(total_similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Excluir el libro de entrada

    filtered_scores = []
    seen_titles = set()
    
    for i, score in sim_scores:
        candidate_title = df['Name'].iloc[i]
        base_title = candidate_title.split(":")[0].strip().lower()
        
        # Verificación mejorada para evitar recomendar títulos demasiado similares
        if base_title not in seen_titles and similar(base_title, input_title.lower()) < 0.8:  # Ajuste de similitud
            if score > 0.7:  # Ajusta el umbral de similitud si es necesario
                filtered_scores.append((i, score))
                seen_titles.add(base_title)
                if len(filtered_scores) == num_recommendations:
                    break
    
    book_indices = [i[0] for i in filtered_scores]
    recommendations = df[['Name', 'Authors']].iloc[book_indices]
    return recommendations, filtered_scores

def remove_duplicate_titles_global(recommendations, scores):
    seen_titles = set()
    unique_recommendations = []
    scores_dict = {}
    
    for _, row in recommendations.iterrows():
        title = row['Name']
        base_title = title.split(":")[0].strip().lower()
        if base_title not in seen_titles:
            seen_titles.add(base_title)
            unique_recommendations.append(row)
            index = recommendations.index.get_loc(row.name)
            scores_dict[base_title] = scores[index][1]
    
    unique_recommendations_df = pd.DataFrame(unique_recommendations)
    return unique_recommendations_df, scores_dict

@st.cache_data

def download_database():
    url = "https://storage.googleapis.com/recomendacion_libros/librosfinal.db"
    response = requests.get(url)
    if response.status_code == 200:
        with open('librosfinal.db', 'wb') as f:
            f.write(response.content)
        if os.path.exists('librosfinal.db'):
            st.write("Base de datos descargada con éxito.")
        else:
            st.write("Error: La base de datos no se descargó correctamente.")
    else:
        st.write(f"Error al descargar la base de datos. Código de estado: {response.status_code}")

def connect_to_database():
    download_database()
    engine = create_engine(DATABASE_URL)
    return engine

def load_data():
    engine = connect_to_database()
    query = "SELECT * FROM Librosfinal"
    df = pd.read_sql(query, engine)
    return df


@st.cache_data
def load_model():
    url_model = "https://storage.googleapis.com/recomendacion_libros/modelo_libros_entrenado.pkl"
    response_model = requests.get(url_model)
    model = joblib.load(BytesIO(response_model.content))
    return model

df = load_data()

# Título de la aplicación
st.title('Recomendador de Libros')
# Crear un campo de búsqueda para filtrar los títulos de los libros
search_query = st.text_input('Buscar libro:', '')
# Filtrar libros basados en la búsqueda
filtered_titles = df['Name'][df['Name'].str.contains(search_query, case=False, na=False)]
selected_book = st.selectbox('Selecciona un libro:', filtered_titles)

if selected_book:
    book_info = df[df['Name'] == selected_book]
    st.write(book_info)
    
    model = load_model()
    numerical_features = df[['pagesNumber', 'Rating']].values  
    
    if st.button('Obtener recomendaciones'):
        st.write(f'Recomendaciones para: {selected_book}')
        
        # Obtener recomendaciones
        recommendations, scores = get_book_recommendations(selected_book, df, numerical_features)
        
        # Filtrar duplicados globalmente
        unique_recommendations, scores_dict = remove_duplicate_titles_global(recommendations, scores)
        
        st.write('Recomendaciones:')
        for _, row in unique_recommendations.iterrows():
            base_title = row['Name'].split(":")[0].strip().lower()
            st.write(f"- Libro: {row['Name']}, Autor: {row['Authors']}, Puntuación de Similaridad: {scores_dict.get(base_title, 0):.4f}")
