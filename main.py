import pandas as pd
import nltk
import math
import json
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import os
import time  # Para medir el tiempo de ejecución

# Descargar recursos de NLTK necesarios
nltk.download('punkt')
stemmer = SnowballStemmer('spanish')

# Cargar la lista de stopwords
with open("utils/stoplist.txt", encoding="latin1") as file:
    stoplist = set(line.rstrip().lower() for line in file)
stoplist.update(['?', '-', '.', ':', ',', '!', ';'])

# Función de preprocesamiento optimizada
def preProcesamiento(texto, stemming=False):
    texto = texto.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(texto)
    words = [word for word in words if word not in stoplist]
    if stemming:
        words = [stemmer.stem(word) for word in words]
    return words

# Función para calcular TF-IDF
def w_tf_idf(tf, df, N):
    return math.log10(1 + tf) * math.log10(N / df)

# Implementación del algoritmo SPIMI
def spimi_construir_indice(dataset, bloque_tam=100, ruta="indice_invertido"):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

    # Variables iniciales
    N = len(dataset)  # Número total de documentos
    contador_bloque = 0  # Contador de bloques
    indice_invertido = defaultdict(dict)  # Diccionario temporal para el índice en memoria

    # Procesar cada documento (token_stream) uno por uno
    for i, (doc_id, palabras) in enumerate(dataset.items()):
        for palabra in palabras:
            if palabra not in indice_invertido:
                # Si el término no está en el diccionario, añadirlo
                indice_invertido[palabra] = {doc_id: 1}  # Inicializa la lista de postings
            else:
                # Si el término ya está, incrementa la frecuencia
                if doc_id in indice_invertido[palabra]:
                    indice_invertido[palabra][doc_id] += 1
                else:
                    indice_invertido[palabra][doc_id] = 1

        # Cuando se llena el bloque, escribir a disco
        if (i + 1) % bloque_tam == 0:
            guardar_bloque(indice_invertido, ruta, contador_bloque)
            indice_invertido.clear()  # Limpiar el diccionario para el siguiente bloque
            contador_bloque += 1

    # Guardar cualquier bloque restante
    if indice_invertido:
        guardar_bloque(indice_invertido, ruta, contador_bloque)

# Función para guardar el bloque en un archivo JSON
def guardar_bloque(bloque_indice, ruta, contador_bloque):
    sorted_terms = sorted(bloque_indice.keys())  # Ordenar términos
    sorted_indice = {term: bloque_indice[term] for term in sorted_terms}

    with open(f"{ruta}/bloque_{contador_bloque}.json", "w", encoding="utf-8") as f:
        json.dump(sorted_indice, f, ensure_ascii=False, indent=4)

# Función para realizar la consulta optimizada sobre el índice invertido en bloques
def buscar_letra(query, ruta="indice_invertido", top_k=5):
    query_words = preProcesamiento(query)
    resultados = defaultdict(float)

    # Cargar cada bloque una vez
    for filename in os.listdir(ruta):
        if filename.endswith(".json"):
            with open(os.path.join(ruta, filename), "r", encoding="utf-8") as f:
                bloque = json.load(f)
                for palabra in query_words:
                    if palabra in bloque:
                        for doc_id, freq in bloque[palabra].items():
                            resultados[doc_id] += freq

    # Ordena los resultados por frecuencia (simula el peso) y devuelve los top_k resultados
    return sorted(resultados.items(), key=lambda x: x[1], reverse=True)[:top_k]

# Función principal
def main():
    # Cargar el dataset y procesar las letras
    df = pd.read_csv("dataset.csv")  # Reemplaza "dataset.csv" por el nombre de tu archivo
    dataset = {row['song_id']: preProcesamiento(row['lyrics']) for _, row in df.iterrows()}

    # Medir tiempo de construcción del índice invertido
    start_time_indexing = time.time()
    spimi_construir_indice(dataset)
    end_time_indexing = time.time()
    print(f"Tiempo de construcción del índice invertido: {end_time_indexing - start_time_indexing:.2f} segundos")

    # Realizar y medir el tiempo de una búsqueda de ejemplo
    query = "I look at you look at me I can tell you only see some type of end Fuckin' with me now  actin' like"
    start_time_search = time.time()
    resultados = buscar_letra(query)
    end_time_search = time.time()
    print(f"Tiempo de consulta de búsqueda: {end_time_search - start_time_search:.2f} segundos")

    # Imprimir resultados de la consulta
    print(f"Resultados para la consulta '{query}':")
    for song_id, score in resultados:
        song_info = df[df['song_id'] == song_id][['song', 'artists']].values[0]
        print(f"{song_info[0]} por {song_info[1]} - Frecuencia: {score}")

if __name__ == "__main__":
    main()
