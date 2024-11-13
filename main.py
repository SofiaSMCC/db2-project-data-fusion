import pandas as pd
import nltk
import math
import orjson
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import os
import time
import heapq

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

# Implementación del algoritmo SPIMI con generación de bloques
def spimi_construir_indice(dataset, bloque_tam=500, ruta="indice_invertido"):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

    N = len(dataset)  # Número total de documentos
    contador_bloque = 0  # Contador de bloques
    indice_invertido = defaultdict(dict)  # Diccionario temporal para el índice en memoria

    for i, (doc_id, palabras) in enumerate(dataset.items()):
        for palabra in palabras:
            if palabra not in indice_invertido:
                indice_invertido[palabra] = {doc_id: 1}
            else:
                if doc_id in indice_invertido[palabra]:
                    indice_invertido[palabra][doc_id] += 1
                else:
                    indice_invertido[palabra][doc_id] = 1

        # Cuando se llena el bloque, escribir a disco
        if (i + 1) % bloque_tam == 0:
            guardar_bloque(indice_invertido, ruta, contador_bloque)
            indice_invertido.clear()
            contador_bloque += 1

    # Guardar cualquier bloque restante
    if indice_invertido:
        guardar_bloque(indice_invertido, ruta, contador_bloque)

    # Mezcla de bloques al estilo BSBI
    mezclar_bloques_bsbi(ruta, contador_bloque + 1)

# Función para guardar el bloque en un archivo binario usando orjson
def guardar_bloque(bloque_indice, ruta, contador_bloque):
    sorted_terms = sorted(bloque_indice.keys())
    sorted_indice = {term: bloque_indice[term] for term in sorted_terms}
    with open(f"{ruta}/bloque_{contador_bloque}.bin", "wb") as f:  # "wb" para modo binario
        f.write(orjson.dumps(sorted_indice))

# Función para mezclar bloques utilizando BSBI en formato binario
def mezclar_bloques_bsbi(ruta, num_bloques):
    archivos = [open(f"{ruta}/bloque_{i}.bin", "rb") for i in range(num_bloques)]
    entradas = [orjson.loads(f.read()) for f in archivos]  # Lee en binario
    colas = [iter(sorted(entrada.items())) for entrada in entradas]
    
    # Usar un heap para realizar la mezcla ordenada de múltiples bloques
    heap = []
    for i, cola in enumerate(colas):
        try:
            term, postings = next(cola)
            heapq.heappush(heap, (term, i, postings))
        except StopIteration:
            pass

    with open(f"{ruta}/indice_final.bin", "wb") as f_out:  # Archivo final en formato binario
        indice_final = {}
        while heap:
            term, i, postings = heapq.heappop(heap)
            
            # Verifica si el término ya está en el índice final
            if term in indice_final:
                indice_final[term].update(postings)
            else:
                indice_final[term] = postings

            # Añade la siguiente entrada de la misma cola al heap
            try:
                next_term, next_postings = next(colas[i])
                heapq.heappush(heap, (next_term, i, next_postings))
            except StopIteration:
                pass

        f_out.write(orjson.dumps(indice_final))  # Guarda el índice final en binario

    for archivo in archivos:
        archivo.close()

# Función para realizar la consulta optimizada sobre el índice invertido en bloques
def buscar_letra(query, ruta="indice_invertido", top_k=5):
    query_words = preProcesamiento(query)
    resultados = defaultdict(float)

    # Cargar el índice final en binario
    with open(os.path.join(ruta, "indice_final.bin"), "rb") as f:
        indice_final = orjson.loads(f.read())  # Lee en binario
        for palabra in query_words:
            if palabra in indice_final:
                for doc_id, freq in indice_final[palabra].items():
                    resultados[doc_id] += freq

    return sorted(resultados.items(), key=lambda x: x[1], reverse=True)[:top_k]

# Función principal
def main():
    # Cargar el dataset y procesar las letras
    df = pd.read_csv("dataset.csv")
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
