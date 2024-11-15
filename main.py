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
stemmer = SnowballStemmer('english')

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

# Función para calcular TF
def calcular_tf(documento):
    tf = defaultdict(int)
    for palabra in documento:
        tf[palabra] += 1
    for palabra in tf:
        tf[palabra] /= len(documento)  # Normalización por la longitud del documento
    return tf

# Función para calcular IDF
def calcular_idf(dataset):
    idf = defaultdict(float)
    total_docs = len(dataset)
    for doc_id, documento in dataset.items():
        palabras_unicas = set(documento)
        for palabra in palabras_unicas:
            idf[palabra] += 1
    for palabra in idf:
        idf[palabra] = math.log(total_docs / idf[palabra])  # IDF = log(N / df)
    return idf

# Función para calcular el vector TF-IDF de un documento
def calcular_tfidf(documento, idf):
    tf = calcular_tf(documento)
    tfidf = {palabra: tf[palabra] * idf[palabra] for palabra in documento if palabra in idf}
    return tfidf

# Función para calcular la similitud de coseno
def similitud_coseno(vec1, vec2):
    # Producto punto de vec1 y vec2
    producto_punto = sum(vec1[palabra] * vec2.get(palabra, 0) for palabra in vec1)
    
    # Normas de los vectores
    norma_vec1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    norma_vec2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    # Similaridad de coseno
    if norma_vec1 == 0 or norma_vec2 == 0:
        return 0.0
    else:
        return producto_punto / (norma_vec1 * norma_vec2)

# Implementación del algoritmo SPIMI con generación de bloques
def spimi_construir_indice(dataset, bloque_tam=500, ruta="indice_invertido"):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    
    if os.path.exists(f"{ruta}/indice_final.bin"):
        print("Índice invertido ya construido.")
        return

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
    with open(f"{ruta}/bloque_{contador_bloque}.bin", "wb") as f:
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

# Función para realizar la consulta optimizada usando similitud de coseno
def buscar_letra(query, dataset, idf, df, ruta="indice_invertido", top_k=5):
    query_words = preProcesamiento(query)
    query_tfidf = calcular_tfidf(query_words, idf)

    # Cargar el índice final en binario
    with open(os.path.join(ruta, "indice_final.bin"), "rb") as f:
        indice_final = orjson.loads(f.read())  # Lee en binario

    # Calcular la similitud de coseno entre la consulta y cada documento
    similitudes = []
    for doc_id, palabras in dataset.items():
        doc_tfidf = calcular_tfidf(palabras, idf)
        similitud = similitud_coseno(query_tfidf, doc_tfidf)
        similitudes.append((doc_id, similitud))

    # Ordenar los documentos por similitud y devolver los top_k
    resultados_ordenados = sorted(similitudes, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Imprimir resultados de la consulta
    formattedRes = []
    for doc_id, score in resultados_ordenados:
        song_info = df[df['song_id'] == doc_id].values[0]
        data = {
            "song": song_info[3],
            "artist": eval(song_info[4])[0],
            "genre": ", ".join(song_info[6].split(";")[:3]),
            "score": round(score, 4)
        }
        formattedRes.append(data)
    
    return formattedRes

def setupIndex():
    global _df, _dataset, _idf
    
    _df = pd.read_csv("dataset.csv")
    _dataset = {row['song_id']: preProcesamiento(row['lyrics']) for _, row in _df.iterrows()}
    _idf = calcular_idf(_dataset)

    spimi_construir_indice(_dataset)

def querySearch(query, top_k=5):
    global _df, _dataset, _idf
    return buscar_letra(query, _dataset, _idf, _df, top_k=top_k)

# Función principal
def main():
    # Cargar el dataset y procesar las letras
    df = pd.read_csv("dataset.csv")
    dataset = {row['song_id']: preProcesamiento(row['lyrics']) for _, row in df.iterrows()}

    # Calcular IDF para todo el dataset
    idf = calcular_idf(dataset)

    # Medir tiempo de construcción del índice invertido
    start_time_indexing = time.time()
    spimi_construir_indice(dataset)
    end_time_indexing = time.time()
    print(f"Tiempo de construcción del índice invertido: {end_time_indexing - start_time_indexing:.2f} segundos")

    # Realizar y medir el tiempo de una búsqueda de ejemplo
    query = "Hello darkness my old friend"
    start_time_search = time.time()
    buscar_letra(query, dataset, idf, df)
    end_time_search = time.time()
    print(f"Tiempo de consulta de búsqueda: {end_time_search - start_time_search:.2f} segundos")

if __name__ == "__main__":
    main()
