import pandas as pd
import nltk
import math
import json
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import os

# Descargar recursos de NLTK necesarios
nltk.download('punkt')
stemmer = SnowballStemmer('spanish')

# Cargar la lista de stopwords
with open("utils/stoplist.txt", encoding="latin1") as file:
    stoplist = [line.rstrip().lower() for line in file]
stoplist += ['?', '-', '.', ':', ',', '!', ';']

# Función de preprocesamiento de texto
def preProcesamiento(texto, stemming=False):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z0-9_À-ÿ]', ' ', texto)
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(texto)
    words = [word for word in words if word not in stoplist]
    if stemming:
        words = [stemmer.stem(word) for word in words]
    return words

# Función para calcular TF (frecuencia de término)
def get_tf(word, doc):
    return doc.count(word) / len(doc) if doc else 0

# Función para calcular DF (frecuencia de documento) para cada palabra en el conjunto de documentos
def get_df(docs):
    df = {}
    for words in docs.values():
        unique_words = set(words)
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    return df

# Función para calcular el peso TF-IDF
def w_tf_idf(tf, df, N):
    return math.log10(1 + tf) * math.log10(N / df)

# Función para construir el índice invertido y guardar en disco
def construir_indice_invertido(dataset, bloque_tam=100, ruta="indice_invertido"):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

    indice_invertido = defaultdict(dict)
    df = get_df(dataset)  # Calcula la frecuencia de documentos
    N = len(dataset)  # Número total de documentos
    contador_bloque = 0  # Contador de bloques

    for i, (song_id, palabras) in enumerate(dataset.items()):
        for palabra in set(palabras):
            tf = get_tf(palabra, palabras)  # Calcula TF para la palabra en el documento
            tf_idf = w_tf_idf(tf, df[palabra], N)  # Calcula TF-IDF
            indice_invertido[palabra][song_id] = tf_idf

        # Guardar el índice en bloques en disco cuando alcanza el tamaño de bloque
        if (i + 1) % bloque_tam == 0:
            with open(f"{ruta}/bloque_{contador_bloque}.json", "w", encoding="utf-8") as f:
                json.dump(indice_invertido, f, ensure_ascii=False, indent=4)
            indice_invertido.clear()  # Limpia el índice para el siguiente bloque
            contador_bloque += 1

    # Guarda cualquier índice restante que no alcanzó el tamaño del bloque
    if indice_invertido:
        with open(f"{ruta}/bloque_{contador_bloque}.json", "w", encoding="utf-8") as f:
            json.dump(indice_invertido, f, ensure_ascii=False, indent=4)

# Función para realizar la consulta sobre el índice invertido
def buscar_letra(query, ruta="indice_invertido", top_k=5):
    query_words = preProcesamiento(query)
    resultados = defaultdict(float)
    
    # Carga y busca en cada bloque del índice
    for filename in os.listdir(ruta):
        if filename.endswith(".json"):
            with open(os.path.join(ruta, filename), "r", encoding="utf-8") as f:
                bloque = json.load(f)
                for palabra in query_words:
                    if palabra in bloque:
                        for doc_id, peso in bloque[palabra].items():
                            resultados[doc_id] += peso

    # Ordena los resultados por peso TF-IDF y devuelve los top_k resultados
    sorted_resultados = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    return sorted_resultados[:top_k]

# Función principal
def main():
    # Cargar el dataset y procesar las letras
    df = pd.read_csv("dataset.csv")  # Reemplaza "dataset.csv" por el nombre de tu archivo
    dataset = {row['song_id']: preProcesamiento(row['lyrics']) for _, row in df.iterrows()}

    # Construir el índice invertido en disco
    construir_indice_invertido(dataset)

    # Realizar una búsqueda de ejemplo
    query = "better than home"
    resultados = buscar_letra(query)
    print(f"Resultados para la consulta '{query}':")
    for song_id, score in resultados:
        song_info = df[df['song_id'] == song_id][['song', 'artists']].values[0]
        print(f"{song_info[0]} por {song_info[1]} - Puntaje: {score}")

if __name__ == "__main__":
    main()
