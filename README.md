# Data Fusion

## Introducción

### Objetivo del Proyecto

Desarrollar un sistema integral de base de datos multimedia que optimice la búsqueda y recuperación de información, implementando un índice invertido para documentos de texto y una estructura multidimensional para la búsqueda eficiente de imágenes, audio y otros objetos multimedia.

### Descripción del dataset

Se utilizó una base de datos de Kaggle ([dataset](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset)) que contiene 8,673 registros de canciones junto con su información asociada. Cada entrada incluye atributos como `song_id` (ID de la canción), `artist_id` (ID del artista), `song` (nombre de la canción), `artists` (nombres de los artistas), `explicit` (indicador de contenido explícito), `genres` (géneros musicales) y `lyrics` (letras de las canciones). Este conjunto de datos ofrece una amplia variedad de características relacionadas con las canciones y sus artistas, lo que lo convierte en una valiosa fuente para análisis musicales y tareas de recuperación de información.

### Importancia de Aplicar Indexación

La indexación es esencial para mejorar la eficiencia y velocidad en la recuperación de información, tanto en bases de datos relacionales como en sistemas de búsqueda. Los índices permiten organizar los datos de manera estructurada, facilitando consultas rápidas, especialmente en grandes volúmenes de información. Además de los índices tradicionales, los índices multidimensionales son fundamentales para manejar datos complejos, como imágenes o audio, ya que permiten realizar búsquedas eficientes en espacios con múltiples características o dimensiones. En conjunto, estos métodos optimizan el rendimiento, reducen los tiempos de respuesta y mejoran la escalabilidad de los sistemas.

## Backend

## Índice Invertido

### 1. Construcción del índice invertido en memoria secundaria

```python
def guardar_bloque(bloque_indice, ruta, contador_bloque):
    sorted_terms = sorted(bloque_indice.keys())
    sorted_indice = {term: bloque_indice[term] for term in sorted_terms}
    with open(f"{ruta}/bloque_{contador_bloque}.bin", "wb") as f:
        f.write(orjson.dumps(sorted_indice))
```

```python
def mezclar_bloques_bsbi(ruta, num_bloques):
    archivos = [open(f"{ruta}/bloque_{i}.bin", "rb") for i in range(num_bloques)]
    entradas = [orjson.loads(f.read()) for f in archivos] 
    colas = [iter(sorted(entrada.items())) for entrada in entradas]
    
    heap = []
    for i, cola in enumerate(colas):
        try:
            term, postings = next(cola)
            heapq.heappush(heap, (term, i, postings))
        except StopIteration:
            pass

    with open(f"{ruta}/indice_final.bin", "wb") as f_out: 
        indice_final = {}
        while heap:
            term, i, postings = heapq.heappop(heap)
            
            if term in indice_final:
                indice_final[term].update(postings)
            else:
                indice_final[term] = postings

            try:
                next_term, next_postings = next(colas[i])
                heapq.heappush(heap, (next_term, i, next_postings))
            except StopIteration:
                pass

        f_out.write(orjson.dumps(indice_final)) 

    for archivo in archivos:
        archivo.close()
```

```python
def spimi_construir_indice(dataset, bloque_tam=500, ruta="indice_invertido"):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

    contador_bloque = 0  
    indice_invertido = defaultdict(dict) 

    for i, (doc_id, palabras) in enumerate(dataset.items()):
        for palabra in palabras:
            if palabra not in indice_invertido:
                indice_invertido[palabra] = {doc_id: 1}
            else:
                if doc_id in indice_invertido[palabra]:
                    indice_invertido[palabra][doc_id] += 1
                else:
                    indice_invertido[palabra][doc_id] = 1

        if (i + 1) % bloque_tam == 0:
            guardar_bloque(indice_invertido, ruta, contador_bloque)
            indice_invertido.clear()
            contador_bloque += 1

    if indice_invertido:
        guardar_bloque(indice_invertido, ruta, contador_bloque)

    mezclar_bloques_bsbi(ruta, contador_bloque + 1)
```

### 2. Ejecución óptima de consultas aplicando Similitud de Coseno

```python
def similitud_coseno(vec1, vec2):
    producto_punto = sum(vec1[palabra] * vec2.get(palabra, 0) for palabra in vec1)
    
    norma_vec1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    norma_vec2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    if norma_vec1 == 0 or norma_vec2 == 0:
        return 0.0
    else:
        return producto_punto / (norma_vec1 * norma_vec2)
```

### 3. Procesamiento de Consulta

**Funciones Auxiliares**

```python
def calcular_tf(documento):
    tf = defaultdict(int)
    for palabra in documento:
        tf[palabra] += 1
    for palabra in tf:
        tf[palabra] /= len(documento)
    return tf
```

```python
def calcular_idf(dataset):
    idf = defaultdict(float)
    total_docs = len(dataset)
    for doc_id, documento in dataset.items():
        palabras_unicas = set(documento)
        for palabra in palabras_unicas:
            idf[palabra] += 1
    for palabra in idf:
        idf[palabra] = math.log(total_docs / idf[palabra])
    return idf
```

```python
def calcular_tfidf(documento, idf):
    tf = calcular_tf(documento)
    tfidf = {palabra: tf[palabra] * idf[palabra] for palabra in documento if palabra in idf}
    return tfidf
```

```python
def preProcesamiento(texto, stemming=False):
    texto = texto.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(texto)
    words = [word for word in words if word not in stoplist]
    if stemming:
        words = [stemmer.stem(word) for word in words]
    return words
```

**Consulta**

```python
def buscar_letra(query, dataset, idf, df, ruta="indice_invertido", top_k=5):
    query_words = preProcesamiento(query)
    query_tfidf = calcular_tfidf(query_words, idf)

    with open(os.path.join(ruta, "indice_final.bin"), "rb") as f:
        indice_final = orjson.loads(f.read())

    similitudes = []
    for doc_id, palabras in dataset.items():
        doc_tfidf = calcular_tfidf(palabras, idf)
        similitud = similitud_coseno(query_tfidf, doc_tfidf)
        similitudes.append((doc_id, similitud))

    resultados_ordenados = sorted(similitudes, key=lambda x: x[1], reverse=True)[:top_k]
    
    for doc_id, score in resultados_ordenados:
        song_info = df[df['song_id'] == doc_id][['song', 'artists']].values[0]
        print(f"{song_info[0]} por {song_info[1]} - Similitud de Coseno: {score:.4f}")
```


