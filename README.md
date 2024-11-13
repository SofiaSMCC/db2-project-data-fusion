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

Se guarda un bloque del índice invertido en un archivo binario después de ordenar los términos dentro del bloque. Utiliza `orjson` para la serialización del diccionario y lo guarda en un archivo.

```python
def guardar_bloque(bloque_indice, ruta, contador_bloque):
    sorted_terms = sorted(bloque_indice.keys())
    sorted_indice = {term: bloque_indice[term] for term in sorted_terms}
    with open(f"{ruta}/bloque_{contador_bloque}.bin", "wb") as f:
        f.write(orjson.dumps(sorted_indice))
```

Una vez que todos los bloques han sido guardados, se combinan (fusionan) en un solo índice invertido utilizando una estructura de datos de "heap" (montículo). Esto permite combinar los bloques de manera eficiente, siempre seleccionando el término más pequeño de cada bloque.

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

El algoritmo SPIMI (Single Pass In-Memory Indexing) es un método eficiente para construir un índice invertido en memoria secundaria, dividiendo el trabajo en bloques y luego fusionándolos.

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

1. Se crea el directorio para almacenar los bloques del índice invertido.
2. Se inicializa un contador para los bloques y se crea un índice invertido en memoria usando un defaultdict para almacenar términos y sus documentos.
3. Para cada documento en el conjunto de datos, se itera sobre las palabras del documento. Para cada palabra, se actualiza el índice invertido, añadiendo el documento y contando la frecuencia de la palabra en ese documento.
4. Cada vez que se alcanza el tamaño máximo de un bloque de documentos, se guarda el bloque en un archivo. Después, el índice invertido se limpia para empezar un nuevo bloque, y el contador de bloques se incrementa.
5. Al finalizar el procesamiento de todos los documentos, se verifica si hay algún bloque pendiente de guardar. Si es así, se guarda este último bloque.
6. Finalmente se fusionan todos los bloques en un índice invertido único y ordenado.

### 2. Ejecución óptima de consultas aplicando Similitud de Coseno
La similitud de coseno es una métrica matemática utilizada para medir la similitud entre dos vectores, que en este contexto representan los documentos o la consulta. Cuanto más alto sea el valor de similitud, más relevante es el documento con respecto a la consulta.

La función calcula la similitud de coseno entre dos vectores de características (en este caso, vectores de término ponderados).

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

La frecuencia de término (TF) mide cuántas veces aparece una palabra en un documento normalizado por su longitud. Esto es útil para saber la relevancia de una palabra en un documento específico.

```python
def calcular_tf(documento):
    tf = defaultdict(int)
    for palabra in documento:
        tf[palabra] += 1
    for palabra in tf:
        tf[palabra] /= len(documento)
    return tf
```

El IDF mide la importancia de una palabra en el conjunto de documentos. Las palabras que aparecen en muchos documentos tienen un IDF bajo, mientras que las palabras que aparecen en pocos documentos tienen un IDF alto.

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

El TF-IDF es el producto de la frecuencia de término (TF) y la frecuencia inversa de documentos (IDF). Este valor refleja la importancia de una palabra en un documento dado el contexto del conjunto completo de documentos.

```python
def calcular_tfidf(documento, idf):
    tf = calcular_tf(documento)
    tfidf = {palabra: tf[palabra] * idf[palabra] for palabra in documento if palabra in idf}
    return tfidf
```

Preprocesa el texto de la consulta, convirtiéndolo a minúsculas, tokenizándolo, eliminando las stop words y aplicando stemming si es necesario.

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

Recibe una consulta y busca documentos relevantes utilizando el índice invertido y el cálculo de similitud de coseno. Devuelve los top_k documentos más relevantes en función de la similitud.

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

1. La consulta que recibe la función (query) se somete a un proceso de preprocesamiento.
2. Se calcula el TF-IDF de las palabras procesadas en la consulta.
3. Se carga el índice invertido final desde un archivo binario que contiene el índice ya procesado y listo para ser utilizado en la consulta. El índice invertido es deserializado. 
4. Se calcula la similitud de coseno entre el vector TF-IDF de la consulta y los vectores TF-IDF de cada uno de los documentos del conjunto de datos.
5. Después de calcular la similitud de coseno para cada documento, se ordenan los resultados de mayor a menor similitud. Se seleccionan solo los top_k documentos más relevantes según la similitud.
6. Para cada uno de los documentos en los resultados ordenados, se busca la información de la canción correspondiente a doc_id en el dataframe df. Se obtiene el nombre de la canción y el artista(s) asociado(s) a ese documento (song_id).  
7. Finalmente, se imprime el nombre de la canción, los artistas y la similitud de coseno correspondiente.