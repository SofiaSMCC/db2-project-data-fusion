# Data Fusion


## Introducción

### Objetivo del Proyecto

Desarrollar un sistema integral de base de datos multimedia que optimice la búsqueda y recuperación de información, implementando un índice invertido para documentos de texto y una estructura multidimensional para la búsqueda eficiente de imágenes, audio y otros objetos multimedia.

### Descripción del dataset

Se utilizó una base de datos de Kaggle ([dataset](https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs?resource=download)) que contiene 18,000 registros de canciones junto con su información asociada. Cada entrada incluye atributos como `track_id` (ID de la canción), `track_name` (nombre de la canción), `track_artist` (nombres del artista), `track_popularity` (indicador de popularidad), `playlist_genre` (géneros musicales) y `lyrics` (letras de las canciones). Este conjunto de datos ofrece una amplia variedad de características relacionadas con las canciones y sus artistas, lo que lo convierte en una valiosa fuente para análisis musicales y tareas de recuperación de información.

### Importancia de Aplicar Indexación

La indexación es esencial para mejorar la eficiencia y velocidad en la recuperación de información, tanto en bases de datos relacionales como en sistemas de búsqueda. Los índices permiten organizar los datos de manera estructurada, facilitando consultas rápidas, especialmente en grandes volúmenes de información. Además de los índices tradicionales, los índices multidimensionales son fundamentales para manejar datos complejos, como imágenes o audio, ya que permiten realizar búsquedas eficientes en espacios con múltiples características o dimensiones. En conjunto, estos métodos optimizan el rendimiento, reducen los tiempos de respuesta y mejoran la escalabilidad de los sistemas.

## Backend

> ## Proyecto 2: Indice Invertido Textual

### 1. Construcción del índice invertido en memoria secundaria

El algoritmo SPIMI (Single Pass In-Memory Indexing) es un método eficiente para construir un índice invertido en memoria secundaria, dividiendo el trabajo en bloques y luego fusionándolos.

```python
def spimi_invert(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        block_count = 0
        dictionary = defaultdict(dict)

        # Usar paralelización para distribuir el procesamiento de documentos
        for i, (doc_id, words) in enumerate(self.dataset.items()):
            for word in words:
                if word not in dictionary:
                    dictionary[word] = { doc_id: 1 }
                else:
                    if doc_id in dictionary[word]:
                        dictionary[word][doc_id] += 1
                    else:
                        dictionary[word][doc_id] = 1

            # Guardar el bloque cuando el límite se alcanza
            if (i + 1) % self.block_limit == 0:
                self.save_temp_block(dictionary, block_count)
                dictionary.clear()
                block_count += 1

        self.total_blocks = block_count
        if dictionary:
            self.save_temp_block(dictionary, block_count)

        self.merge_all_blocks()
```

1. Se crea el directorio para almacenar los bloques del índice invertido.
2. Se inicializa un contador para los bloques y se crea un índice invertido temporal en memoria usando un defaultdict para almacenar términos y sus documentos.
3. Para cada documento en el conjunto de datos, se itera sobre las palabras del documento. Para cada palabra, se actualiza el índice invertido local, añadiendo el documento y contando la frecuencia de la palabra en ese documento.
4. Una vez que se alcanza el tamaño máximo de un bloque de documentos, se guarda el bloque en un archivo. Después, el diccionaro local (índice invertido) se limpia para empezar un nuevo bloque, y el contador de bloques se incrementa.
5. Al finalizar el procesamiento de todos los documentos, se verifica si hay algún bloque pendiente de guardar. Si es así, se guarda este último bloque.
6. Finalmente se fusionan todos los bloques creados mediante la función `merge_all_blocks()`, resultando en una serie de bloques ordenados por la llave (palabra) distribuidos de forma uniforme.

```python
def merge_all_blocks(self):
        """Fusionar todos los bloques de índice invertido en múltiples pasadas."""
        levels = math.ceil(math.log2(self.total_blocks))
        level = 1
        
        while level <= levels:
            step = 2 ** level
            for i in range(0, self.total_blocks, step):
                start = i
                finish = min(i + step - 1, self.total_blocks)
                self.merge_blocks(start, finish)
            level += 1
```

1. Se calcula la cantidad de niveles necesarios para realizar el merge de todos los bloques.
2. Mientras el nivel actual no exceda el total de niveles, se realiza el merge de una cantidad específica de bloques. (Ej. En el primer nivel se realiza merge de los bloques en pares, en el segundo nivel se realiza merge de los bloques en grupos de 4 y así sucesivamente).
3. Se utiliza la función `merge_blocks(start, finish)`, que tiene como parámetro el indice del primer bloque del grupo que se va a unir y el indice del último bloque.

```python
def merge_blocks(self, start, finish):
        """Fusionar bloques de índice invertido."""
        dictionary = defaultdict(dict)
        
        for i in range(start, min(finish + 1, self.total_blocks)):
            with open(f"{self.path}/temp_block_{i}.bin", "rb") as file:
                data = pickle.load(file)
                for word, postings in data.items():
                    dictionary[word].update(postings)
        
        sorted_dict = sorted(dictionary.keys())  # Ordenamos las claves
        
        total_elements = len(sorted_dict)
        num_blocks = finish - start + 1
        block_size, remainder = divmod(total_elements, num_blocks)
        
        temp_dict = {}
        current_block_size = block_size + (1 if remainder > 0 else 0)
        remainder -= 1
        block_count = 0
        
        for word in sorted_dict:
            temp_dict[word] = dictionary[word]
            
            if len(temp_dict) == current_block_size:
                self.save_block(temp_dict, start + block_count)
                temp_dict = {}
                block_count += 1
                
                if remainder > 0:
                    current_block_size = block_size + 1
                    remainder -= 1
                else:
                    current_block_size = block_size
        
        if temp_dict:
            self.save_block(temp_dict, start + block_count)
```

1. Se crea un indice invertido local donde se van a almacenar las palabras de los bloques que se van a fusionar y sus postings.
2. Se abre cada bloque temporal con permisos de lectura y se añaden las palabras al diccionario.
3. Se ordena el diccionario por el valor de la llave.
4. Finalmente, se desea guardar el indice invertido en la misma cantidad de bloques leidos, por lo que se calcula una división de los datos lo más uniforme posible.

### 2. Procesamiento de Consulta

Recibe una consulta y busca documentos relevantes utilizando el índice invertido y el cálculo de similitud de coseno. Devuelve los top_k documentos más relevantes en función de la similitud.

```python
def query_search(self, query, top_k=5):
        """Realizar búsqueda por consulta usando TF-IDF."""
        query_words = self.pre_processing(query)

        term_freq = defaultdict(int)
        weights = defaultdict(float)

        for word in query_words:
            term_freq[word] += 1

        query_pow2_len = 0
        docs_pow2_lens = defaultdict(float)

        for block in range(self.total_blocks):
            with open(f"{self.path}/block_{block}.bin", "rb") as file:
                data = pickle.load(file)

                for word, postings in data.items():
                    idf = math.log10(self.total_docs / len(postings))
                    query_tf_idf = math.log10(1 + term_freq[word]) * idf
                    query_pow2_len += query_tf_idf ** 2

                    for doc_id, tf in postings.items():
                        docs_pow2_lens[doc_id] += (math.log10(1 + tf) * idf) ** 2
                        weights[doc_id] += query_tf_idf * math.log10(1 + tf) * idf
            
        for i in weights:
            if (query_pow2_len > 0 and weights[i] > 0):
                weights[i] /= (math.sqrt(query_pow2_len) * math.sqrt(docs_pow2_lens[i]))
            
        results = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_k]

        formatted_res = []
        for doc_id, score in results:
            song_info = self.data[self.data['track_id'] == doc_id].values[0]
            data = {
                "song_id": song_info[0],
                "song": song_info[1],
                "artist": song_info[2],
                "genre": song_info[10],
                "score": round(score, 4),
                "lyrics": song_info[3]
            }
            formatted_res.append(data)
        
        return formatted_res
```
1. Se Pre-procesa la query (Limpiando cada palabra y convirtiéndolas en tokens) y retorna una lista con las palabras de la consulta.
2. Por cada palabra en la query se calcula su TF y se almacena en un diccionario.
3. Para facilitar el cálculo de la norma se crean las variables `query_pow2_len` y `docs_pow2_lens` donde se va a almacenar la norma cuadrada de la query los documentos.
4. Para cada bloque final, se abre el archivo y se carga su índice invertido. Para cada término se calcula su IDF, el peso TF-IDF para los términos de la query, se aumentan las normas y se actualiza el peso (score) acumulado de cada término.
5. Se calcula la similitud coseno normalizando los pesos (score) para cada documento haciendo uso de las normas acumuladas anteriormente.
6. Finalmente, se ordenan los datos por el score y se retorna los K resultados con mayor score.

### Experimento

Para realizar el experimento de realizar una búsqueda textual entre nuestro Índice Invertido y PostgreSQL, se utilizaron distintos tamaños del Dataset, como se visualiza en la siguiente tabla:

|           | MyIndex        | PostgreSQL |
|-----------| -------------- | ---------- |
| N = 500   | 12.60 ms      | 48.56 ms  |
| N = 1000  | 41.80 ms      | 77.26 ms  |
| N = 2000  | 65.10 ms      | 135.74 ms  |
| N = 4000  | 107.80 ms      | 251.167 ms  |
| N = 8000  | 187.90 ms      | 486.58 ms  |
| N = 10000  | 219.50 ms      | 615.16 ms  |
| N = 14000  | 262.90 ms      | 850.45 ms  |
| N = 18000  | 373.70 ms      | 584.05 ms  |

Para poder comparar los resultados, se realizó una gráfica a partir de los datos obtenidos que será analizada más adelante:

<img src="graph_comparison.png" alt="drawing" width="800"/>

### Análisis de los resultados

La búsqueda textual en PostgreSQL se realizó con el siguiente comando, donde `%s` se reemplaza por la query separada por el operador `| (OR)`.

```sql
SELECT track_id, track_name, track_artist, playlist_genre, lyrics, ts_rank(to_tsvector('english', lyrics), to_tsquery('english', %s)) AS score
    FROM songs
    WHERE to_tsvector('english', lyrics) @@ to_tsquery('english', %s)
    ORDER BY score DESC
    LIMIT 10;
```

A continuación se detallará el funcionamiento de cada función especial de PostgreSQL para la búsqueda textual:
- `to_tsvector`: Esta función es equivalente a la función de pre-procesamiento de los documentos. Su propósito es normalizar las palabras eliminando StopWords, carácteres especiales y tokenizando las palabras para devolver finalmente un vector de tokens junto a la posición donde aparecen en el documento.
- `to_tsquery`: Esta función normaliza y tokeniza una consulta textual, donde cada palabra está separado por un operador booleano `& (AND)`, `| (OR)` y `! (NOT)`.
- `ts_rank`: Esta función calcula un puntaje de relevancia basado en la cantidad de coincidencias de la consulta (`to_tsquery('english', %s)`) con en el documento (`to_tsvector('english', lyrics)`).

Analizando el gráfico del tiempo de ejecución, se puede observar que para todas las consultas el tiempo del índice implementado por nosotros fue menor al de PostgreSQL. Como se explica en la [documentación de PostgreSQL](https://www.postgresql.org/docs/current/textsearch-controls.html#TEXTSEARCH-RANKING), el proceso de ranking es muy costoso debido a que debe consultar el tsvector de cada documento que coincida con la query, lo que vuelve el proceso más lento.

## Backend

> ## Proyecto 3: Indice Multidimensional

## Extracción de Características
Este proyecto utiliza un modelo ResNet50 preentrenado para extraer vectores de características de imágenes.

### 1. Configuración
- **Carga del Modelo ResNet50:**
  - Se utiliza un modelo ResNet50 preentrenado como extractor de características. El modelo se ajusta para eliminar la capa final (clasificadora), conservando solo las capas convolucionales que generan los vectores de características.

  ```python
  def load_resnet_feature_extractor():
      resnet50 = models.resnet50(pretrained=True).eval()
      feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
      return feature_extractor
  ```

- **Transformaciones de las Imágenes:**
  - Se define un conjunto de transformaciones para preprocesar las imágenes antes de pasarlas por el modelo.
    - Redimensionar a 224x224 (dimensiones requeridas por ResNet50).
    - Normalizar los valores de píxeles utilizando los valores promedio y desviación estándar del conjunto ImageNet.

  ```python
  def get_transform():
      return transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
          )
      ])
  ```

  ### 2. Extracción de Características

  - Se recorre una carpeta con imágenes y se aplica el proceso de extracción a cada archivo.

  ```python
  def extract_features_from_folder(folder_path, feature_extractor, transform):
      image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]
      features = [extract_features(image_path, feature_extractor, transform) for image_path in image_paths]
      return image_paths, features
  ```
- **Para cada imagen dentro de la carpeta:**
  - Se abre la imagen, se aplica la transformación definida y se extrae su vector de características utilizando el modelo ResNet50.

  ```python
  def extract_features(image_path, feature_extractor, transform):
      image = Image.open(image_path).convert('RGB')
      input_tensor = transform(image).unsqueeze(0)
      with torch.no_grad():
          features = feature_extractor(input_tensor)
      return features.squeeze().numpy()
  ```

## KNN Sequencial

### 1. Implementación de Búsqueda KNN
Realiza la búsqueda de los k vecinos más cercanos usando una cola de prioridad (heap).

```python
def knn_priority_queue(data_features, query_feature, k):

    heap = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, idx))
        else:
            heapq.heappushpop(heap, (-dist, idx))

    return [(abs(dist), idx) for dist, idx in sorted(heap)]
```

1. Se crea una lista vacía heap para almacenar los vecinos más cercanos.
2. Para cada vector de características en data_features, se calcula la distancia Euclidiana entre ese vector y el vector de la imagen de consulta (query_feature).
3. Si la cola tiene menos de k elementos, se agrega la tupla (-dist, idx) donde dist es la distancia negativa (para que el heap lo ordene de mayor a menor distancia) y idx es el índice de la imagen.
4. Caso Contrario,  se agrega el nuevo vecino, pero primero se elimina el vecino más distante (es decir, el que tiene la mayor distancia).
5. Se devuelven los k vecinos más cercanos en orden de distancia, convirtiendo las distancias negativas a positivas y ordenando el heap por las distancias.


### Procesamiento de Consulta

Se realiza la consulta para encontrar las k imágenes más similares a una imagen en un conjunto de datos, utilizando el algoritmo KNN sin indexación. 

```python
def Run_KnnSequential(query_image_path='poke2/00000001.jpg', k=5):

    feature_extractor = load_resnet_feature_extractor()
    transform = get_transform()

    feature_file = 'image_features.npz'

    if os.path.exists(feature_file):
        print("Cargando características desde el archivo guardado...")
        data_features, image_paths = load_features(feature_file)
    else:
        print("Extrayendo características de las imágenes...")
        folder_path = 'poke2'
        image_paths, data_features = extract_features_from_folder(folder_path, feature_extractor, transform)

        # Guardar características para usos futuros
        save_features(feature_file, data_features, image_paths)
        print(f"Características guardadas en {feature_file}")

    query_feature = extract_features(query_image_path, feature_extractor, transform)

    if len(data_features) > 1:
        pca = PCA(n_components=min(100, len(data_features[0])))
        data_features_reduced, pca_model = reduce_dimensions(data_features, n_components=pca.n_components)
        query_feature_reduced = reduce_single_feature(query_feature, pca_model)
    else:
        data_features_reduced = data_features
        query_feature_reduced = query_feature

    # KNN sin indexación
    knn_results = knn_priority_queue(data_features_reduced, query_feature_reduced, k)
    print("\nKNN (sin R-Tree):")
    for dist, idx in knn_results:
        print(f"- {image_paths[idx]} (Distancia: {dist:.4f})")

    return [(float(dist), image_paths[idx]) for dist, idx in knn_results]
```

1. Inicializa el extractor de características y las transformaciones para las imágenes. Si existen características, las carga; si no, extrae las características de todas las imágenes.
2. Procesa la imagen de consulta para obtener su vector de características.
2. Si hay múltiples características, aplica PCA para reducir las dimensiones de las características y la imagen de consulta.
3. Calcula las distancias entre la imagen de consulta y las demás imágenes para encontrar las k imágenes más cercanas y mostrar los resultados.

### 2. Implementación de Búsqueda por Rango

La búsqueda por rango se utiliza para encontrar todos los puntos dentro de un radio específico desde un punto de consulta. En lugar de buscar un número fijo de vecinos, se define un rango y se encuentran todos los puntos que están dentro de ese rango.

```python
def range_search(data_features, query_feature, radius):
    neighbors = []
    
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if dist <= radius:
            neighbors.append((dist, idx))
    neighbors.sort(key=lambda x: x[0])
    
    return neighbors
```
1. Se crea una lista vacía llamada neighbors para almacenar los vecinos que están dentro del radio especificado.
2. Para cada vector de características (data_features), se calcula la distancia Euclidiana entre el vector de la imagen de consulta y el vector actual.
3. Si la distancia calculada es menor o igual al radio, se agrega el vecino a la lista neighbors como una tupla (dist, idx), donde dist es la distancia y idx es el índice de la imagen.
4. Después de recopilar todos los vecinos dentro del radio, se ordenan en función de la distancia.
5. Se devuelve la lista neighbors ordenada por distancia.

### Procesamiento de Consulta

```python
def Run_RangeSearch(query_image_path='poke2/00000001.jpg', radius=0.5):

    feature_extractor = load_resnet_feature_extractor()
    transform = get_transform()

    feature_file = 'image_features.npz'

    if os.path.exists(feature_file):
        print("Cargando características desde el archivo guardado...")
        data_features, image_paths = load_features(feature_file)
    else:
        print("Extrayendo características de las imágenes...")
        folder_path = 'poke2'
        image_paths, data_features = extract_features_from_folder(folder_path, feature_extractor, transform)

        # Guardar características para usos futuros
        save_features(feature_file, data_features, image_paths)
        print(f"Características guardadas en {feature_file}")

    query_feature = extract_features(query_image_path, feature_extractor, transform)

    if len(data_features) > 1:
        pca = PCA(n_components=min(100, len(data_features[0])))
        data_features_reduced, pca_model = reduce_dimensions(data_features, n_components=pca.n_components)
        query_feature_reduced = reduce_single_feature(query_feature, pca_model)
    else:
        data_features_reduced = data_features
        query_feature_reduced = query_feature

    # Búsqueda por rango sin R-tree
    range_results = range_search(data_features_reduced, query_feature_reduced, radius)
    print("\nBúsqueda por rango (sin R-Tree):")
    for dist, idx in range_results:
        print(f"- {image_paths[idx]} (Distancia: {dist:.4f})")

    return [(float(dist), image_paths[idx]) for dist, idx in range_results]
```

1. Inicializa el extractor de características y las transformaciones para las imágenes. Si existen características, las carga; si no, extrae las características de todas las imágenes.
2. Procesa la imagen de consulta para obtener su vector de características.
3. Si hay múltiples características, aplica PCA para reducir las dimensiones de las características y la imagen de consulta.
4. Realiza la búsqueda por rango. Encuentra todas las imágenes dentro de un radio de distancia de la imagen de consulta.
5. Devuelve las imágenes encontradas junto con su distancia

## KNN RTree

### Implementación

### Consulta

## Análisis de la Maldición de Dimensionalidad

- A medida que el número de dimensiones de los datos aumenta, las métricas de distancia (como la distancia Euclidiana) pierden su eficacia. Esto se debe a que, en dimensiones altas, la mayoría de los puntos en el espacio están muy alejados unos de otros, lo que hace que las distancias entre ellos se vuelvan más similares, incluso si son puntos muy diferentes. Esto puede causar que los algoritmos de búsqueda y clasificación sean mucho más lentos y menos precisos, ya que se vuelve difícil diferenciar entre puntos cercanos y lejanos.
- El R-tree es eficiente en espacios de baja dimensión, pero su rendimiento disminuye en alta dimensión debido al aumento de la superposición de nodos y la homogenización de las distancias entre puntos, lo que provoca más búsquedas innecesarias.
- Para mitigar este problema se utilizó la solución de LSH (Locality Sensitive Hashing), que reduce la dimensionalidad preservando la proximidad entre puntos, utilizando la librería faiss.

### KNN HighD

### Implementación con faiss

Este código implementa la clase knnHighD_LSH, que utiliza Locality Sensitive Hashing (LSH) con la librería Faiss para realizar búsquedas eficientes de los k vecinos más cercanos (KNN) en conjuntos de datos de alta dimensión. El objetivo de esta implementación es mejorar el rendimiento de las búsquedas KNN en espacios de características de alta dimensión, que pueden volverse ineficientes utilizando métodos tradicionales, como el índice RTree. LSH permite hacer búsquedas aproximadas rápidas y precisas al reducir la dimensionalidad de los datos en un espacio de hashes.

```python
class knnHighD_LSH:
    def __init__(self, dimension, num_bits=512):
        self.dimension = dimension
        self.num_bits = num_bits
        self.idx = faiss.IndexLSH(dimension, num_bits)

    def insert(self, features):
        self.idx.add(features)

    def knn_search(self, query_vector, k=5):
        distances, indices = self.idx.search(query_vector, k)
        return distances, indices
```

### Inicialización

Inicializa la clase configurando la dimensión de los vectores de características que se van a indexar, así como el número de bits utilizados para la codificación LSH (por defecto, 512). Utiliza el índice LSH de Faiss, que permite la búsqueda eficiente en grandes volúmenes de datos de alta dimensión.

### Inserción

Inserta los vectores de características en el índice LSH. Esto prepara el índice para las búsquedas KNN.

### Búsqueda KNN

Permite realizar una búsqueda de los k vectores más cercanos a un query_vector dado. Al llamar al método search() de Faiss, devuelve las distancias y los índices de los vectores más cercanos en el índice. El valor preterminado de k significa que se devolverá los k vecinos más cercanos.

### Consulta

```python
def Run_KnnLSH(query_image_path='poke2/00000001.jpg', k=5):

    feature_extractor = load_resnet_feature_extractor()
    transform = get_transform()

    feature_file = 'image_features.npz'

    if os.path.exists(feature_file):
        print("Cargando características desde el archivo guardado...")
        data_features, image_paths = load_features(feature_file)
    else:
        print("Extrayendo características de las imágenes...")
        folder_path = 'poke2'
        image_paths, data_features = extract_features_from_folder(folder_path, feature_extractor, transform)

        # Guardar características para usos futuros
        save_features(feature_file, data_features, image_paths)
        print(f"Características guardadas en {feature_file}")

    query_feature = extract_features(query_image_path, feature_extractor, transform)

    if len(data_features) > 1:
        pca = PCA(n_components=min(100, len(data_features[0])))
        data_features_reduced, pca_model = reduce_dimensions(data_features, n_components=pca.n_components)
        query_feature_reduced = reduce_single_feature(query_feature, pca_model)
    else:
        data_features_reduced = data_features
        query_feature_reduced = query_feature

    # Búsqueda KNN con Faiss
    knn_faiss = knnHighD_LSH(dimension=data_features_reduced.shape[1], num_bits=512)
    knn_faiss.insert(data_features_reduced)

    distances, indices = knn_faiss.knn_search(query_feature_reduced.reshape(1, -1), k)
    print("\nKNN con FAISS:")
    for dist, idx in zip(distances[0], indices[0]):
        print(f"- {image_paths[idx]} (Distancia: {dist:.4f})")

    return [(float(dist), image_paths[idx]) for dist, idx in zip(distances[0], indices[0])]

```

1. Inicializa el extractor de características y las transformaciones para las imágenes. Si existen características, las carga; si no, extrae las características de todas las imágenes.
2. Procesa la imagen de consulta para obtener su vector de características.
3. Si hay múltiples características, aplica PCA para reducir las dimensiones de las características y la imagen de consulta.
4. Se crea una instancia de la clase knnHighD_LSH, que es responsable de realizar la búsqueda KNN utilizando Locality Sensitive Hashing (LSH) a través de Faiss. Se insertan las características de las imágenes y se realiza la búsqueda de los k vecinos más cercanos a las características de la imagen de consulta.
5. La función devuelve una lista con las distancias y las rutas de las imágenes, ordenadas según los resultados de la búsqueda.

### Experimento

|           | Rango Secuencial | KNN Secuencial | KNN RTree  | KNN HighD |
|-----------|------------------|----------------|------------|-----------|
| N = 500   | 1.0815 ms        | 1.9994 ms      | 1.2739 ms  | 0 ms      |
| N = 1000  | 4.0681 ms        | 4.0588 ms      | 3.0615 ms  | 0 ms      |
| N = 2000  | 8.0481 ms        | 8.1513 ms      | 5.0008 ms  | 0 ms      |
| N = 5000  | 17.7212 ms       | 20.9551 ms     | 8.65638 ms | 0 ms      |
| N = 10000 |                  |                |            |           |
| N = 15000 |                  |                |            |           |
| N = 20000 |                  |                |            |           |


### Análisis de Resultados

## Integrantes

|                              **Luciano Aguirre**                               |                                           **Abigail**                                           |                                         **Adrian**                                          |                                      **Jose**                                      |                           **Sofia Salazar**                            |
|:------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|
| <img src="https://github.com/lajesfen.png?size=100" width="100" height="100" style="border-radius: 50%;"> | <img src="https://github.com/abigail777sq.png?size=100" width="100" height="100" style="border-radius: 50%;"> | <img src="https://github.com/Auky216.png?size=100" width="100" height="100" style="border-radius: 50%;"> | <img src="https://github.com/JoseBarrenechea.png?size=100" width="100" height="100" style="border-radius: 50%;"> | <img src="https://github.com/SofiaSMCC.png?size=100" width="100" height="100" style="border-radius: 50%;"> |
|      <a href="https://github.com/lajesfen" target="_blank">`lajesfen`</a>      | <a href="https://github.com/abigail777sq" target="_blank">`abigail777sq`</a>                   | <a href="https://github.com/Auky216" target="_blank">`Auky216`</a>                        | <a href="https://github.com/JoseBarrenechea" target="_blank">`JoseBarrenechea`</a> | <a href="https://github.com/SofiaSMCC" target="_blank">`SofiaSMCC`</a> |

