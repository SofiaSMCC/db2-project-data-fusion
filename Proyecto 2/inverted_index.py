import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import os
import math
import pickle
from joblib import Parallel, delayed
import time
import bisect  # Para la búsqueda binaria en claves ordenadas

nltk.download('punkt')

class InvertedIndex:
    def __init__(self, dataset):
        self.stemmer = SnowballStemmer('english')
        self.stoplist = set()

        # Cargar la lista de stopwords
        with open("utils/stoplist.txt", encoding="utf-8") as file:
            self.stoplist = set(line.rstrip().lower() for line in file)
        self.stoplist.update(['?', '-', '.', ':', ',', '!', ';', '_'])

        # Leer el dataset
        self.data = pd.read_csv(dataset)
        self.dataset = { row['song_id']: self.pre_processing(row['lyrics']) for _, row in self.data.iterrows() }

        self.path = 'utils/inverted_index'
        self.total_docs = len(self.data)
        self.total_blocks = 0
        self.block_limit = 500
        self.idf = {}  # Diccionario para almacenar IDF precalculado

    def pre_processing(self, text, stemming=False):
        text = text.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(text)
        words = [
            word for word in words
            if word.isascii() and word.isalpha() and word not in self.stoplist
        ]
        if stemming:
            words = [self.stemmer.stem(word) for word in words]
        return words

    def precalculate_idf(self):
        """Precalcular el IDF para todos los términos en el dataset."""
<<<<<<< HEAD:Proyecto 2/inverted_index.py
        dictionary = defaultdict(dict)

        for doc_id, words in self.dataset.items():
            for word in words:
                if word not in dictionary:
                    dictionary[word] = { doc_id: 1 }
                else:
                    if doc_id in dictionary[word]:
                        dictionary[word][doc_id] += 1
                    else:
                        dictionary[word][doc_id] = 1

        for word, postings in dictionary.items():
            self.idf[word] = math.log10(self.total_docs / len(postings))

=======
        for block in range(self.total_blocks):
            with open(f"{self.path}/block_{block}.bin", "rb") as file:
                data = pickle.load(file)

                for word, postings in data.items():
                    self.idf[word] = math.log10(self.total_docs / len(postings))
    
>>>>>>> f5c1dc038a88d5bae9d026c8b682bc50425c9468:inverted_index.py
    def spimi_invert(self):
        start_time = time.time()  # Medir el tiempo de inicio
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
        self.precalculate_idf()

        end_time = time.time()  # Medir el tiempo de finalización
        print(f"Tiempo para construir el índice invertido: {end_time - start_time:.2f} segundos.")

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

    def save_temp_block(self, dictionary, block_count):
        """Guardar bloque temporal en disco en formato binario."""
        sorted_keys = sorted(dictionary.keys())
        sorted_values = { term: dictionary[term] for term in sorted_keys }
        with open(f"{self.path}/temp_block_{block_count}.bin", "wb") as file:
            pickle.dump(sorted_values, file)

    def save_block(self, dictionary, block_count):
        """Guardar bloque final en disco en formato binario."""
        sorted_keys = sorted(dictionary.keys())  # Asegúrate de que las claves estén ordenadas
        sorted_values = { term: dictionary[term] for term in sorted_keys }
        with open(f"{self.path}/block_{block_count}.bin", "wb") as file:
            pickle.dump(sorted_values, file)

    def search_in_blocks(self, word):
        """Buscar un término en los bloques invertidos usando búsqueda binaria en archivos binarios."""
        self.total_blocks = len([name for name in os.listdir(self.path)
                                 if os.path.isfile(os.path.join(self.path, name)) and name.startswith('block_')])

        left, right = 0, self.total_blocks - 1
        while left <= right:
            mid = (left + right) // 2

            with open(f"{self.path}/block_{mid}.bin", "rb") as file:
                data = pickle.load(file)
                # Convertimos las claves a lista para búsqueda binaria
                sorted_keys = list(data.keys())
                index = bisect.bisect_left(sorted_keys, word)

                if index < len(sorted_keys) and sorted_keys[index] == word:
                    return data[sorted_keys[index]]
                elif word < sorted_keys[0]:
                    right = mid - 1
                elif word > sorted_keys[-1]:
                    left = mid + 1
                else:
                    break

        return -1

    def query_search(self, query, top_k=5):
        """Realizar búsqueda por consulta usando TF-IDF."""
<<<<<<< HEAD:Proyecto 2/inverted_index.py
        start_time = time.time()  # Medir el tiempo de inicio
        query = self.pre_processing(query)  # Procesamiento previo de la consulta
        query_tf = {term: query.count(term) for term in query}  # Calcular TF para la consulta
        query_tfidf = {}
        document_magnitude = {}
        scores = {}
=======
        query_words = self.pre_processing(query)
>>>>>>> f5c1dc038a88d5bae9d026c8b682bc50425c9468:inverted_index.py

        term_freq = defaultdict(int)
        weights = defaultdict(float)

<<<<<<< HEAD:Proyecto 2/inverted_index.py
            idf = self.idf[term]
            query_tfidf[term] = math.log10(1 + query_tf[term]) * idf  # Calcular TF-IDF de la consulta

            # Buscar el término en los bloques (índice invertido)
            term_data = self.search_in_blocks(term)
            if term_data == -1:
                continue  # Si no hay datos para el término, continuar con el siguiente término

            # Calcular el puntaje de similitud para cada documento
            for doc_id, freq in term_data.items():
                doc_tfidf = math.log10(1 + freq) * idf  # TF-IDF del documento
                scores[doc_id] = scores.get(doc_id, 0) + doc_tfidf * query_tfidf[term]  # Actualizar puntaje
                document_magnitude[doc_id] = document_magnitude.get(doc_id,
                                                                    0) + doc_tfidf ** 2  # Magnitud del documento

        # Normalizar los puntajes utilizando las magnitudes de los vectores
        query_magnitude = math.sqrt(sum(val ** 2 for val in query_tfidf.values()))  # Magnitud de la consulta
        for doc_id in scores:
            doc_vector_magnitude = math.sqrt(document_magnitude[doc_id])  # Magnitud del documento
            # Normalizar el puntaje del documento
            scores[doc_id] = scores[doc_id] / (query_magnitude * doc_vector_magnitude)

        end_time = time.time()  # Medir el tiempo de finalización
        print(f"Tiempo para procesar la consulta: {end_time - start_time:.2f} segundos.")

        # Devolver los resultados ordenados por puntuación de mayor a menor, limitado a 'top_k'
        return list(sorted(scores.items(), key=lambda x: x[1], reverse=True))[:top_k]
=======
        for word in query_words:
            term_freq[word] += 1

        query_pow2_len = 0
        docs_pow2_lens = defaultdict(float)

        for block in range(self.total_blocks):
            with open(f"{self.path}/block_{block}.bin", "rb") as file:
                data = pickle.load(file)

                for word, postings in data.items():
                    query_tf_idf = math.log10(1 + term_freq[word]) * self.idf[word]
                    query_pow2_len += query_tf_idf ** 2

                    for doc_id, tf in postings.items():
                        docs_pow2_lens[doc_id] += (math.log10(1 + tf) * self.idf[word]) ** 2
                        weights[doc_id] += query_tf_idf * math.log10(1 + tf) * self.idf[word]
            
        for i in weights:
            if (query_pow2_len > 0 and weights[i] > 0):
                weights[i] /= (math.sqrt(query_pow2_len) * math.sqrt(docs_pow2_lens[i]))
            
        results = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:top_k]

        formatted_res = []
        for doc_id, score in results:
            song_info = self.data[self.data['song_id'] == doc_id].values[0]
            data = {
                "song_id": song_info[1],
                "song": song_info[3],
                "artist": eval(song_info[4])[0],
                "genre": ", ".join(song_info[6].split(";")[:3]) if isinstance(song_info[6], str) else "",
                "score": round(score, 4),
                "lyrics": song_info[7]
            }
            formatted_res.append(data)
        
        return formatted_res
>>>>>>> f5c1dc038a88d5bae9d026c8b682bc50425c9468:inverted_index.py

if __name__ == "__main__":
    # Medir tiempo de construcción del índice
    start_time = time.time()
    index = InvertedIndex('utils/dataset.csv')

    # Crear el índice invertido
    index.spimi_invert()
    end_time = time.time()
    print(f"Tiempo total para construir el índice invertido: {end_time - start_time:.2f} segundos.")

    # Realizar una búsqueda de ejemplo
    query_result = index.query_search('In a haze, a stormy haze', 5)
    print("Top 5 resultados de búsqueda:", query_result)