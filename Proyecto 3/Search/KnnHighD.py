import faiss
import time

class knnHighD_LSH:
    def __init__(self, dimension, num_bits=512):
        self.dimension = dimension
        self.num_bits = num_bits
        self.idx = faiss.IndexLSH(dimension, num_bits)

    def insert(self, features):
        self.idx.add(features)

    def knn_search(self, query_vector, k=5):
        start_time = time.time()

        distances, indices = self.idx.search(query_vector, k)

        end_time = time.time()
        print(f"\nTiempo de búsqueda KNN con Faiss: {end_time - start_time:.6f} segundos")

        return distances, indices