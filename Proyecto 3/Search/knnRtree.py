import time
from rtree import index
import numpy as np

class KnnRTree:
    def __init__(self, dimension=100):
        p = index.Property()
        p.dimension = dimension
        self.idx = index.Index(properties=p)
        self.data = {}

    def insert(self, features):
        for idx, vector in enumerate(features):
            vector = [float(coord) for coord in vector]
            unique_id = idx
            bbox = tuple(vector) + tuple(vector)
            self.idx.insert(unique_id, bbox)
            self.data[unique_id] = vector

    def knn_search(self, query_vector, k):
        start_time = time.time()

        bbox_query = tuple(query_vector) + tuple(query_vector)
        nearest = list(self.idx.nearest(bbox_query, k, objects=True))
        top_k = []
        for item in nearest:
            current_vector = self.data[item.id]
            distance = np.linalg.norm(np.array(current_vector) - np.array(query_vector))
            top_k.append((distance, item.id))
        top_k.sort(key=lambda x: x[0])

        end_time = time.time()
        print(f"\nTiempo de búsqueda KNN con RTree: {end_time - start_time:.4f} segundos")

        return top_k[:k]

    def insert_in_batches(self, features, batch_size=50):
        start_time = time.time()

        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            for idx, vector in enumerate(batch):
                vector = [float(coord) for coord in vector]
                unique_id = i + idx
                bbox = tuple(vector) + tuple(vector)
                self.idx.insert(unique_id, bbox)
                self.data[unique_id] = vector
                
        end_time = time.time()
        print(f"\nTiempo de inserción por lotes: {end_time - start_time:.4f} segundos")