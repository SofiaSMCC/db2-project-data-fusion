from rtree import index
import numpy as np

class KnnRTree:
    def __init__(self, dimension = 2048):
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
        bbox = tuple(query_vector) + tuple(query_vector)
        nearest = list(self.idx.nearest(bbox, k + 1, objects=True))  # Buscar k+1 para considerar la exclusión
        top_k = []
        for item in nearest:
            current_vector = self.data[item.id]
            distance = np.linalg.norm(current_vector - query_vector)
            if distance != 0:
                top_k.append((distance, item.id))
        top_k.sort(key=lambda x: x[0])
        return top_k[:k]