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
        bbox_query = tuple(query_vector) + tuple(query_vector)
        nearest = list(self.idx.nearest(bbox_query, k, objects=True))
        top_k = []
        for item in nearest:
            current_vector = self.data[item.id]
            distance = np.linalg.norm(np.array(current_vector) - np.array(query_vector))
            top_k.append((distance, item.id))
        top_k.sort(key=lambda x: x[0])
        return top_k[:k]

    def insert_in_batches(self, features, batch_size=50):
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            for idx, vector in enumerate(batch):
                vector = [float(coord) for coord in vector]
                unique_id = i + idx
                bbox = tuple(vector) + tuple(vector)
                self.idx.insert(unique_id, bbox)
                self.data[unique_id] = vector