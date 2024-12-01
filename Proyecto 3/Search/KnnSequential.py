import numpy as np
import heapq

# Búsqueda KNN
def knn_priority_queue_images(data_features, query_feature, k):
    result = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if len(result) < k:
            heapq.heappush(result, (-dist, idx))
        else:
            heapq.heappushpop(result, (-dist, idx))
    nearest_neighbors = [idx for _, idx in result]
    return nearest_neighbors

# Búsqueda por Rango
def range_search_images(data_features, query_feature, radius):
    neighbors = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if dist <= radius:
            neighbors.append((dist, idx))
    neighbors.sort(key=lambda x: x[0])
    return neighbors
