import time
import numpy as np
import heapq

# Búsqueda KNN
def knn_priority_queue_images(data_features, query_feature, k):
    start_time = time.time()

    result = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if len(result) < k:
            heapq.heappush(result, (-dist, idx))
        else:
            heapq.heappushpop(result, (-dist, idx))
    nearest_neighbors = [idx for _, idx in result]

    end_time = time.time()
    print(f"\nTiempo de búsqueda KNN con cola de prioridad sin indexar: {end_time - start_time:.4f} segundos")

    return nearest_neighbors

# Búsqueda por Rango
def range_search_images(data_features, query_feature, radius):
    start_time = time.time()

    neighbors = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if dist <= radius:
            neighbors.append((dist, idx))
    neighbors.sort(key=lambda x: x[0])

    end_time = time.time()
    print(f"\nTiempo de búsqueda por rango sin indexar: {end_time - start_time:.4f} segundos")
    
    return neighbors