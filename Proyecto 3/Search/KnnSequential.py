import time
import numpy as np
import heapq

# Búsqueda KNN
def knn_priority_queue(data_features, query_feature, k):
    start_time = time.time()

    """Realiza la búsqueda KNN usando una cola de prioridad."""
    heap = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, idx))
        else:
            heapq.heappushpop(heap, (-dist, idx))

    end_time = time.time()
    print(f"\nTiempo de búsqueda knn sin indexación: {end_time - start_time:.7f} segundos")

    return [(abs(dist), idx) for dist, idx in sorted(heap)]

# Búsqueda por Rango
def range_search(data_features, query_feature, radius):
    start_time = time.time()

    """Realiza una búsqueda por rango en las características."""
    neighbors = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if dist <= radius:
            neighbors.append((dist, idx))
    neighbors.sort(key=lambda x: x[0])

    end_time = time.time()
    print(f"\nTiempo de búsqueda por rango: {end_time - start_time:.7f} segundos")

    return neighbors
