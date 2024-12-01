import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from knnRtree import KnnRTree
from PIL import Image
import numpy as np
import heapq
import os
from sklearn.decomposition import PCA


def reduce_dimensions(features, n_components=100):
    #Reduce las dimensiones de los vectores de características usando PCA.
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca


def reduce_single_feature(feature, pca_model):
    #Reduce las dimensiones de un único vector usando un modelo PCA preajustado.
    return pca_model.transform([feature])[0]



def load_resnet_feature_extractor():
    #Carga un modelo ResNet50 preentrenado como extractor de características
    resnet50 = models.resnet50(pretrained=True).eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return feature_extractor


def get_transform():
    #Define las transformaciones necesarias para preprocesar las imágenes."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


#Extracción de características
def extract_features(image_path, feature_extractor, transform):
    """Extrae el vector de características de una imagen dada."""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(input_tensor)
    return features.squeeze().numpy()


def extract_features_from_folder(folder_path, feature_extractor, transform):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    features = [extract_features(image_path, feature_extractor, transform) for image_path in image_paths]
    return image_paths, features


#KNN

def knn_priority_queue(data_features, query_feature, k):
    #Realiza la búsqueda KNN usando una cola de prioridad
    heap = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, idx))
        else:
            heapq.heappushpop(heap, (-dist, idx))
    return [(abs(dist), idx) for dist, idx in sorted(heap)]


#Búsqueda por rango

def range_search(data_features, query_feature, radius):
    """Realiza una búsqueda por rango en las características."""
    neighbors = []
    for idx, feature in enumerate(data_features):
        dist = np.linalg.norm(query_feature - feature)
        if dist <= radius:
            neighbors.append((dist, idx))
    neighbors.sort(key=lambda x: x[0])
    return neighbors



def plot_distance_distribution(data_features, query_feature):
    """Grafica la distribución de distancias entre imágenes."""
    distances = [np.linalg.norm(query_feature - feature) for feature in data_features]
    plt.hist(distances, bins=30, edgecolor='black')
    plt.title('Distribución de Distancias')
    plt.xlabel('Distancia')
    plt.ylabel('Frecuencia')
    plt.show()

def main():
    feature_extractor = load_resnet_feature_extractor()
    transform = get_transform()

    # Extracción de características
    query_image_path = 'poke2/0965b2be7b9c436a870dcbb797d57818.jpg'
    query_feature = extract_features(query_image_path, feature_extractor, transform)

    folder_path = 'poke2'
    image_paths, data_features = extract_features_from_folder(folder_path, feature_extractor, transform)

    # Reducción de dimensiones para características de datos y consulta
    if len(data_features) > 1:
        pca = PCA(n_components=min(100, len(data_features[0])))
        data_features_reduced, pca_model = reduce_dimensions(data_features, n_components=pca.n_components)
        query_feature_reduced = reduce_single_feature(query_feature, pca_model)
    else:
        # Usar los vectores originales si no es posible aplicar PCA
        data_features_reduced = data_features
        query_feature_reduced = query_feature

    # KNN sin R-tree
    k = 5
    knn_results = knn_priority_queue(data_features_reduced, query_feature_reduced, k)
    print("\nKNN (sin R-Tree):")
    for dist, idx in knn_results:
        print(f"- {image_paths[idx]} (Distancia: {dist:.4f})")

    # Búsqueda por rango sin R-tree
    radius = 0.5
    range_results = range_search(data_features_reduced, query_feature_reduced, radius)
    print("\nBúsqueda por rango (sin R-Tree):")
    for dist, idx in range_results:
        print(f"- {image_paths[idx]} (Distancia: {dist:.4f})")

    # Inserción en lotes en el R-tree
    rtree = KnnRTree()
    rtree.insert_in_batches(data_features_reduced, batch_size=25)

    # Búsqueda KNN con R-tree
    knn_rtree_results = rtree.knn_search(query_feature_reduced, k)
    print("\nImágenes más similares con R-Tree:")
    for dist, result_id in knn_rtree_results:
        print(f"- {image_paths[result_id]} (Distancia: {dist:.4f})")

    # Visualización
    plot_distance_distribution(data_features_reduced, query_feature_reduced)


if __name__ == "__main__":
    main()
