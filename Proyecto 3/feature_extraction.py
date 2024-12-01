import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from knnRtree import KnnRTree
from PIL import Image
import numpy as np
import heapq
import os

def load_resnet_feature_extractor():
    #Carga un modelo ResNet50 preentrenado como extractor de características
    resnet50 = models.resnet50(pretrained=True).eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return feature_extractor


def get_transform():
    #Define las transformaciones necesarias para preprocesar las imágenes.
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
    return [idx for _, idx in heap]


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
    query_image_path = 'pokemon/0a3a642e700b4153b115e0f645d273f1.jpg'
    query_feature = extract_features(query_image_path, feature_extractor, transform)

    folder_path = 'pokemon'
    image_paths, data_features = extract_features_from_folder(folder_path, feature_extractor, transform)

    # Búsqueda KNN
    k = 5
    knn_results = knn_priority_queue(data_features, query_feature, k)
    print(f"Imagen consulta: {query_image_path}")
    print("5 imágenes más similares:")
    for idx in knn_results:
        print(f"- {image_paths[idx]}")

    # Búsqueda por rango
    radius = 0.5
    range_results = range_search(data_features, query_feature, radius)
    print("\nImágenes dentro del radio de búsqueda:")
    for _, idx in range_results:
        print(f"- {image_paths[idx]}")

    plot_distance_distribution(data_features, query_feature)

    # Búsqueda con R-Tree
    rtree = KnnRTree()
    rtree.insert(data_features)
    knn_rtree_results = rtree.knn_search(query_feature, k)
    print("\nImágenes más similares con R-Tree:")
    for dist, result_id in knn_rtree_results:
        print(f"- {image_paths[result_id]} (Distancia: {dist:.4f})")


if __name__ == "__main__":
    main()
