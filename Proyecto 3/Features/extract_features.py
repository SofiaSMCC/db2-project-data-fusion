import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import os

# === Reducción de Dimensiones ===

# Reduce las dimensiones de los vectores de características usando PCA
def reduce_dimensions(features, n_components=100):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca

# Reduce las dimensiones de un único vector usando un modelo PCA preajustado
def reduce_single_feature(feature, pca_model):
    return pca_model.transform([feature])[0]


# === Configuración ===

# Carga un modelo ResNet50 preentrenado como extractor de características
def load_resnet_feature_extractor():
    resnet50 = models.resnet50(pretrained=True).eval()
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return feature_extractor

# Define las transformaciones necesarias para preprocesar las imágenes
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# === Extracción de características ===

# Extrae el vector de características de una imagen dada
def extract_features(image_path, feature_extractor, transform):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(input_tensor)
    return features.squeeze().numpy()

# Extrae características de todas las imágenes en una carpeta
def extract_features_from_folder(folder_path, feature_extractor, transform):
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]
    features = [extract_features(image_path, feature_extractor, transform) for image_path in image_paths]
    return image_paths, features


# === Guardar y cargar características ===

# Guarda los vectores de características y las rutas de las imágenes en un archivo
def save_features(file_path, features, image_paths):
    np.savez(file_path, features=features, image_paths=image_paths)

# Carga los vectores de características y las rutas de las imágenes desde un archivo
def load_features(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data['features'], data['image_paths']