import os
from sklearn.decomposition import PCA
from Search.knnRtree import KnnRTree
from Features.extract_features import load_resnet_feature_extractor, get_transform, load_features, extract_features, reduce_single_feature, reduce_dimensions, extract_features_from_folder, save_features

def Run_KnnRtree(query_image_path='poke2/00000001.jpg', k=5):

    feature_extractor = load_resnet_feature_extractor()
    transform = get_transform()

    feature_file = 'image_features.npz'

    if os.path.exists(feature_file):
        print("Cargando características desde el archivo guardado...")
        data_features, image_paths = load_features(feature_file)
    else:
        print("Extrayendo características de las imágenes...")
        folder_path = 'poke2'
        image_paths, data_features = extract_features_from_folder(folder_path, feature_extractor, transform)

        # Guardar características para usos futuros
        save_features(feature_file, data_features, image_paths)
        print(f"Características guardadas en {feature_file}")

    query_feature = extract_features(query_image_path, feature_extractor, transform)

    if len(data_features) > 1:
        pca = PCA(n_components=min(100, len(data_features[0])))
        data_features_reduced, pca_model = reduce_dimensions(data_features, n_components=pca.n_components)
        query_feature_reduced = reduce_single_feature(query_feature, pca_model)
    else:
        data_features_reduced = data_features
        query_feature_reduced = query_feature

    # Inserción en lotes en el R-tree
    rtree = KnnRTree()
    rtree.insert_in_batches(data_features_reduced, batch_size=25)

    # Búsqueda KNN con R-tree
    knn_rtree_results = rtree.knn_search(query_feature_reduced, k)
    print("\nImágenes más similares con R-Tree:")
    for dist, result_id in knn_rtree_results:
        print(f"- {image_paths[result_id]} (Distancia: {dist:.4f})")

    return knn_rtree_results

if __name__ == "__main__":
    Run_KnnRtree()
