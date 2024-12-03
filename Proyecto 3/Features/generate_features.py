import os.path
from Features.extract_features import load_resnet_feature_extractor, get_transform, reduce_dimensions, extract_features_from_folder, save_features

def generate_feature_archive(data_path, n_components=100):

    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} no existe")

    feature_extractor = load_resnet_feature_extractor()
    transform = get_transform()

    print(f"Extrayendo características de imágenes en: {data_path}")

    image_paths, features = extract_features_from_folder(data_path, feature_extractor, transform)
    reduced_features, pca_model = reduce_dimensions(features, n_components=n_components)
    folder_name = os.path.basename(os.path.normpath(data_path))
    output_file = f"image_features_{folder_name}.npz"
    save_features(output_file, reduced_features, image_paths)

