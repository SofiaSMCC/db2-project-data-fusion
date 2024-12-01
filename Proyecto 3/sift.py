import cv2

# Cargar las imágenes
image1 = cv2.imread('poke2/f4116f977cf94064a9acec285237d4f8.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('poke2/fae7e19b15694b738383d0f57f600ed2.jpg', cv2.IMREAD_GRAYSCALE)

# Crear el detector SIFT
sift = cv2.SIFT_create()

# Detectar puntos clave y calcular descriptores para ambas imágenes
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Crear el emparejador de fuerza bruta
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # NORM_L2 es para SIFT

# Encontrar las mejores coincidencias
matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # Obtener las dos mejores coincidencias para cada descriptor

# Aplicar el ratio test de Lowe para filtrar coincidencias
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Umbral para filtrar coincidencias
        good_matches.append(m)

# Dibujar las mejores coincidencias
result_image = cv2.drawMatches(
    image1, keypoints1,
    image2, keypoints2,
    good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

output_path = 'result_image.jpg'  # Ruta donde se guardará la imagen
cv2.imwrite(output_path, result_image)
print(f"Imagen guardada en: {output_path}")

# Mostrar el resultado
cv2.imshow("Coincidencias entre imágenes", result_image)

total_keypoints = min(len(keypoints1), len(keypoints2))
similitud = len(good_matches) / total_keypoints if total_keypoints > 0 else 0

print(f"Similitud entre imágenes: {similitud * 100:.2f}%")

