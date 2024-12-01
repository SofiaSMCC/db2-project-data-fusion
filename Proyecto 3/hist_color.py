import cv2

def calcular_histograma_y_comparar(imagen1_path, imagen2_path):
    imagen1 = cv2.imread(imagen1_path)
    imagen2 = cv2.imread(imagen2_path)

    if imagen1 is None or imagen2 is None:
        print("Error al cargar las imágenes.")
        return

    # Convertir a HSV para mejor representación de color
    imagen1_hsv = cv2.cvtColor(imagen1, cv2.COLOR_BGR2HSV)
    imagen2_hsv = cv2.cvtColor(imagen2, cv2.COLOR_BGR2HSV)

    # Calcular histogramas normalizados para cada imagen (32 bins para Hue y Saturación)
    hist1 = cv2.calcHist([imagen1_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist2 = cv2.calcHist([imagen2_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])

    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Comparar histogramas utilizando Correlación
    correlacion = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    print(f"Relación entre las imágenes (correlación): {correlacion:.2f}")

# Rutas de las imágenes
imagen1_path = "/content/0cf7fc0398e54c889d321fdd2c8bb083.jpg"
imagen2_path = "/content/fae7e19b15694b738383d0f57f600ed2.jpg"
calcular_histograma_y_comparar(imagen1_path, imagen2_path)
