# Data Fusion

## Introducción

### Objetivo del Proyecto

Desarrollar un sistema integral de base de datos multimedia que optimice la búsqueda y recuperación de información, implementando un índice invertido para documentos de texto y una estructura multidimensional para la búsqueda eficiente de imágenes, audio y otros objetos multimedia.

### Descripción del dataset

Se utilizó una base de datos de Kaggle ([dataset](https://www.kaggle.com/datasets/evabot/spotify-lyrics-dataset)) que contiene 8,673 registros de canciones junto con su información asociada. Cada entrada incluye atributos como `song_id` (ID de la canción), `artist_id` (ID del artista), `song` (nombre de la canción), `artists` (nombres de los artistas), `explicit` (indicador de contenido explícito), `genres` (géneros musicales) y `lyrics` (letras de las canciones). Este conjunto de datos ofrece una amplia variedad de características relacionadas con las canciones y sus artistas, lo que lo convierte en una valiosa fuente para análisis musicales y tareas de recuperación de información.

### Importancia de Aplicar Indexación

La indexación es esencial para mejorar la eficiencia y velocidad en la recuperación de información, tanto en bases de datos relacionales como en sistemas de búsqueda. Los índices permiten organizar los datos de manera estructurada, facilitando consultas rápidas, especialmente en grandes volúmenes de información. Además de los índices tradicionales, los índices multidimensionales son fundamentales para manejar datos complejos, como imágenes o audio, ya que permiten realizar búsquedas eficientes en espacios con múltiples características o dimensiones. En conjunto, estos métodos optimizan el rendimiento, reducen los tiempos de respuesta y mejoran la escalabilidad de los sistemas.

## Backend

## Índice Invertido
