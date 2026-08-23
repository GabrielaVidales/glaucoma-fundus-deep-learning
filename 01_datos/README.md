# 01 · Datos

Etapa de descarga y preparación de las imágenes. Ejecutar en este orden:

1. **`01_descarga_dataset_kaggle.ipynb`** — Descarga el dataset público de Kaggle (fotos de fondo de ojo: normal / glaucoma) y lo copia a Google Drive.
2. **`02_creacion_muestra_reducida.ipynb`** — Crea una muestra más pequeña del dataset, útil para probar el pipeline completo rápidamente antes de correr con todos los datos.
3. **`03_generacion_matrices_X_y.ipynb`** — Convierte las imágenes en matrices numéricas (`X`) con sus etiquetas (`y`: 0 = normal, 1 = glaucoma) y las guarda como archivos `.npy`, listos para alimentar cualquier modelo de las carpetas siguientes.

El código incluye comentarios explicando cada operación (lectura de imágenes, redimensionado, conversión de color, etc.).
