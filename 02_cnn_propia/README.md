# 02 · CNN Propia

Una red neuronal convolucional (CNN) diseñada desde cero (capas Conv2D + MaxPooling2D + Dense), sin usar redes pre-entrenadas.

| Notebook | Qué prueba |
|---|---|
| `01_cnn_muestra_2000.ipynb` | Primera versión, con una muestra pequeña (~2000 imágenes de entrenamiento) para validar que el pipeline funciona. |
| `02_cnn_muestra_5000.ipynb` | Mismo modelo con más datos (~5000 imágenes) y ya incluye matriz de confusión. |
| `03_cnn_dataset_completo.ipynb` | El mismo tipo de modelo entrenado con **todo** el dataset disponible. |
| `04_cnn_data_augmentation_v1.ipynb` | Añade data augmentation (rotaciones, volteos, zoom) para mejorar la generalización. |
| `05_cnn_data_augmentation_v2.ipynb` | Segunda iteración/ajuste sobre la versión anterior con data augmentation. |

Los modelos resultantes de estos notebooks están guardados en `../05_modelos/` con el prefijo `cnn_propia_`.
