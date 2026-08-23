# 03 · Transfer Learning

Modelos construidos a partir de arquitecturas ya entrenadas en ImageNet (millones de imágenes genéricas), adaptadas a la tarea de este proyecto.

| Notebook | Qué hace |
|---|---|
| `01_resnet50v2.ipynb` | Usa **ResNet50V2** como base, re-entrenada sobre el dataset completo de fondo de ojo. |
| `02_mobilenetv3small.ipynb` | Usa **MobileNetV3Small** (arquitectura ligera). Flujo completo: cargar datos → entrenar con el backbone congelado → fine-tuning descongelando las últimas capas → evaluación. |
| `03_mobilenetv3small_test_final.ipynb` | Versión final y más completa: agrega evaluación rigurosa sobre un conjunto de **TEST** independiente, con Precision, Recall, F1-score, matriz de confusión, curva ROC y AUC. |

Los modelos resultantes están en `../05_modelos/` con el prefijo `transferlearning_`.
