# 04 · Evaluación

| Archivo | Qué contiene |
|---|---|
| `01_predicciones_y_matriz_confusion.ipynb` | Carga un modelo entrenado y genera predicciones sobre imágenes de prueba, comparándolas con la etiqueta real mediante una matriz de confusión. |
| `02_presentacion_comparativa_modelos.pptx` | Presentación con la comparación de desempeño entre los distintos modelos entrenados (tamaños de muestra, arquitecturas, etc.). Es el mejor punto de partida para entender los resultados sin leer código. |

## Cómo leer una matriz de confusión

|                     | Predicho: Normal | Predicho: Glaucoma |
|---------------------|:---:|:---:|
| **Real: Normal**    | ✅ Acierto | ❌ Falso positivo |
| **Real: Glaucoma**  | ❌ Falso negativo | ✅ Acierto |

En un contexto médico, los **falsos negativos** (decir "normal" cuando hay glaucoma real) son el error más grave, porque implican no detectar una enfermedad presente.
