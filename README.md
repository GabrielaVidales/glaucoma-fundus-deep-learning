# 👁️ Detección de Glaucoma en Imágenes de Fondo de Ojo

Proyecto de **deep learning** que entrena y compara distintos modelos de clasificación de imágenes (una CNN propia vs. transfer learning) para distinguir entre imágenes de **fondo de ojo normales** y con **signos de glaucoma**.

> ⚠️ **Nota**: este es un proyecto académico/de práctica. No es un dispositivo médico ni sustituye el diagnóstico de un oftalmólogo.

---

## 📖 Contenido

- [¿Qué es el glaucoma?](#-qué-es-el-glaucoma-y-por-qué-importa)
- [Dataset](#-dataset)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Enfoques probados](#-enfoques-probados)
- [Glosario para no iniciados](#-glosario-para-no-iniciados)
- [Cómo ejecutar los notebooks](#-cómo-ejecutar-los-notebooks)
- [Resultados](#-resultados)

---

## 🩺 ¿Qué es el glaucoma y por qué importa?

El glaucoma es una enfermedad que daña progresivamente el nervio óptico y puede causar ceguera irreversible si no se detecta a tiempo. Suele diagnosticarse revisando el **fondo de ojo** (la parte trasera del ojo, incluyendo el nervio óptico), algo que normalmente hace un especialista con equipo médico. Este proyecto explora si un modelo de inteligencia artificial puede aprender a distinguir, a partir de una foto de fondo de ojo, si es **normal** o si muestra señales de **glaucoma**.

## 📸 Dataset

Las imágenes provienen de un dataset público de Kaggle con fotografías de fondo de ojo, clasificadas en `normal` y `glaucoma`, y ya divididas en tres subconjuntos:
- **train**: para entrenar los modelos
- **val (validación)**: usado durante el entrenamiento para ajustar y elegir el mejor modelo
- **test**: usado solo al final, para medir el desempeño real con imágenes que el modelo nunca vio

## 🗂 Estructura del repositorio

```
├── 01_datos/                  → Descarga y preparación de las imágenes
├── 02_cnn_propia/             → Red neuronal convolucional diseñada desde cero
├── 03_transfer_learning/      → Modelos basados en redes ya pre-entrenadas
├── 04_evaluacion/             → Métricas, matriz de confusión y comparación final
└── 05_modelos/                → Modelos ya entrenados (.keras), listos para usar
```

Cada carpeta tiene su propio `README.md` con el detalle de cada archivo.

## 🧪 Enfoques probados

| Enfoque | Carpeta | Idea |
|---|---|---|
| **CNN propia** | `02_cnn_propia/` | Red neuronal convolucional construida capa por capa, entrenada desde cero solo con las imágenes de este proyecto. Se prueba con distintos tamaños de muestra y con/sin data augmentation. |
| **Transfer Learning** | `03_transfer_learning/` | Se parte de redes ya entrenadas con millones de imágenes genéricas (MobileNetV3Small, ResNet50V2) y se adaptan a esta tarea específica. Suele requerir menos datos y dar mejores resultados. |

## 📚 Glosario para no iniciados

| Término | Significado |
|---|---|
| **CNN** (Red Neuronal Convolucional) | Tipo de red diseñada especialmente para "ver" y reconocer patrones en imágenes. |
| **Transfer Learning** | Reutilizar una red ya entrenada por otros (en un dataset genérico enorme) y adaptarla a una tarea nueva y específica, en vez de entrenar desde cero. |
| **MobileNetV3Small / ResNet50V2** | Arquitecturas de redes neuronales famosas y ya pre-entrenadas, usadas como base para el transfer learning. |
| **Fine-tuning** (ajuste fino) | Descongelar y re-entrenar ligeramente las últimas capas de una red pre-entrenada para especializarla mejor en la tarea nueva. |
| **Data Augmentation** | Generar variaciones artificiales de las imágenes de entrenamiento (rotarlas, voltearlas, hacer zoom) para que el modelo generalice mejor y no memorice. |
| **Época (epoch)** | Una pasada completa del modelo por todo el conjunto de entrenamiento. |
| **X_train / y_train / X_val / y_val / X_test / y_test** | `X` = imágenes convertidas en números; `y` = la respuesta correcta (0 = normal, 1 = glaucoma). `train` entrena, `val` ajusta durante el entrenamiento, `test` evalúa al final de forma imparcial. |
| **Matriz de confusión** | Tabla que compara lo que el modelo predijo contra la respuesta correcta, mostrando aciertos y tipos de error. |
| **`.ipynb`** | Un notebook de Jupyter/Google Colab: mezcla código, texto y resultados en celdas que se ejecutan una por una. |
| **`.keras`** | Formato de archivo donde se guarda un modelo ya entrenado, listo para cargarse y usarse sin volver a entrenar. |

## ▶️ Cómo ejecutar los notebooks

1. Sube la carpeta del proyecto a tu Google Drive (los notebooks fueron hechos para **Google Colab** y montan Drive automáticamente).
2. Abre cualquier `.ipynb` con [Google Colab](https://colab.research.google.com) o con Jupyter localmente.
3. Sigue el orden numerado de las carpetas: `01_datos` → `02_cnn_propia` / `03_transfer_learning` → `04_evaluacion`.
4. Si solo quieres usar un modelo ya entrenado sin re-entrenar nada, ve directo a `05_modelos/` y carga el archivo `.keras` que prefieras con:
   ```python
   from tensorflow import keras
   modelo = keras.models.load_model("05_modelos/transferlearning_mobilenetv3small_best.keras")
   ```

### Dependencias
Ver [`requirements.txt`](requirements.txt). Instalación rápida:
```bash
pip install -r requirements.txt
```

### ⚠️ Sobre los archivos `.keras` grandes (Git LFS)
Dos modelos (`05_modelos/cnn_propia_v2.keras` ~132 MB y `05_modelos/transferlearning_resnet50v2.keras` ~94 MB) superan o rondan el límite de 100 MB que GitHub permite sin Git LFS. Este repo ya incluye un `.gitattributes` configurado para trackear `*.keras` con [Git LFS](https://git-lfs.com). Antes de subir el repo, instala Git LFS y ejecuta:
```bash
git lfs install
git add .gitattributes
git add .
git commit -m "Proyecto inicial"
git push
```

## 📊 Resultados

El resumen comparativo de desempeño entre todos los modelos está en `04_evaluacion/02_presentacion_comparativa_modelos.pptx`, y el detalle de matriz de confusión en `04_evaluacion/01_predicciones_y_matriz_confusion.ipynb`.

En términos médicos, al evaluar estos modelos conviene prestar especial atención a los **falsos negativos** (el modelo dice "normal" cuando en realidad hay glaucoma), ya que representan el error más costoso: una enfermedad real que pasa desapercibida.

---

## 📄 Licencia
Ver [`LICENSE`](LICENSE).
