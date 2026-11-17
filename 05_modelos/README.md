# 05 · Trained models

Keras models of the baseline CNN (input: float32 array `(N, 220, 220, 3)` scaled to [0, 1]; output: probability of glaucoma).

| File | Description |
|---|---|
| `cnn_propia_mejor_modelo_conservador.keras` | Best baseline CNN (high-specificity operating point). |
| `cnn_propia_muestra_5000.keras` | Baseline CNN trained on the 5,000-image subset. |
| `cnn_propia_muestra_5000_100epocas.keras` | Same configuration trained for 100 epochs. |

```python
from tensorflow import keras
model = keras.models.load_model("05_modelos/cnn_propia_mejor_modelo_conservador.keras")
prob_glaucoma = model.predict(X / 255.0)
```

Transfer-learning models (ResNet50V2, MobileNetV3Small) are not included because of file size; they can be regenerated with the notebooks in `03_transfer_learning/`.
