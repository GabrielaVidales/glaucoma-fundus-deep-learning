# 04 · Evaluation

| File | Contents |
|---|---|
| `01_predicciones_y_matriz_confusion.ipynb` | Loads a trained model, predicts on the test images, and compares predictions with the true labels using a confusion matrix; includes examples of correct and incorrect predictions. |
| `02_presentacion_comparativa_modelos.pptx` | Slides comparing all trained models (sample sizes, architectures, augmentation). The quickest way to understand the results without reading code (in Spanish). |

## Reading a confusion matrix

|                    | Predicted: normal | Predicted: glaucoma |
|--------------------|:---:|:---:|
| **True: normal**   | ✅ True negative | ❌ False positive |
| **True: glaucoma** | ❌ False negative | ✅ True positive |

In screening, **false negatives** (predicting "normal" when glaucoma is present) are the most serious error, because the disease is irreversible. For this reason the thesis prioritizes sensitivity and selects the decision threshold on validation data, reporting specificity, PPV and NPV alongside it.
