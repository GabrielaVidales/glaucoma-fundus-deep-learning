# 👁️ Glaucoma Detection from Fundus Photographs with Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-thesis%20in%20progress-yellow.svg)]()

Undergraduate thesis project (B.Sc. Engineering Physics, Universidad Autónoma de Yucatán) that develops and compares convolutional neural networks (CNNs) for **binary classification of color fundus photographs as normal or glaucomatous**, designed as a **screening aid**, not a diagnostic replacement.

- **Author:** Gabriela Yasmin Vidales Ayala
- **Advisor:** Dr. Enrique Camacho Pérez · **Co-advisor:** Dra. Karla Beatriz Cantún Avila
- **Funding of the preliminary stage:** University Scientific Impulse Project (UADY), Nov 2025 – Apr 2026

> ⚠️ **Disclaimer:** research code only. This is not a medical device and must not be used for clinical decisions.

---

## Contents

- [Why glaucoma screening](#why-glaucoma-screening)
- [Dataset](#dataset)
- [Methods](#methods)
- [Results](#results)
- [What the baseline taught us](#what-the-baseline-taught-us)
- [Thesis roadmap](#thesis-roadmap)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [References](#references)

---

## Why glaucoma screening

Glaucoma is a group of progressive, irreversible optic neuropathies and a leading cause of blindness. It is asymptomatic in early stages, so many patients are diagnosed only after substantial damage. An estimated **76.0 million people aged 40–80 had glaucoma in 2020, projected to reach 111.8 million by 2040** (Tham et al., 2014), and glaucoma caused blindness in **3.61 million people in 2020** (Vision Loss Expert Group, 2024).

Fundus photography lets clinicians assess the optic nerve head (cup-to-disc ratio, neuroretinal rim, ISNT rule), but interpreting it requires specialists who are scarce in rural and low-resource settings. This project asks whether a CNN can learn these signs from public data reliably enough to support screening.

The thesis combines three domains:
1. **Medical:** the diagnostic signs the network should learn (optic disc cupping, CDR, rim thinning).
2. **Physical optics:** how the retinal image is formed and degraded (eye optics, indirect ophthalmoscopy with a condensing lens, illumination, reflections).
3. **Computational:** deep learning, regularization, transfer learning, threshold calibration, and interpretability (Grad-CAM).

---

## Dataset

Images come from the public **Fundus images dataset for Glaucoma** (Poveda, 2024), a compilation of **22 scientific repositories** (including ACRIMA, ORIGA, G1020, REFUGE, DRISHTI-GS1, PAPILA, BEH, HRF and FIVES), which provides diversity in cameras, populations and labeling criteria. Data are public and de-identified; they are **not redistributed** in this repository.

| Configuration | Train (normal / glaucoma) | Validation (normal / glaucoma) | Independent test |
|---|---|---|---|
| Baseline subset | 4,000 (2,000 / 2,000) | 400 (200 / 200) | 480 (240 / 240) |
| Intermediate subset | 9,998 (~5,000 per class) | 1,000 | 480 (240 / 240) |
| Full training set | 16,899 (10,400 / 6,499) | 929 (480 / 449) | 480 (240 / 240) |

The **same balanced test set of 480 images** is held out across all experiments and is never used for training or model selection.

---

## Methods

**Preprocessing.** Images are read with OpenCV, converted BGR→RGB, resized to **220 × 220 px**, and stored as NumPy arrays `(N, 220, 220, 3)` with labels 0 = normal, 1 = glaucoma.

**Models.**

| Model | Description | Parameters |
|---|---|---|
| **Baseline CNN** (trained from scratch) | 3 blocks of Conv2D (32/64/128 filters, 3×3) + BatchNorm + MaxPool → GlobalAveragePooling → Dense(128) + Dropout(0.5) → sigmoid | 110,785 |
| **MobileNetV3Small** (transfer learning) | ImageNet weights; stage 1 frozen backbone (feature extraction), stage 2 fine-tuning of the last 30 layers with a lower learning rate | 1,013,105 |
| **ResNet50V2** (transfer learning) | ImageNet weights with a new classification head (GlobalAveragePooling + Dense + Dropout) | — |

**Training.** Adam, binary cross-entropy, Google Colab GPU. The baseline was trained for 100 epochs (batch 16); transfer-learning models use early stopping on validation loss with best-weight restoration.

**Augmentation variants of the baseline.** (a) none, (b) real-time horizontal flips with `ImageDataGenerator` (50 epochs), (c) static horizontal flip that doubles the training set to 8,000 images (100 epochs).

---

## Results

### Baseline CNN on the independent test set (n = 480, threshold 0.5)

95% Wilson confidence intervals in parentheses.

| Metric | E0a: no augmentation | E0b: real-time augmentation | E0c: static flip (final) |
|---|---|---|---|
| TP / FN / TN / FP | 128 / 112 / 170 / 70 | 184 / 56 / 167 / 73 | 130 / 110 / 223 / 17 |
| Accuracy | 62.08% | 73.13% (69.0–76.9) | **73.54%** (69.4–77.3) |
| Sensitivity | 53.33% | **76.67%** (70.9–81.6) | 54.17% (47.8–60.4) |
| Specificity | 70.83% | 69.58% (63.5–75.1) | **92.92%** (89.0–95.5) |
| PPV / NPV | 64.6% / 60.3% | 71.6% / 74.9% | 88.4% / 67.0% |
| F1 | 0.584 | **0.740** | 0.672 |
| Youden's J | 0.242 | 0.462 | **0.471** |

The final baseline reached a **validation AUC-ROC of 0.922** (best epoch); test-set AUC will be computed in Phase 1 of the thesis.

### Scaling data and transfer learning (validation set, used for model selection)

| Model | Training images | Validation n | Accuracy | Glaucoma precision | Glaucoma recall |
|---|---|---|---|---|---|
| CNN from scratch | 9,998 | 1,000 | 87.6% | — | — |
| MobileNetV3Small (fine-tuned) | 16,899 | 929 | **87.1%** | 91.7% | 80.6% |
| ResNet50V2 | 9,998 | 1,000 | 85.3% | — | — |

> These figures come from the validation set that also guided early stopping, so they are optimistic. Evaluation on the held-out test set is implemented in `03_transfer_learning/03_mobilenetv3small_test_final.ipynb` and is part of the next phase.

---

## What the baseline taught us

1. **Sensitivity is too low at the default threshold.** The final baseline misses 110 of 240 glaucoma cases (45.8%), which is unacceptable for screening, even though specificity is 92.9%. The threshold must be selected on validation data for a target sensitivity.
2. **Validation–test gap.** Validation accuracy peaked at 87.0% but test accuracy was 73.5%. Part of this is optimism from selecting the best epoch on validation; a ~13-point drop also suggests domain shift between sources or leakage between splits.
3. **High AUC, low sensitivity.** AUC is threshold-independent, so a model can rank cases well (AUC 0.92) and still miss many at τ = 0.5. Calibration and operating-point selection are essential.
4. **Moderate overfitting.** Training accuracy kept rising to 93.1% while validation loss increased after epoch ~50.
5. **Resolution of the optic disc.** At 220 px the optic disc spans only ~25 px, which may erase cup-to-disc information; higher resolution and disc-centered crops will be tested.
6. **No interpretability or external validation yet.** Addressed in the thesis plan below.

At realistic population prevalence (~3.5%), PPV drops sharply while NPV stays above 98%, another reason to frame the system as screening rather than diagnosis.

---

## Thesis roadmap

The thesis (Aug 2026 – Jun 2027) changes one factor at a time:

| ID | Experiment | Question |
|---|---|---|
| E0 | Baseline CNN, 220 px | Reference point |
| E1 | E0 + broader augmentation and early stopping | How much does diverse augmentation help generalization? |
| E2 | ResNet-50, feature extraction | Does transfer learning help? |
| E3 | ResNet-50, fine-tuning last layers | Does adapting more layers help? |
| E4 | EfficientNet-B0, fine-tuning | Is a lightweight, mobile-ready network competitive? |
| E5 | Best model + CLAHE and/or optic-disc crop, 512 px | Does disc resolution matter? |
| E6 | Best model + calibration and threshold selection | Can we reach sensitivity ≥ 85% with specificity ≥ 80%? |
| E7 | Leave-one-source-out external validation + robustness tests | Does it generalize to an unseen camera or population? |

**Methodological commitments**
- **Data audit first:** perceptual-hash de-duplication across repositories, patient-level splits where metadata allow, source tracking for every image, and quality flags.
- **Statistics:** 95% CIs (Wilson for proportions, bootstrap/DeLong for AUC), DeLong and McNemar tests for model comparison, at least three random seeds per key experiment, and PPV/NPV at 3.5%, 5%, 10% and 50% prevalence.
- **Interpretability:** Grad-CAM maps for correct and incorrect predictions, reviewed for agreement with the optic disc and cup.
- **Reporting:** following TRIPOD+AI (Collins et al., 2024).
- **Optional extension:** a 3D-printed smartphone adapter with a 20 D condensing lens plus a simple web interface, to test robustness on low-cost captures.

---

## Repository structure

```
├── 01_datos/              → Data download, sampling, and conversion to NumPy arrays
├── 02_cnn_propia/         → Baseline CNN trained from scratch (sample sizes, augmentation)
├── 03_transfer_learning/  → ResNet50V2 and MobileNetV3Small
├── 04_evaluacion/         → Predictions, confusion matrices, and model comparison slides
├── 05_modelos/            → Trained baseline CNN models (.keras)
├── requirements.txt
└── LICENSE
```

Each folder has its own `README.md`. Folder and notebook names, and code comments, are in Spanish; all documentation is in English.

---

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/GabrielaVidales/glaucoma-fundus-deep-learning.git
   cd glaucoma-fundus-deep-learning
   pip install -r requirements.txt
   ```
2. The notebooks were written for **Google Colab** and mount Google Drive. Update the project path in the first cells (`os.chdir(...)`) to your own Drive folder.
3. Run the folders in order: `01_datos` → `02_cnn_propia` / `03_transfer_learning` → `04_evaluacion`.
4. To use a trained baseline model without retraining:
   ```python
   from tensorflow import keras
   model = keras.models.load_model("05_modelos/cnn_propia_mejor_modelo_conservador.keras")
   # Input: float32 array of shape (N, 220, 220, 3) scaled to [0, 1]
   prob_glaucoma = model.predict(X / 255.0)
   ```

---

## References

- Collins, G. S., et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, 385, e078378.
- Poveda, J. (2024). *Fundus images dataset for glaucoma* [Data set]. Kaggle. https://www.kaggle.com/datasets/jeremypoveda/fondo-del-ojo-normal-glaucoma-retinopatia
- Tham, Y.-C., Li, X., Wong, T. Y., Quigley, H. A., Aung, T., & Cheng, C.-Y. (2014). Global prevalence of glaucoma and projections of glaucoma burden through 2040: a systematic review and meta-analysis. *Ophthalmology*, 121(11), 2081–2090. https://doi.org/10.1016/j.ophtha.2014.05.013
- Vision Loss Expert Group of the Global Burden of Disease Study & GBD 2019 Blindness and Vision Impairment Collaborators. (2024). Global estimates on the number of people blind or visually impaired by glaucoma: a meta-analysis from 2000 to 2020. *Eye*. https://doi.org/10.1038/s41433-024-02995-5

Please also cite the original repositories included in the Poveda (2024) compilation when using the data.

---

## Author

**Gabriela Yasmin Vidales Ayala**, B.Sc. Engineering Physics (UADY) and medical student (Universidad Anáhuac Mayab)
🔗 [LinkedIn](https://www.linkedin.com/in/gabriela-vidales) · [GitHub](https://github.com/GabrielaVidales) · [ORCID](https://orcid.org/0009-0006-5799-2460)

## License

Code released under the [MIT License](LICENSE). The dataset is subject to the licenses of its original sources.
