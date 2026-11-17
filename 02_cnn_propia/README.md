# 02 · Baseline CNN (trained from scratch)

A convolutional neural network designed from scratch: three blocks of Conv2D (32/64/128 filters) + BatchNormalization + MaxPooling2D, followed by GlobalAveragePooling2D, Dense(128) with Dropout(0.5), and a sigmoid output (110,785 parameters). No pretrained weights are used.

| Notebook | What it tests |
|---|---|
| `01_cnn_muestra_2000.ipynb` | First version on the baseline subset (2,000 images per class; 4,000 training, 400 validation) to validate the pipeline end to end. |
| `02_cnn_muestra_5000.ipynb` | CNN trained with more data (~5,000 images per class; 9,998 training, 1,000 validation; validation accuracy 87.6%), including a confusion matrix. |
| `03_cnn_dataset_completo.ipynb` | Same model trained on the full training set. |
| `04_cnn_data_augmentation_v1.ipynb` | Adds real-time augmentation with `ImageDataGenerator` (horizontal flips; rotation, shift and zoom options left disabled), 50 epochs. |
| `05_cnn_data_augmentation_v2.ipynb` | Second iteration with static horizontal-flip augmentation (training set doubled to 8,000 images); this is the final baseline (test accuracy 73.54%, specificity 92.92%, sensitivity 54.17%; validation AUC 0.922). |

Trained models are saved in `../05_modelos/` with the prefix `cnn_propia_`.
