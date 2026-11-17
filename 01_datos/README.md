# 01 · Data

Download and preparation of the fundus images. Run in this order:

| Notebook | What it does |
|---|---|
| `01_descarga_dataset_kaggle.ipynb` | Downloads the public *Fundus images dataset for Glaucoma* (Poveda, 2024) from Kaggle and copies it to Google Drive. |
| `02_creacion_muestra_reducida.ipynb` | Builds smaller class-balanced subsets (2,000 and 5,000 training images) to test the full pipeline quickly before using all the data. |
| `03_generacion_matrices_X_y.ipynb` | Reads each image with OpenCV, converts BGR→RGB, resizes it to 220 × 220 px, and saves `X` (images) and `y` (0 = normal, 1 = glaucoma) as `.npy` arrays for every split. |

Output folders (created in Google Drive, not included in this repository):

```
data_2000train_200val_240test/
data_5000train_500val_240test/
data_Alltrain_Allval_240test/
    train/  X_train.npy  y_train.npy
    val/    X_val.npy    y_val.npy
    test/   X_test.npy   y_test.npy   ← same 480 balanced images in every configuration
```
