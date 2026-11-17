# 03 · Transfer learning

Models built on architectures pretrained on ImageNet and adapted to glaucoma classification.

| Notebook | What it does |
|---|---|
| `01_resnet50v2.ipynb` | **ResNet50V2** backbone with a new classification head, trained on 9,998 images (validation accuracy 85.3%, n = 1,000). |
| `02_mobilenetv3small.ipynb` | **MobileNetV3Small** (lightweight, mobile-ready). Two stages: frozen backbone (feature extraction), then fine-tuning of the last 30 layers with a lower learning rate. Trained on 16,899 images; validation accuracy 87.1%, glaucoma precision 91.7% and recall 80.6% (n = 929). |
| `03_mobilenetv3small_test_final.ipynb` | Final evaluation pipeline on the **independent test set** (480 images): precision, recall, F1, confusion matrix, ROC curve and AUC. |

> Validation metrics are optimistic because the same set guided early stopping. Test-set results from notebook `03` are the ones to report.
