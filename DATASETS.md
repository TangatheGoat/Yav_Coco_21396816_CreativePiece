# Datasets

## Deepfake Faces (Kaggle)

**Name:** Deepfake Faces  
**Source:** https://www.kaggle.com/datasets/mafiosoquasar/deepfake-faces  
**Uploader:** mafiosoquasar (Kaggle)  
**Licence:** The dataset page does not declare an explicit open-source licence. Access is subject to Kaggle's standard terms of service and the platform's dataset-specific usage rules. The dataset is not redistributed in this submission.  
**Version:** As downloaded in April 2026 (no versioned release tag).  
**Use in this project:** A balanced sample of 5,000 images per class (real and fake) was drawn from the full dataset using a fixed random seed (42). These 10,000 images were split into training (3,500 per class), validation (750 per class), and test (750 per class) partitions. The split manifest is recorded at `data/working/manifest.csv`. No raw images from the dataset are included in the submission archive.

The Kaggle dataset aggregates face crops originally derived from the DeepFake Detection Challenge (DFDC), organised by Facebook AI Research. The underlying DFDC data is described in:

> Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., & Ferrer, C. C. (2020). *The DeepFake Detection Challenge (DFDC) dataset*. arXiv preprint arXiv:2006.07397.

---

## FaceForensics++

**Name:** FaceForensics++  
**Source:** https://github.com/ondyari/FaceForensics  
**Citation:**

> Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M. (2019). FaceForensics++: Learning to detect manipulated facial images. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 1–11).

**Licence:** The dataset is available for non-commercial research purposes only under a custom academic licence. See https://github.com/ondyari/FaceForensics/blob/master/LICENSE for the full terms.  
**Version:** As accessed via the official request form (c23 compression variant).  
**Use in this project:** Access to FaceForensics++ was obtained via the official academic request process. The intent was to use a small FF++ subset as a cross-dataset evaluation set, which would have provided evidence of cross-generator generalisation. This was not completed within the project timeline and is flagged as future work. No FF++ images are included in the submission archive.

---

## Ethics statement

Both datasets contain images of real human faces captured from video footage. No images depicting children have been identified in either dataset by the project's original collectors; no additional screening was conducted as part of this project, and this is flagged as a limitation. No attempt has been made to re-identify any individual depicted in the data. No raw face images from either dataset are redistributed in the submission archive or hosted publicly as part of this project. The dataset licences have been respected: the Kaggle dataset is used under Kaggle's terms of service, and the FaceForensics++ data is used strictly for non-commercial academic purposes in accordance with its custom licence. A known fairness limitation is that the demographic composition of the training data has not been audited; model performance may vary across demographic groups in ways that are not captured by the aggregate evaluation metrics reported in `results/evaluation_metrics.json`.