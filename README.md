# Fall-Detector

**Note:** Code reference for MHI and CNN training: https://medium.com/diving-in-deep/fall-detection-with-pytorch-b4f19be71e80

## Preprocessing

1. Download and unzip datasets.
1. Confirm annotations included the start and stop frames at the beginning of the `.txt`.
1. Excludes dataset without annotations.
1. Run `fallDetectorV0\preprocess.py`.
    - this creates the MHIs and the contour feature csv files
    - PCA mean and eigenvalues are saved in `fallDetectorV0\pcaEig.joblib`

## Training

### SVM

1. Run `fallDetectorV0\training.py`
    - sklearn SVM object is saved in `fallDetectorV0\svm.joblib`

### CNN

1. Run `fallDetectorV0\training.ipynb`.
    - Weights are saved to `trained_model\fdnet.pt`

## Online fall detection

### SVM

1. Run `fallDetectorV0\onlineFallDetector.py`

### CNN

Not implemented
