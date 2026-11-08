# Multimodal Biometric Recognition (Python, Keras)

A desktop application (Tkinter GUI) that authenticates a person by combining several biometric traits in one deep-learning model. Bachelor's final-year project (B.Tech CSE, Data Science).

## How it works (from `Multimodal.py`)
- **Modalities:** face, fingerprint, iris, ear and palm images, each resized to 64x64.
- **Feature extraction:** fingerprint images are filtered with a bank of Gabor kernels; features for all modalities are concatenated into one vector per person (feature fusion).
- **Model:** a small CNN in Keras (two Conv2D + MaxPooling blocks, a 256-unit dense layer and a softmax output), trained with Adam and categorical cross-entropy, 80/20 train/test split.
- **GUI buttons:** upload dataset, preprocess and extract features, train the model, plot accuracy/loss, and authenticate a test person from a folder of images.

## Requirements
Python, TensorFlow/Keras (the code uses the older Keras API), OpenCV, NumPy, scikit-learn, Matplotlib.

## Not included
The image datasets, extracted feature files (`features/`) and trained weights (`model/`) are not part of this repository.

## Run
```bash
python Multimodal.py
```
