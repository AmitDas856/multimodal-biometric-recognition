# Multimodal Biometric Recognition (Python, Keras)

A desktop application (Tkinter GUI) that authenticates a person by combining several biometric traits in one deep-learning model. Bachelor's final-year project (B.Tech CSE, Data Science).

## How it works (from `Multimodal.py`)
- **Modalities:** face, fingerprint, iris, ear and palm images, each resized to 64x64.
- **Feature extraction:** fingerprint images are filtered with a bank of Gabor kernels; features for all modalities are concatenated into one vector per person (feature fusion).
- **Model:** a small CNN in Keras (two Conv2D + MaxPooling blocks, a 256-unit dense layer and a softmax output), trained with Adam and categorical cross-entropy, 80/20 train/test split.
- **GUI buttons:** upload dataset, preprocess and extract features, train the model, plot accuracy/loss, and authenticate a test person from a folder of images.

## Requirements
Python 3.10+, TensorFlow 2.16+ (Keras 3), OpenCV, NumPy, scikit-learn, Matplotlib:

```bash
pip install -r requirements.txt
```

## Not included
The image datasets, extracted feature files (`features/`) and trained weights (`model/`) are not part of this repository.

## Run
```bash
python Multimodal.py
```

## Tests
`python -m pytest` runs smoke tests with random arrays in place of the private biometric images: the Gabor filter bank, the image readers, the five-way fusion, and one training step of the CNN. They run on GitHub Actions on every push. They check that the code works, not the reported accuracy, which comes from the original dataset.

## Changes since the original submission
- Updated to the current TensorFlow/Keras API (the original used Keras 1-era calls such as `Dense(output_dim=...)` that no longer run).
- The palm image was read with the ear reader; it now uses the palm reader.
- The GUI only opens when the file is run, so the model code can be imported and tested.
- Model weights saved by the old Keras version are tried first; if they cannot be loaded, the model is retrained.
