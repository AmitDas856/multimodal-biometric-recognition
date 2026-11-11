"""An Enhanced Multi-Modal Biometric Authentication (Tkinter + Keras).

Fuses face, fingerprint, iris, ear and palm images into one feature array and
classifies the person with a small CNN. Bachelor's final-year project.

The model and feature code can be imported without opening the GUI (see
tests/); run `python Multimodal.py` to start the desktop app.
"""
import os
import pickle

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

try:  # TensorFlow 2.x / Keras 3
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
    from tensorflow.keras.models import Sequential, model_from_json
    from tensorflow.keras.utils import to_categorical
except ImportError:  # standalone Keras 3
    from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
    from keras.models import Sequential, model_from_json
    from keras.utils import to_categorical

IMAGE_SIZE = (64, 64)
MODALITIES = ("face", "finger", "iris", "ear", "palm")
FEATURES_DIR = "features"
MODEL_DIR = "model"

X = Y = X_train = X_test = y_train = y_test = None
multimodal = None
filename = None
main = text = pathlabel = None  # created in build_gui()


# ---------------------------------------------------------------------------
# Feature extraction (no GUI needed)
# ---------------------------------------------------------------------------
def KLDAFilters():
    """Bank of 16 Gabor kernels used to enhance fingerprint ridges."""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def KLDAProcess(img, filters):
    """Keep the strongest Gabor response at every pixel."""
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


klda_filter = KLDAFilters()


def _read(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.resize(img, IMAGE_SIZE)


def getTestFingerImage(path):
    return KLDAProcess(_read(path), klda_filter)


def getTestFaceImage(path):
    return _read(path)


def getTestIrisImage(path):
    return _read(path)


def getTestEarImage(path):
    return _read(path)


def getTestPalmImage(path):
    return _read(path)


def fuse(face, finger, iris, ear, palm):
    """Stack the five modality arrays into one input (feature-level fusion)."""
    return np.hstack((face, finger, iris, ear, palm))


def load_features(features_dir=FEATURES_DIR):
    """Load the saved per-modality features and labels, fused into X and Y."""
    arrays = {m: np.load(os.path.join(features_dir, f"{m}X.txt.npy")) for m in MODALITIES}
    labels = np.load(os.path.join(features_dir, "faceY.txt.npy"))
    return fuse(*(arrays[m] for m in MODALITIES)), labels


def build_model(input_shape, n_classes):
    """Two Conv2D + MaxPooling blocks, a 256-unit dense layer, softmax output."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def history_accuracy(history):
    """Training accuracy from a Keras history dict (old Keras used 'acc')."""
    return history.get("accuracy", history.get("acc"))


# ---------------------------------------------------------------------------
# GUI actions
# ---------------------------------------------------------------------------
def uploadDataset():
    global filename
    from tkinter import END, filedialog
    text.delete("1.0", END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, str(filename) + " Dataset Loaded\n\n")
    pathlabel.config(text=str(filename) + " Dataset Loaded\n\n")


def preprocessKLDAFeatures():
    global X, Y, X_train, X_test, y_train, y_test
    from tkinter import END
    text.delete("1.0", END)
    X, Y = load_features()
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Preprocessing & KLDA Features extraction process completed\n\n")
    text.insert(END, f"Total images found in dataset : {X.shape[0]}\n\n")
    text.insert(END, "Dataset train and test split details\n\n")
    text.insert(END, f"Total 80% images used for training : {X_train.shape[0]}\n")
    text.insert(END, f"Total 20% images used for testing : {X_test.shape[0]}\n")


def _load_saved_model():
    with open(os.path.join(MODEL_DIR, "multimodal_model.json"), "r") as json_file:
        model = model_from_json(json_file.read())
    for name in ("multimodal.weights.h5", "multimodal_weights.h5"):  # new, then old Keras name
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            model.load_weights(path)
            return model
    raise FileNotFoundError("No saved weights in " + MODEL_DIR)


def trainModel():
    global multimodal
    from tkinter import END
    text.delete("1.0", END)
    if X is None:
        text.insert(END, "Run 'Preprocess & KLDA Features Extraction' first.\n")
        return
    history_path = os.path.join(MODEL_DIR, "multimodal_history.pckl")
    multimodal = None
    if os.path.exists(os.path.join(MODEL_DIR, "multimodal_model.json")):
        try:
            multimodal = _load_saved_model()
        except Exception as exc:  # weights saved by an older Keras version
            print("Saved model could not be loaded, retraining:", exc)
    if multimodal is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        multimodal = build_model(X.shape[1:], Y.shape[1])
        hist = multimodal.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True,
                              verbose=2, validation_data=(X_test, y_test))
        multimodal.save_weights(os.path.join(MODEL_DIR, "multimodal.weights.h5"))
        with open(os.path.join(MODEL_DIR, "multimodal_model.json"), "w") as json_file:
            json_file.write(multimodal.to_json())
        with open(history_path, "wb") as f:
            pickle.dump(hist.history, f)
    multimodal.summary()
    with open(history_path, "rb") as f:
        accuracy = history_accuracy(pickle.load(f))
    text.insert(END, "Multimodal training accuracy : " + str(accuracy[-1]))


def graph():
    import matplotlib.pyplot as plt
    with open(os.path.join(MODEL_DIR, "multimodal_history.pckl"), "rb") as f:
        cnn_data = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel("Iterations/Epoch")
    plt.ylabel("Accuracy")
    plt.plot(history_accuracy(cnn_data), "o-", color="green")
    plt.plot(cnn_data["loss"], "o-", color="orange")
    plt.legend(["Multimodal Training Accuracy", "Multimodal Training Loss"], loc="upper left")
    plt.title("Deep Learning Multimodal Accuracy & Loss Comparison Graph")
    plt.show()


def authentication():
    from tkinter import END, filedialog
    if multimodal is None:
        text.insert(END, "\nTrain or load the model first.\n")
        return
    folder = filedialog.askdirectory(initialdir="testImages")
    parts = [
        np.asarray([getTestFaceImage(folder + "/face.jpg")]),
        np.asarray([getTestFingerImage(folder + "/finger.jpg")]),
        np.asarray([getTestIrisImage(folder + "/iris.jpg")]),
        np.asarray([getTestEarImage(folder + "/ear.jpg")]),
        np.asarray([getTestPalmImage(folder + "/palm.jpg")]),
    ]
    test = fuse(*parts).astype("float32") / 255
    predict = int(np.argmax(multimodal.predict(test))) + 1

    shown = [cv2.resize(cv2.imread(f"{folder}/{m}.jpg"), (128, 128)) for m in MODALITIES]
    images = cv2.hconcat(shown)
    cv2.putText(images, "Multimodal Authenticate Person ID as : " + str(predict), (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Authentication Result", images)
    cv2.waitKey(0)


def close():
    main.destroy()


def build_gui():
    global main, text, pathlabel
    import tkinter
    from tkinter import Button, Label, Scrollbar, Text

    main = tkinter.Tk()
    main.title("An Enhanced Multi-Modal Biometric Authentication")
    main.geometry("1200x1200")

    title = Label(main, text="An Enhanced Multi-Modal Biometric Authentication")
    title.config(bg="DarkGoldenrod1", fg="black", font=("times", 14, "bold"), height=3, width=120)
    title.place(x=5, y=5)

    font1 = ("times", 13, "bold")
    buttons = [
        ("Upload Multi-Modal Dataset", uploadDataset, 100),
        ("Preprocess & KLDA Features Extraction", preprocessKLDAFeatures, 150),
        ("Train Deep Learning Multi-Modal Model", trainModel, 200),
        ("Training Accuracy Graph", graph, 250),
        ("Authentication from Multi-Modal Images", authentication, 300),
        ("Exit", close, 350),
    ]
    for label, command, y in buttons:
        b = Button(main, text=label, command=command)
        b.place(x=50, y=y)
        b.config(font=font1)

    pathlabel = Label(main)
    pathlabel.config(bg="brown", fg="white", font=font1)
    pathlabel.place(x=560, y=100)

    text = Text(main, height=25, width=100)
    scroll = Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=400, y=150)
    text.config(font=("times", 12, "bold"))

    main.config(bg="LightSteelBlue1")
    return main


if __name__ == "__main__":
    build_gui().mainloop()
