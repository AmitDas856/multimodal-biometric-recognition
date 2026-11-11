"""Smoke tests that run without the (private) biometric datasets.

They use random arrays in place of real images, so they check the code
paths, not the reported accuracy.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Multimodal as mm  # noqa: E402  (importing must not open the GUI)


def test_gabor_filter_bank():
    assert len(mm.klda_filter) == 16
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    out = mm.KLDAProcess(img, mm.klda_filter)
    assert out.shape == img.shape


def test_image_readers_resize(tmp_path):
    import cv2
    path = str(tmp_path / "x.jpg")
    cv2.imwrite(path, np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8))
    for reader in (mm.getTestFaceImage, mm.getTestFingerImage, mm.getTestIrisImage,
                   mm.getTestEarImage, mm.getTestPalmImage):
        assert reader(path).shape == (64, 64, 3)


def test_fusion_shape():
    parts = [np.zeros((4, 64, 64, 3)) for _ in mm.MODALITIES]
    assert mm.fuse(*parts).shape == (4, 320, 64, 3)


def test_model_trains_and_predicts():
    from tensorflow.keras.utils import to_categorical
    x = np.random.rand(20, 320, 64, 3).astype("float32")
    y = to_categorical(np.random.randint(0, 4, 20), 4)
    model = mm.build_model(x.shape[1:], 4)
    hist = model.fit(x, y, epochs=1, batch_size=10, verbose=0)
    assert mm.history_accuracy(hist.history) is not None
    assert model.predict(x[:2], verbose=0).shape == (2, 4)


def test_old_history_key():
    assert mm.history_accuracy({"acc": [0.5]}) == [0.5]
