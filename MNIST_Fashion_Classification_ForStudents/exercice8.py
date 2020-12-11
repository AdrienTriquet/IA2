import cv2
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np


def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = np.array(img)
    img = np.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32')
    img = img / 255
    return img


def main():
    filename = "sample_image.png"
    image = load_image(filename)
    model = load_model('Fashion_MNIST_model.h5')
    return model.predict_classes(image)


if __name__ == '__main__':
    print("Result = " + str(main()))
