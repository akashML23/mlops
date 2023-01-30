import numpy as np
import tensorflow as tf
from keras.models import load_model
model = load_model("models/cnn-model.h5", compile = True)
img_width, img_height = 28, 28
def predict_one(image):

    input_shape = (img_width, img_height)
    image = image.reshape(*input_shape)
    img = image/ 255.0
    predictions = model.predict(np.array([img]))
    # print(predictions)

    # Generate arg maxes for predictions
    digit = np.argmax(predictions, axis=1)
    # print(digit)
    return digit

def predict_many(images):
    samples_to_predict = []
    for image in images:
        input_shape = (img_width, img_height)
        image = image.reshape(*input_shape)
        img = image / 255.0
        samples_to_predict.append(img)
    samples_to_predict = np.array(samples_to_predict)
    predictions = model.predict(samples_to_predict)
    digits = np.argmax(predictions, axis=1)
    return digits


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

print(x_test[0])
print(predict_one(x_test[0]))
print(predict_many(x_test[1:5]))