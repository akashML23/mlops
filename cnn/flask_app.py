import os

from PIL import Image
from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np

from numpy import asarray

app=Flask(_name_)
model = load_model("models/cnn-model.h5", compile = True)
img_width, img_height = 28, 28


@app.route("/predict", methods=["GET"])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    npimg = asarray(img)
    # npimg = np.fromstring(posted_data, np.uint8)
    input_shape = (img_width, img_height)
    image = npimg.reshape(*input_shape)
    img = image / 255.0
    predictions = model.predict(np.array([img]))
    digit = np.argmax(predictions, axis=1)
    # print(digit)
    # response = pd.Series(digit).to_json(orient='values')
    return jsonify({'message': 'success', 'digit': digit.tolist()})

if _name_ == '_main_':
    # app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)