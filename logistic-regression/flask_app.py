import os

from PIL import Image
from flask import Flask, jsonify, request
import numpy as np
import pickle
from numpy import asarray

app=Flask(_name_)
model = pickle.load(open("models/logistic_regression_mnist.pkl", 'rb'))


@app.route("/predict", methods=["GET"])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    npimg = asarray(img)
    # npimg = np.fromstring(posted_data, np.uint8)
    npimg = npimg.flatten()
    digit = model.predict([npimg])
    # print(digit)
    # response = pd.Series(digit).to_json(orient='values')
    return jsonify({'message': 'success', 'digit': digit.tolist()})

if _name_ == '_main_':
    # app.debug = True
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)