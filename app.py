import json
import os
import sys

# Flask
from PIL import Image
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from keras_applications import imagenet_utils
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from keras import backend as K

K.clear_session()
# Some utilites
import numpy as np
from util import base64_to_pil

# Declare a flask app
app = Flask(__name__)

TARGET_CLASSES = {
    0: "Normal",
    1: "Tuberculosis"

}

print('Model loaded. Check http://127.0.0.1:5000/')

graph = tf.get_default_graph()
# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function()  # Necessary
print('Model loaded. Start serving...')


def model_predict(img):
    # Preprocessing the image
    img = prepare_image(img)
    preds = []
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        preds = model.predict(img)

    return preds


def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    im = image.resize((96, 96), Image.ANTIALIAS)
    doc = keras.preprocessing.image.img_to_array(im)  # -> numpy array

    doc = np.expand_dims(doc, axis=0)

    return doc


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = {"Message": "Cannot process"}
    if request.method == 'POST':

        try:

            # Get the image from post request
            img = base64_to_pil(request.json)

            # Make prediction
            preds = model_predict(img)
            index_min = np.argmin(preds[0])

            return jsonify(result=TARGET_CLASSES[index_min], probability=json.dumps(str(preds[0][index_min])))

        except Exception as ex:
            return None

    return None


if __name__ == '__main__':
    # app.run(port=5000, threaded=False, debug=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
