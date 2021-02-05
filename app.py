from flask import Flask, request,jsonify,render_template,flash

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import tensorflow as tf

from tensorflow.keras.models import load_model

import numpy as np

import cv2

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    uploadedfile=request.files['file']
    uploadedfile.save(os.path.join("temp/images", uploadedfile.filename))
    img = load_img(os.path.join("temp/images", uploadedfile.filename))
    arr = img_to_array(img)
    dim1 = arr.shape[0]
    dim2 = arr.shape[1]
    dim3 = arr.shape[2]
    arr = cv2.resize(arr, (28,28))
    dim1 = arr.shape[0]
    dim2 = arr.shape[1]
    dim3 = arr.shape[2]
    arr = np.array(arr, dtype="float") / 255.0
    finalArr=np.array(arr)
    finalArr = tf.expand_dims(finalArr, axis=0)

    model = load_model('model.h5')
    pred = model.predict(finalArr)
    pred=np.argmax(pred)

    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True)
