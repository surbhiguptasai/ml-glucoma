from flask import Flask, request,jsonify,render_template,flash

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import tensorflow as tf

from tensorflow.keras.models import load_model

import numpy as np

import cv2
import boto3
from io import BytesIO
import matplotlib.image as mpimg
import io

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    AWS_ACCESS_KEY_ID=os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY=os.environ.get('AWS_SECRET_ACCESS_KEY')

    uploadedfile=request.files['file']
    s3_client = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    S3_BUCKET="diabeticretinopathy1"
    s3=boto3.resource('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,
                   aws_secret_access_key=AWS_SECRET_ACCESS_KEY).Bucket(S3_BUCKET)
    #
    response = s3_client.upload_fileobj(uploadedfile, S3_BUCKET, uploadedfile.filename,ExtraArgs={'ACL': 'public-read'})
    flash(response)

    # uploadedfile.save(os.path.join("temp/images", uploadedfile.filename))
    # img = load_img("https://diabeticretinopathy1.s3.amazonaws.com/13_right.jpeg")
    object = s3.Object(uploadedfile.filename)


    # img = load_img(object.get()['Body'].read())
    arr = mpimg.imread(BytesIO(object.get()['Body'].read()), 'jpeg')


    # img = load_img(io.BytesIO(object.get()['Body'].read()))
    # arr = img_to_array(img)
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
