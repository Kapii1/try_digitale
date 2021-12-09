import os
from uuid import uuid4
import keras
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, request, send_from_directory,jsonify
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img, save_img
from PIL import Image
from time import time
from functions import preprocess_image, verifyFace, findCosineSimilarity
import shutil
from werkzeug.utils import secure_filename
 
__author__ = 'ibininja'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")
import tensorflow as tf


config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)
filename=""
img1_representation= ""
destination = ""
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
somme=0
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    global destination
    # check if the post request has the file part
    print('________________')
    if 'files[]' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
     
    files = request.files.getlist('files[]')
     
    errors = {}
    success = False
     
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            print('______',os.path.join(app.config['UPLOAD_FOLDER'], filename))
            destination= os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(destination)
            success = True
            print(success)
        else:
            errors[file.filename] = 'File type is not allowed'
     
    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 206
        return resp
    if success:
        resp = jsonify({'message' : 'Files successfully uploaded', 'path_to_file': destination})

        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

@app.route("/test/",methods=['GET'])
def similarity_zoom():
    global model
    global img1_representation
    global somme
    t= time()
    print("start")
    req_image_zoom=DeepFace.detectFace(img_path= destination,detector_backend="opencv", enforce_detection=False)
    im =Image.fromarray((req_image_zoom * 255).astype(np.uint8))
    destination_zooom = destination.split('.')[0] + '_zoomed.' +destination.split('.')[-1]
    im.save(destination_zooom)
    path_req=destination_zooom
    cosinsim=[]
    dir_path  = '../simulation/train'
    dirname = os.path.dirname(__file__)
    listDir = sorted(os.listdir(dir_path))
    name=listDir
    L_images=[]
    model.load_weights("../simulation/vgg_face_weights.h5")
    for d in listDir:
        listFiles = sorted(os.listdir(dir_path+'/'+d))
        L_images.append(listFiles)
    img1_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (path_req)))[0,:]
    for i in range (len(L_images)):
        for j in range(len(L_images[i])):
            path_img="../simulation/"+L_images[i][j]     
            cosin=verifyFace(path_req, path_img,vgg_face_descriptor,img1_representation)
            cosinsim.append((cosin,i,j))
    cosinsim.sort(key = lambda x: x[0])
    filename1 = L_images[cosinsim[0][1]][cosinsim[0][2]]
    print('done')
    shutil.copyfile('../simulation/' +filename1 ,"static/files/" +filename1)
    return ({'path_to_file': "static/files/" +filename1,'ressemblance': 1-cosinsim[0][0]})



if __name__ == '__main__':
    app.run(port='4555',debug=True, threaded=True)
