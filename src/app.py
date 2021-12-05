import os
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, request, send_from_directory
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img, save_img
from PIL import Image
from time import time
import shutil
__author__ = 'ibininja'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

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

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    global filename
    global destination
    target = os.path.join(APP_ROOT, 'images/')
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)
    # return send_from_directory("images", filename, as_attachment=True)
    return "Hello world!"

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
    
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def verifyFace(img1, img2):
    global img1_representation
    global vgg_face_descriptor
    global somme
    img2_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (img2)))[0,:]
    t1= time()
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    somme+=time()- t1
    return cosine_similarity

@app.route("/test/",methods=['POST'])
def similarity_zoom():
    global model
    global img1_representation
    global somme
    t= time()
    req_image_zoom=DeepFace.detectFace(img_path= destination,detector_backend="opencv", enforce_detection=False)
    im =Image.fromarray((req_image_zoom * 255).astype(np.uint8))
    destination_zooom = destination.split('.')[0] + '_zoomed.' +destination.split('.')[-1]
    im.save(destination_zooom)
    path_req=destination_zooom
    print(2)
    cosinsim=[]
    dir_path  = './simulation/train'
    listDir = sorted(os.listdir(dir_path))
    name=listDir
    L_images=[]
    model.load_weights("./simulation/vgg_face_weights.h5")
    for d in listDir:
        listFiles = sorted(os.listdir(dir_path+'/'+d))
        L_images.append(listFiles)
    img1_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (path_req)))[0,:]
    for i in range (len(L_images)):
        for j in range(len(L_images[i])):
            path_img="./simulation/"+L_images[i][j]     
            cosin=verifyFace(path_req, path_img)
            cosinsim.append((cosin,i,j))
    print(somme)
    cosinsim.sort(key = lambda x: x[0])
    filename1 = L_images[cosinsim[0][1]][cosinsim[0][2]]
    shutil.copyfile('./simulation/' +filename1 ,"src/images/" +filename1)
    print('time ____________         ' + str(time()-t))
    return render_template("upload.html",image_names=[filename, filename1])

@app.route('/upload/<filename1>')
def download_file(filename1):
    print('____________sending_________________')
    return send_from_directory("images", filename1)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
