
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import os
from uuid import uuid4
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.backend import set_session
from deepface import DeepFace
from flask import Flask, render_template, request, send_from_directory,jsonify
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential, model_from_json
from tensorflow.keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from time import time
from functions import preprocess_image, verifyFace, findCosineSimilarity
import shutil
from werkzeug.utils import secure_filename
import scipy

from sklearn.preprocessing import normalize
from flask_ngrok import run_with_ngrok

sess = tf.Session()

#This is a global session and graph
graph = tf.get_default_graph()
set_session(sess)
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
model.summary()
from keras.models import model_from_json
model.load_weights("../simulation/vgg_face_weights.h5") 

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
def main():
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-4].output)
    dir_path  = '../simulation/lfw_funneled'   #A changer par le lien du file train
    listDir = sorted(os.listdir(dir_path))#glob.glob(dir_path)
    name=listDir[:200]
    L_images=[]
    for d in listDir:
        listFiles = sorted(os.listdir(dir_path+'/'+d))
        if 'desktop.ini' in listFiles:
            listFiles.remove('desktop.ini')
        if(len(L_images)<200):
            L_images.append(listFiles)
        else:
            break
    L_features=[[]]*200
    L_features_2 = []
    im_index = []
    for i in range (len(L_images)):
        for j in range(len(L_images[i])):
            try:
                if 'zoom' not in L_images[i][j]:
                    img_zoom=DeepFace.detectFace(img_path="../simulation/lfw_funneled/" + name[i] +"/"+L_images[i][j],detector_backend="opencv")
                    im = Image.fromarray((img_zoom * 255).astype(np.uint8))
                    im.save("../simulation/lfw_funneled/" + name[i] +"/"+L_images[i][j][:-4]+"_zoom"+".jpg") 
                    name_img= "../simulation/lfw_funneled/" + name[i] +"/"+L_images[i][j][:-4]+"_zoom"+".jpg"
            except:
                name_img = "../simulation/lfw_funneled/" + name[i] +"/"+L_images[i][j]
            with graph.as_default():
                set_session(sess)
                vec=vgg_face_descriptor.predict(preprocess_image(name_img))[0,:]
                
            
            L_features[i].append(vec)

            L_features_2.append(vec[0,0,:])
            #toto=np.array(L_features_2)
            #print(toto.shape)
            im_index.append([i,j])
    

    toto= normalize(np.array(L_features_2))
    print(toto.shape,np.sum(np.power(toto,2.0),0), np.sum(np.power(toto,2.0),1))
    L_features_2= normalize(np.array(L_features_2))

    np.save('image_index.npy',np.array(im_index))
    np.save('features3.npy', L_features_2, allow_pickle=True)


def extract_features():
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    dir_path= "../simulation/train"
    listDir = sorted(os.listdir(dir_path))
    name=listDir
    L_images=[]
    for d in listDir:
    #read subfolder
        listFiles = sorted(os.listdir(dir_path+'/'+d))
        L_images.append(listFiles)
    for i in range (len(L_images)):
        for j in range(len(L_images[i])):
            try:
                img_zoom=DeepFace.detectFace(img_path="../simulation/"+L_images[i][j],detector_backend="opencv")
                im =Image.fromarray((img_zoom * 255).astype(np.uint8))
                im.save("../simulation/image_zoom1/"+L_images[i][j][:-4]+"_zoom"+".jpg") 
            except:
                img=Image.open("../simulation/"+L_images[i][j])    
                img.save("../simulation/image_zoom1/"+L_images[i][j][:-4]+"_zoom"+".jpg") 
    L_features_zoom=[]
    for i in range (len(L_images)):
        for j in range(len(L_images[i])):
            vec=vgg_face_descriptor.predict(preprocess_image("../simulation/image_zoom1/"+L_images[i][j][:-4]+"_zoom"+".jpg"))[0,:]
            L_features_zoom.append(vec)
            L_img_zoom=[L_images[i][j] for i in range (len(L_images)) for j in range(len(L_images[i]))]
    print(L_features_zoom)
    np.save('features.npy', L_features_zoom, allow_pickle=True)
if __name__=="__main__":
    main()