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
 
from flask_ngrok import run_with_ngrok
sess = tf.Session()

#This is a global session and graph
graph = tf.get_default_graph()
set_session(sess)
def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return 
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)
app = Flask(__name__)
# app = Flask(__name__, static_folder="images")
import tensorflow as tf

config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1 , 'CPU': 8}) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)

# On défini des variables globales qui vont nous servir au long d'une session (ouverture du site coté client)
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

from keras.models import model_from_json
model.load_weights("../simulation/vgg_face_weights.h5") 
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-4].output)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
run_with_ngrok(app)
@app.route("/")
def index():
    return render_template("index.html")


# On upload en assignant le path de l'image à destination
@app.route('/upload', methods=['POST'])
def upload_file():
    global destination
    # check if the post request has the file part
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
            destination= os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(destination)
            success = True
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

""" @app.route("/test/",methods=['GET'])
def similarity_zoom():
    global model
    global img1_representation
    global somme
    global sess
    global graph
    t= time()
    L_features=np.load("features.npy")
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

    from tensorflow.python.keras.backend import set_session



    #now where you are calling the model

    with graph.as_default():
        set_session(sess)
        img1_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (path_req)))[0,:]

    cos_list=[]
    for i in range (len(L_features)):
        cosine_similarity = findCosineSimilarity(img1_representation, L_features[i])
        print("Cosine similarity: ",cosine_similarity)
        cos_list.append((cosine_similarity,i))
    L_img_zoom=[L_images[i][j] for i in range (len(L_images)) for j in range(len(L_images[i]))]
    print(L_img_zoom)
    cos_list.sort(key = lambda x: x[0])
    filename1 = L_img_zoom[cos_list[0][1]]
    print('done')
    shutil.copyfile('../simulation/' +filename1 ,"static/files/" +filename1)
    dir = 'images/raw_images'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    shutil.copyfile('../simulation/' +filename1 ,"images/raw_images/" +filename1)
    shutil.copyfile(destination,"images/raw_images/temp."+ destination.split('.')[1] )
    return ({'path_to_file': "static/files/" +filename1,'ressemblance': 1-cos_list[0][0]}) """



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
 # Ici on créé la route retrieval qui va être appelée lorsque l'on appuie sur le btn du front:
@app.route("/retrieval/",methods=['GET'])
def similarity2():
    req_image=destination
    dir_path= "../simulation/lfw_funneled"
    listDir = sorted(os.listdir(dir_path))
    name=listDir
    L_images2=[]
    for d in listDir:
        listFiles = sorted(os.listdir(dir_path+'/'+d))
        if 'desktop.ini' in listFiles:
            listFiles.remove('desktop.ini')
        if(len(L_images2)<200):
            L_images2.append(listFiles)
        else:
            break
    L_features=np.load('features3.npy')
    im_index= np.load('image_index.npy')
    cosinsim=[]
    path=req_image
    with graph.as_default():
        set_session(sess)
        req_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (path)))[0,:]

    # Faire un dot product entre une grosse matrice et un vecteur 
    cosim_2= np.dot(L_features,np.transpose(normalize(req_representation[0,:,:])))

    #for i in range (len(L_images2)):
     #   for j in range(len(L_images2[i])):
      #      cosin=findCosineSimilarity(req_representation, L_features[i][j])   
            #q = normalize(req_representation)
            #D= normalize(L_features[i][j])
            #toto= np.dot(q.T,D)
       #     cosinsim.append((cosin,i,j))
    #cosinsim.sort(key = lambda x: x[0]) #Faire un min
    idx= np.argmax(cosim_2)
    print(cosim_2.shape)
    #path="../simulation/lfw_funneled/" + name[cosinsim[0][1]] +"/"+L_images2[cosinsim[0][1]][cosinsim[0][2]]  
    #filename1=name[cosinsim[0][1]] +"/"+L_images2[cosinsim[0][1]][cosinsim[0][2]]
    #file=L_images2[cosinsim[0][1]][cosinsim[0][2]]
    path="../simulation/lfw_funneled/" + name[im_index[idx,0]] +"/"+L_images2[im_index[idx,0]][im_index[idx,1]]  
    file=L_images2[im_index[idx,0]][im_index[idx,1]]  
    filename1=name[im_index[idx,0]] +"/"+L_images2[im_index[idx,0]][im_index[idx,1]]  
    shutil.copyfile('../simulation/lfw_funneled/' +filename1 ,"static/files/" +file)
    shutil.rmtree('images/raw_images')
    shutil.rmtree('images/aligned_images')
    os.mkdir('images/aligned_images')
    os.mkdir('images/raw_images')
    shutil.copyfile(destination,"images/raw_images/temp."+ destination.split('.')[1] )
    shutil.copyfile("static/files/" +file,"images/raw_images/"+file )
    if 'zoom' in file :
        file= file.replace('_zoom','')
    shutil.copyfile('../simulation/lfw_funneled/'+name[im_index[idx,0]] +"/"+ file,"static/files/"+file )
    path = "static/files/" +file
   # name =  name[cosinsim[0][1]].split('_')
    name =  name[im_index[idx,0]].split('_')
    Name=""
    for i in name:
        Name += i +" "
    return {'path_to_file': path ,'name': Name}



@app.route("/morph/",methods=['GET'])   
def morphing():
    import os
    import shutil
    
    os.system('"python ../stylegan2/align_images.py images/raw_images/ images/aligned_images/"')
    from distutils.dir_util import copy_tree
    # copy subdirectory example
    from_directory = "images/aligned_images/"
    to_directory = "images/aligned_images_B/"
    shutil.rmtree('images/aligned_images_B')
    shutil.rmtree('images/generated_images_no_tiled')
    os.mkdir('images/aligned_images_B')
    os.mkdir('images/generated_images_no_tiled')

    copy_tree(from_directory, to_directory)
    os.system('"python ../stylegan2/project_images.py ' +'images/aligned_images_B/ images/generated_images_no_tiled/ --no-tiled"')
    L= os.listdir('images/generated_images_no_tiled/')
    for i in L:
        if i.split('.')[-1] != "png":
            L.remove(i)
    shutil.copyfile('images/generated_images_no_tiled/' +L[0] ,"static/files/" +L[0])
    print(L)
    return({'path_to_file' : "static/files/" +L[0]})



def similarity(req_image):
  cosinsim=[]
  path=req_image
  req_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (path)))[0,:] 
  for i in range (len(L_images)):
    for j in range(len(L_images[i])):
      cosin=findCosineSimilarity(req_representation, L_features[i][j])    #A changer (deuxieme path)
      cosinsim.append((cosin,i,j))
  cosinsim.sort(key = lambda x: x[0])
  print(cosinsim)
  print(L_images[cosinsim[0][1]])
  print(name[cosinsim[0][1]])
  img = Image.open("/content/drive/MyDrive/projet partagé/simulation/lfw_funneled/" + name[cosinsim[0][1]] +"/"+L_images[cosinsim[0][1]][cosinsim[0][2]])    #A changer
  return img


if __name__ == '__main__':
    app.run()
