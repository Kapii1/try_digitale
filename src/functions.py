from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K
import scipy

import numpy as np
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
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

    
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def verifyFace(img1, img2,vgg_face_descriptor,img1_representation):
    global somme
    img2_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (img2)))[0,:]
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)

    return cosine_similarity

def verifyFaceZoom(req_image, L_features):
  try:
    req_image_zoom=DeepFace.detectFace(img_path=req_image,detector_backend="opencv")
    im =Image.fromarray((req_image_zoom * 255).astype(np.uint8))
    im.save(req_image[:-4]+"_zoom.jpg")
    path_req=req_image[:-4]+"_zoom.jpg"
  except:
    path_req=req_image
  cos_list=[]
  img1_representation=vgg_face_descriptor.predict(preprocess_image(path_req))[0,:]
  for i in range (len(L_features)):
        cosine_similarity = findCosineSimilarity(img1_representation, L_features[i])
        print("Cosine similarity: ",cosine_similarity)
        cos_list.append((cosine_similarity,i))
    
  cos_list.sort(key = lambda x: x[0])
  print(cos_list)
  img = Image.open("/content/drive/MyDrive/projet partagé/simulation/"+L_img_zoom[cos_list[0][1]]) 
  return img