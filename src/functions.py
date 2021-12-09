from keras.applications.imagenet_utils import preprocess_input
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img, save_img
import numpy as np
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
def verifyFace(img1, img2,vgg_face_descriptor,img1_representation):
    global somme
    img2_representation = vgg_face_descriptor.predict(preprocess_image('%s' % (img2)))[0,:]
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)

    return cosine_similarity
