from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json
import os
import pandas as pd
import numpy as np

json_file = open("C:/Users/mateo/XD/deteccion/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("C:/Users/mateo/XD/deteccion/facialemotionmodel.h5")

label = ['angry','disgust','fear','happy','neutral','sad','surprise']
def ef(image):
    img = load_img(image,grayscale =  True )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
    
image = 'C:/Users/mateo/XD/deteccion/images/train/sad/42.jpg'

print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)

import matplotlib.pyplot as plt


image = 'C:/Users/mateo/XD/deteccion/images/train/sad/42.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


