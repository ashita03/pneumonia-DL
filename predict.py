# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:32:35 2020

@author: Ankita
"""

from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('model_vgg19.h5')
img = image.load_img('val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)