# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:10:01 2020

@author: Ankita
"""
from keras.layers import Dense, Lambda, Input, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE = [224,244]

train_path = 'chest_xray/train'
valid_path ='chest_xray/val'

vgg = VGG16(input_shape= IMAGE_SIZE + [3], weights = 'imagenet', include_top =False)

for layer in vgg.layers:
    layer.trainable = False
    
folders = glob('chest_xray/train/*')

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = 'softmax')(x)
    
model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

model.compile(
    loss = 'categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0/255)

training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
#%%#Plotting of Loss & Accuracy
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#%%import tensorflow as tf

from keras.models import load_model

model.save('model_vgg19.h5')


 
    
