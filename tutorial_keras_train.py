#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:22:27 2019

@author: iason
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

data=np.random.random((1000,32))

labels=np.random.random((1000,10))

val_data=np.random.random((100,32))

val_labels=np.random.random((100,10))

# declare datasets from tensorflow to train and validate

dataset=tf.data.Dataset.from_tensor_slices((data,labels))
dataset=dataset.batch(32)

val_dataset=tf.data.Dataset.from_tensor_slices((val_data,val_labels))
val_dataset=val_dataset.batch(32)

model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset,epochs=10,steps_per_epoch=2)

model.evaluate(data,labels, batch_size=32)

result=model.predict(val_data, batch_size=32)
print(result.shape)

