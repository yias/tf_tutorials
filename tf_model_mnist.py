#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
#from matplotlib import ion, show

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import tensorflow as tf

tf.keras.backend.clear_session()

from tensorflow import keras

from tensorflow.keras import layers

inputs=keras.Input(shape=(784,), name='digits')
x=layers.Dense(64,activation='relu',name='dense_1')(inputs)
x=layers.Dense(64,activation='relu',name='dense_2')(x)
outputs=layers.Dense(10,activation='softmax',name='predictions')(x)

model=keras.Model(inputs=inputs, outputs=outputs)

# load MNIST dataset from tensorflow

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# preprocess 

print(y_train[0])


for ii in range(20):
	tmp_img=x_train[ii].reshape(28,28)
	imgplot=plt.imshow(tmp_img)
	plt.title("label: " + str (y_train[ii]))
	plt.savefig("mnist_png/out_"+str(ii)+".png")
	plt.show(block=True)