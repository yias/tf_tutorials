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

# preprocess the data (concatenate all the rows one after another), since they are numpy arrays
x_train= x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32')/255
x_test= x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).astype('float32')/255

# convert the labels from uint to float32
y_train=y_train.astype('float32')
y_test=y_test.astype('float32')

# reserve 10000 samples for validation
nbSamplesVal=10000

x_val=x_train[-nbSamplesVal:]
y_val=y_train[-nbSamplesVal:]

# remove the last 10000 samples from the training set
x_train=x_train[:-nbSamplesVal]
y_train=y_train[:-nbSamplesVal]


# specify the model configuration (optimizer, loss, metrics)

model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])

# train the model by slicing the data into "batches" of size "batch_size", and repeatedly iterating over the entire dataset for a given number of "epochs"

print('# Fit model of training data')
trainedModel=model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_val,y_val))

# print concentrated performance metrics for all the epochs
print('\nhistory dict: ', trainedModel.history)

# evaluate the model with the testing set
print('\n# Evaluate on the test data')
results=model.evaluate(x_test,y_test,batch_size=128)
print('test loss, test acc: ', results)


plt.figure(1,figsize=(8,5))
plt.plot(trainedModel.epoch,trainedModel.history['loss'],label='loss')
plt.plot(trainedModel.epoch,trainedModel.history['val_loss'],label='loss Val')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.title('loss')


plt.figure(2,figsize=(5,5))
plt.plot(trainedModel.epoch,trainedModel.history['val_sparse_categorical_accuracy'],label='validation sparse categorical accuracy')
plt.plot(trainedModel.epoch,trainedModel.history['sparse_categorical_accuracy'],'--',label='training sparse categorical accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()



"""
for ii in range(20):
	tmp_img=x_train[ii].reshape(28,28)
	imgplot=plt.imshow(tmp_img)
	plt.title("label: " + str (y_train[ii]))
	plt.savefig("mnist_png/out_"+str(ii)+".png")
	plt.show(block=True)/
"""