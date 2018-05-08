
#from google.colab import auth
#auth.authenticate_user()

#! ls -al

#project_id = 'leftover-199123'
#!gcloud config set project {project_id}

#! gsutil cp gs://advml-bucket/ads.pickle .
#! gsutil cp gs://advml-bucket/filtered_events.csv .

import numpy as np
import pandas as pd
#import pickle
import tensorflow as tf
import keras
from keras.layers import *
from keras.callbacks import *

#import scipy

def getProbs(model,userInx, adInx):
    if len(model.inputs) == 1:
        probs = model.predict([
            adInx
        ],batch_size=50000).ravel()
    else:
        probs = model.predict([
            userInx,
            adInx
        ],batch_size=50000).ravel()
    
    return probs


def probit_activation(x):
    return tf.distributions.Normal(loc=0., scale=1.).cdf(x)


def createProbitModelGlobal(sparseAdWeights):
    adInxInput = Input(shape=(1,))

    adWeightLayer = Embedding(
        sparseAdWeights.shape[0],
        sparseAdWeights.shape[1],
        input_length=1,
        trainable=False,
        weights=[sparseAdWeights.toarray()]
    )(adInxInput)

    flat_ = Dense(1,name='globalW')(adWeightLayer)

    flat_ = Flatten()(flat_)

    activationLayer = Activation(probit_activation)(flat_)

    model = keras.models.Model(inputs=(adInxInput),outputs=(activationLayer))

    model.compile(loss='mse', optimizer='adam') ## Maybe another optimizer?
    
    return model


def trainModelGlobal(model,trainDf,weights_filename):
    model.fit(
        [trainDf.ad_inx],
        trainDf.clicked,
        epochs = 6,
        shuffle=True,
        batch_size=5000,
        callbacks=[
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True, save_weights_only=True),
        ]
    )
    
    

def createProbitModelUser(sparseAdWeights,nUser):
    userInxInput = Input(shape=(1,))
    adInxInput = Input(shape=(1,))

    adWeightLayer = Embedding(
        sparseAdWeights.shape[0],
        sparseAdWeights.shape[1],
        input_length=1,
        trainable=False,
        weights=[sparseAdWeights.toarray()]
    )(adInxInput)

    userWeightLayer = Embedding(nUser,sparseAdWeights.shape[1],input_length=1)(userInxInput)

    dotLayer = Dot(-1)([adWeightLayer,userWeightLayer])

    flat_ = Flatten()(dotLayer)

    activationLayer = Activation(probit_activation)(flat_)

    model = keras.models.Model(inputs=(userInxInput,adInxInput),outputs=(activationLayer))

    model.compile(loss='mse', optimizer='adam') ## Maybe another optimizer?
    
    return model


def trainModelUser(model,trainDf,weights_filename):
    model.fit(
        [trainDf.user_inx,trainDf.ad_inx],
        trainDf.clicked,
        epochs = 60,
        shuffle=True,
        batch_size=5000,
        callbacks=[
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True, save_weights_only=True),
        ]
    )


def createProbitModelCombined(sparseAdWeights,nUser):
    userInxInput = Input(shape=(1,))
    adInxInput = Input(shape=(1,))

    adWeightLayer = Embedding(
        sparseAdWeights.shape[0],
        sparseAdWeights.shape[1],
        input_length=1,
        trainable=False,
        weights=[sparseAdWeights.toarray()]
    )(adInxInput)

    userWeightLayer = Embedding(
            nUser,
            sparseAdWeights.shape[1],
            input_length=1,
            name='userW'
        )(userInxInput)

    dotLayer = Dot(-1)([adWeightLayer,userWeightLayer])
    
    dense = Dense(1,name='globalW',trainable=False)(adWeightLayer)

    flat_1 = Flatten()(dotLayer)
    flat_2 = Flatten()(dense)
    
    sum_ = Add()([flat_1,flat_2])

    activationLayer = Activation(probit_activation)(sum_)

    model = keras.models.Model(inputs=(userInxInput,adInxInput),outputs=(activationLayer))

    model.compile(loss='mse', optimizer='adam') ## Maybe another optimizer?
    
    return model


def trainModelCombined(model,trainedGlobalModel,trainDf,weights_filename):
    globalW = trainedGlobalModel.get_layer('globalW').get_weights()
    
    model.get_layer('globalW').set_weights(globalW)
    
    model.fit(
        [trainDf.user_inx,trainDf.ad_inx],
        trainDf.clicked,
        epochs = 60,
        shuffle=True,
        batch_size=5000,
        callbacks=[
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True, save_weights_only=True),
        ]
    )
