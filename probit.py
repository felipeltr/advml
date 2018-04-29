
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

#import scipy


# %%



def probit_activation(x):
    return tf.distributions.Normal(loc=0., scale=1.).cdf(x)

def createProbitModel(sparseAdWeights,nUser):
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

#model = createProbitModel()
#
#model.summary()
#
#from keras.callbacks import *
#
#weights_filename = 'probit.h5'
#
#model.fit(
#    [eventsDf.user_inx,eventsDf.ad_inx],
#    eventsDf.clicked,
#    epochs = 60,
#    shuffle=True,
#    batch_size=5000,
#    callbacks=[
#        EarlyStopping(monitor='loss', patience=2),
#        ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True, save_weights_only=True),
#    ]
#)

#! gsutil cp probit.h5 gs://advml-bucket/

#! gsutil cp gs://advml-bucket/probit.h5 .
    
#model.load_weights('probit.h5')


        

    

# print(smd.getSubSet(4))
# print(smd.getSubSet(40))
# print(smd.getSubSet(420))

#smd.getW()