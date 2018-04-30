
import os.path
from keras.callbacks import *
import numpy as np
import pandas as pd
#import pickle

from importlib import reload

import probit
import smd
import dataloader
import os

reload(probit)
reload(smd)
reload(dataloader)

os.chdir('/rmount')

# %%

sparseAdWeights,adIds = dataloader.loadAdWeightsAndIds()
adIdsRev,convertToAdInx = dataloader.getAdIdConverterFunction(adIds)

eventsDf = pd.read_csv('filtered_events.csv')
trainDf, testDf = dataloader.splitTrainTest(eventsDf,0.9)

uniqUser = np.unique(eventsDf.uuid)
nUser = uniqUser.shape[0]

print(eventsDf.columns)

# %%

model = probit.createProbitModelGlobal(sparseAdWeights)
model.summary()

weights_filename = 'probit.h5'

if os.path.isfile(weights_filename):
    print("loading existing weights")
    model.load_weights(weights_filename)
else:    
    probit.trainModelGlobal(model,trainDf,weights_filename)

#%%
    

class Clicker1:
    pass

    
smd = smd.SubModDivUser(nUser,sparseAdWeights,model)

import time

t = time.time()
print(smd.getSubSet(1))
print(time.time()-t)












