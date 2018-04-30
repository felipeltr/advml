
import os.path
from keras.callbacks import *
import numpy as np
import pandas as pd
#import pickle

from importlib import reload

import probit
import submoddiv
import dataloader
import os

reload(probit)
reload(submoddiv)
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
    

class ClickerBatch:
    def __init__(self, smd, testDf, adWeights, its=10):
#        self.testDf = testDf
        self.normAdWeights = adWeights / adWeights.sum(axis=1)
#        self.its = its
        
        userInx, userCounts = np.unique(testDf.user_inx,return_counts=True)
        self.userBatch = np.random.choice(userInx,size=its,p=userCounts/userCounts.sum())
        
        global filteredTestDf
        filteredTestDf = testDf[testDf.user_inx.isin(self.userBatch)]
        
        
        
    def run():
        pass
        
        
        
        

            

    
smd = submoddiv.SubModDivUser(nUser,sparseAdWeights,model)
click1 = ClickerBatch(smd, testDf, sparseAdWeights)

#import time
#
#t = time.time()
#print(smd.getSubSet(1))
#print(time.time()-t)












