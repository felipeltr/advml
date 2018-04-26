
import os.path
from keras.callbacks import *
import numpy as np
import pandas as pd
#import pickle

from importlib import reload

import probit
import smd
import dataloader

reload(probit)
reload(smd)
reload(dataloader)



# %%

sparseAdWeights,adIds = dataloader.loadAdWeightsAndIds()
adIdsRev,convertToAdInx = dataloader.getAdIdConverterFunction(adIds)

eventsDf = pd.read_csv('filtered_events.csv')

uniqUser = np.unique(eventsDf.uuid)
nUser = uniqUser.shape[0]

print(eventsDf.columns)

# %%

model = probit.createProbitModel(sparseAdWeights,nUser)
model.summary()

weights_filename = 'probit.h5'

if os.path.isfile(weights_filename):
    print("loading existing weights")
    model.load_weights(weights_filename)
else:    
    model.fit(
        [eventsDf.user_inx,eventsDf.ad_inx],
        eventsDf.clicked,
        epochs = 60,
        shuffle=True,
        batch_size=5000,
        callbacks=[
            EarlyStopping(monitor='loss', patience=2),
            ModelCheckpoint(weights_filename, monitor='loss', save_best_only=True, save_weights_only=True),
        ]
    )

#%%
    
smd = smd.SubModDivGlobal(sparseAdWeights,model)

import time

t = time.time()
print(smd.getSubSet(1))
print(time.time()-t)