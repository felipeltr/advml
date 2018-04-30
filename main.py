
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
trainDf, testDf = dataloader.splitTrainTest(eventsDf,0.8)

uniqUser = np.unique(eventsDf.uuid)
nUser = uniqUser.shape[0]

print(eventsDf.columns)

# %%

model = probit.createProbitModelGlobal(sparseAdWeights)
#model.summary()

weights_filename = 'probit.h5'

if os.path.isfile(weights_filename):
    print("loading existing weights")
    model.load_weights(weights_filename)
else:    
    probit.trainModelGlobal(model,trainDf,weights_filename)
    
    
combinedModel = probit.createProbitModelCombined(sparseAdWeights,nUser)


combined_weights_filename = 'combined_probit.h5'
if os.path.isfile(combined_weights_filename):
    print("loading existing weights")
    combinedModel.load_weights(combined_weights_filename)
else:    
    probit.trainModelCombined(combinedModel,model,trainDf,combined_weights_filename)



#%%
   
smd = submoddiv.SubModDivUser(nUser,sparseAdWeights,model)
smd.presetWeights(trainDf)
    
#%%    

import time
class ClickerBatch:
    def __init__(self, smd, testDf, adWeights, n=6, its=10):
        self.testDf = testDf
        self.smd = smd
        self.wAd = adWeights
        self.normAdWeights = adWeights / adWeights.sum(axis=1)
        self.n = 6
        self.its = its
        
        userInx, userCounts = np.unique(testDf.user_inx,return_counts=True)
        userCounts = userCounts**2
        self.userBatch = np.random.choice(userInx,size=its,p=userCounts/userCounts.sum())
        
        
    def resetMetrics(self):
        self.acc1 = 0
        self.acc3 = 0
        self.probSum = 0.0
        self.catSum = np.zeros(self.wAd.shape[1])
        
        
    def collectMetrics(self, subset, probs, choice):
        sortInx = np.argsort(probs)
        if choice == sortInx[-1]:
            self.acc1 += 1
        if choice in sortInx[-3:]:
            self.acc3 += 1
        self.probSum += probs.sum()
        self.catSum += self.wAd[subset[choice]]
        
    def calculateMetrics(self):
        return {
                'acc1': self.acc1/self.its,
                'acc3': self.acc3/self.its,
                'meanProb': self.probSum/(self.its*self.n),
                'sparsity': np.sum(self.catSum > 0)/self.catSum.A1.shape[0]
                }

        
    def runClicker1(self, smooth=0.1):
        testClickedDf = self.testDf[self.testDf.clicked == 1]
        filteredTestDf = testClickedDf[testClickedDf.user_inx.isin(self.userBatch)]
        
        groupedDf = filteredTestDf.groupby('user_inx')['ad_inx'].apply(np.array)
        
        self.clickedWeightDict = dict(groupedDf.iteritems())
        
        self.resetMetrics()
        
        self.smd.resetW()
        
        for userInx in self.userBatch:
            subset, probs = self.smd.getSubSet(userInx,self.n,return_probs=True)
            
            subsetAdW = self.normAdWeights[subset]
            clickedAdW = self.normAdWeights[self.clickedWeightDict[userInx]].T
            
            cosim = np.max(subsetAdW*clickedAdW+smooth,axis=1).A1

            choice = np.random.choice(self.n,p=cosim/cosim.sum())
            
            smd.registerClick(userInx,subset[choice])
            self.collectMetrics(subset,probs,choice)
        
        return self.calculateMetrics()

    
    def runClicker2(self, smooth=1.0):
        nAds = self.wAd.shape[0]
        
        adInxTest, adCountTest = np.unique(self.testDf.ad_inx,return_counts=True)
        
        adCounts = np.zeros(nAds)
        adCounts[adInxTest] += adCountTest
         
        self.resetMetrics()
        
        self.smd.resetW()
        
        for userInx in self.userBatch:
#            print(userInx)
            subset, probs = self.smd.getSubSet(userInx,self.n,return_probs=True)
            
#            subsetAdW = self.normAdWeights[subset]
#            clickedAdW = self.normAdWeights[self.clickedWeightDict[userInx]].T
            
#            cosim = np.max(subsetAdW*clickedAdW+smooth,axis=1).A1
            subsetCounts = adCounts[subset]+smooth            
#            print(subset)
#            print(subsetCounts)

            choice = np.random.choice(self.n,p=subsetCounts/subsetCounts.sum())
#            print(subset[choice])
#            print()
            
            smd.registerClick(userInx,subset[choice])
            self.collectMetrics(subset,probs,choice)
        
        return self.calculateMetrics()
    
    
    def runRandomClicker(self, smooth=1.0):         
        self.resetMetrics()
        
        self.smd.resetW()
        
        for userInx in self.userBatch:
#            print(userInx)
            subset, probs = self.smd.getSubSet(userInx,self.n,return_probs=True)
            
#            subsetAdW = self.normAdWeights[subset]
#            clickedAdW = self.normAdWeights[self.clickedWeightDict[userInx]].T
            
#            cosim = np.max(subsetAdW*clickedAdW+smooth,axis=1).A1

            choice = np.random.randint(self.n)
#            print(subset[choice])
#            print()
            
            smd.registerClick(userInx,subset[choice])
            self.collectMetrics(subset,probs,choice)
        
        return self.calculateMetrics()



clickers = ClickerBatch(smd, testDf, sparseAdWeights,its=100)
import time
t = time.time()
smd.regrModel = model
print(clickers.runClicker1())
print(clickers.runClicker2())
print(clickers.runRandomClicker())
print(time.time() - t)


smd.regrModel=combinedModel
t = time.time()
print(clickers.runClicker1())
print(clickers.runClicker2())
print(clickers.runRandomClicker())
print(time.time() - t)



#import time
#
#t = time.time()
#print(smd.getSubSet(1))
#print(time.time()-t)












