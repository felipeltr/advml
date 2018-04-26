
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import scipy.sparse

import pickle


# In[ ]:


def getAdIdConverterFunction(adIds):
    # adIdsRev is a dict mapping from ad_id to ad_inx (i.e. the inx of such ad in adIds)
    adIdsRev = {adId: inx for inx, adId in enumerate(adIds)}
    # Vectorized function to convert ad_id into ad_inx
    convertToAdInx = np.vectorize(lambda adId: adIdsRev[adId])
    return (adIdsRev,convertToAdInx)


# In[ ]:


def loadAdWeightsAndIds():
    with open('ads.pickle','rb') as f:
        obj = pickle.load(f)
    return obj


# In[17]:


if __name__ == "__main__":

    # Load ad data and document categories
    adDf = pd.read_csv('promoted_content.csv')
    docCatDf = pd.read_csv('documents_categories.csv')

    # Join both dataframes using document_id
    adCatDf = adDf.merge(docCatDf,left_on='document_id',right_on='document_id',how='left')[['ad_id','category_id','confidence_level']]

    # New column filled with ones
    adCatDf['one_val'] = 1
    # Pivot and transform into a large sparse matrix
    # Each line is one Ad and each column is one category
    pivotedAdVector = adCatDf.pivot(index='ad_id', columns='category_id', values='one_val').fillna(0)
    sparseAdWeights = scipy.sparse.csr_matrix(pivotedAdVector)

    # adIds is an array containing all ad_id
    adIds = np.array(pivotedAdVector.index)

    ## see function definition above
    adIdsRev,convertToAdInx = getAdIdConverterFunction(adIds)


    # In[24]:


    # Save
    with open('ads.pickle','wb') as f:
        pickle.dump((sparseAdWeights,adIds),f)


    # In[4]:


    # Load events and clicks

    clicksDf = pd.read_csv('clicks_train.csv')
    eventsDf = pd.read_csv('events.csv')[['display_id','uuid','timestamp']]


    # count of clicks per user
    countDf=eventsDf.groupby('uuid').agg('count')

    # Threshold of clicks to filter users
    threshold = 5

    # set of user ids to be selected
    uuids = set(countDf[countDf.display_id >= threshold].index)
    # vectorized function to check if a user id should be selected
    shouldBeSelected = np.vectorize(lambda x: x in uuids)


    # Filter eventsDf based on user id
    selectedEventsDf = eventsDf[shouldBeSelected(eventsDf.uuid)]



    # Join events and ads subsets
    eventWithAds = selectedEventsDf.merge(clicksDf,left_on='display_id',right_on='display_id',how='inner')


    # Add inx data

    eventWithAds['ad_inx'] = convertToAdInx(eventWithAds.ad_id)
    eventWithAds['user_inx'],uniqUser = pd.factorize(eventWithAds.uuid)


    # Save
    eventWithAds.to_csv('filtered_events.csv',index=False)

