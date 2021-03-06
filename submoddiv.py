
import numpy as np
import tensorflow as tf
import probit

class SubModDivGlobal():
    def __init__(self,nUsers,wAds,regrModel,alpha=1.0,beta=1.0):
        self.regrModel = regrModel
        
        self.a = alpha
        self.b = beta
        self.wAds = wAds.toarray() # "a_i" in paper
        self.c = np.zeros(wAds.shape[1]) # "c" in section 4.1 of paper
        self.v = np.zeros(wAds.shape[1]) # "v" in section 4.1 of paper
        
#         with tf.device(' ')
        with tf.device('/cpu:0'):
            self.initTensorflowOp()
        
    def initTensorflowOp(self):
        ## Tensorflow code for subsetiteration
        t_prevAdInx = tf.placeholder(tf.int32,shape=(None))
        t_probs = tf.placeholder(tf.float32,shape=(self.wAds.shape[0]))
        t_wAds = tf.placeholder(tf.float32,shape=self.wAds.shape)
        t_w = tf.placeholder(tf.float32,shape=(self.wAds.shape[1]))
        
        t_prevAdSum = tf.constant(1.0)+tf.reduce_sum(tf.gather(t_wAds,t_prevAdInx),axis=0)
        
        t_prevProbSum = tf.reduce_sum(tf.gather(t_probs,t_prevAdInx),axis=0)
        
        t_newAs = tf.log(t_wAds + t_prevAdSum)
        
        t_dotProds = tf.reduce_sum(t_w * t_newAs,axis=1)
        
        t_scores = t_prevProbSum + t_probs
        
        t_prevAdMask = tf.cond(
            tf.equal( tf.shape(t_prevAdInx)[0], tf.constant(0,dtype=tf.int32) ),
            lambda: tf.zeros(self.wAds.shape[0]),
            lambda: tf.reduce_sum(tf.one_hot(t_prevAdInx,tf.constant(self.wAds.shape[0])),axis=0)
        )
        
        
        t_rho = t_dotProds + t_scores - ( t_prevAdMask * tf.constant(1e5))
        
        t_maxInx = tf.argmax(t_rho)
        
        self.t_prevAdInx = t_prevAdInx
        self.t_probs = t_probs
        self.t_wAds = t_wAds
        self.t_w = t_w
        
        self.t_maxInx = t_maxInx
        
          
        ##
        
        
    def getW(self):
        return (self.c + self.a)/(self.v + self.a + self.b)
    
    def resetW(self):
        self.c[:] = 0
        self.v[:] = 0   
    
    def subSetIteration(self,probs,prevAdInx):
        w = self.getW()
        
        prevAdSum = 1+self.wAds[prevAdInx].sum(axis=0)
        prevProbSum = probs[prevAdInx].sum()
       
        newAs = np.log(self.wAds + prevAdSum)  
    
        dotProds = (w * newAs).sum(axis=1)
        
        scores = prevProbSum + probs

        rho = dotProds + scores
        rho[prevAdInx] = -np.inf
        
        maxInx = np.argmax(rho)
        
        return maxInx
        
        
    def getSubSet(self,userInx,n=6):
#         t = time.time()
#        probs = self.regrModel.predict([
#            np.array([userInx]*self.wAds.shape[0]),
#            np.arange(self.wAds.shape[0])
#        ],batch_size=50000).ravel()
        probs = probit.getProbs(
                self.regrModel,
                np.array([userInx]*self.wAds.shape[0]),
                np.arange(self.wAds.shape[0])
            )
#         print(time.time()-t)
        
        currAdSet = np.empty(0,dtype=np.int)
#         currAdSet = np.array([1])
        
        with tf.device('/cpu:0'):
            with tf.Session() as sess:

                while len(currAdSet) < n:
#                     t = time.time()
                    newAd = sess.run(self.t_maxInx,feed_dict={
                        self.t_prevAdInx: currAdSet,
                        self.t_probs: probs,
                        self.t_wAds: self.wAds,
                        self.t_w: self.getW()
                    })
                    currAdSet = np.append(currAdSet,newAd)
        
        # Update v
        self.v += self.wAds[currAdSet].sum(axis=0)
        
        return currAdSet
    
    def registerClick(self,adInx):
        self.c += self.wAds[adInx]
        
        
        
        
        
        
        
        
class SubModDivUser():
    def __init__(self, nUsers, wAds, regrModel, alpha=None):
        self.regrModel = regrModel
        nCat = wAds.shape[1]
        if alpha == None:
            self.a = np.ones(nCat)
        else:
            assert(len(alpha) == nCat)
            self.a = alpha
        self.wAds = wAds.toarray()
        self.c = np.zeros((nUsers, wAds.shape[1])) # "c" in section 4.1 of paper
        
        self.cDefault = self.c.copy()
        
#         with tf.device(' ')
        with tf.device('/gpu:0'):
            self.initTensorflowOp()
        
    def initTensorflowOp(self):
        ## Tensorflow code for subsetiteration
        t_prevAdInx = tf.placeholder(tf.int32,shape=(None))
        t_probs = tf.placeholder(tf.float32,shape=(self.wAds.shape[0]))
        t_wAds = tf.placeholder(tf.float32,shape=self.wAds.shape)
        t_w = tf.placeholder(tf.float32,shape=(self.wAds.shape[1]))
        
        t_prevAdSum = tf.constant(1.0)+tf.reduce_sum(tf.gather(t_wAds,t_prevAdInx),axis=0)
        
        t_prevProbSum = tf.reduce_sum(tf.gather(t_probs,t_prevAdInx),axis=0)
        
        t_newAs = tf.log(t_wAds + t_prevAdSum)
        
        t_dotProds = tf.reduce_sum(t_w * t_newAs,axis=1)
        
        t_scores = t_prevProbSum + t_probs
        
        t_prevAdMask = tf.cond(
            tf.equal( tf.shape(t_prevAdInx)[0], tf.constant(0,dtype=tf.int32) ),
            lambda: tf.zeros(self.wAds.shape[0]),
            lambda: tf.reduce_sum(tf.one_hot(t_prevAdInx,tf.constant(self.wAds.shape[0])),axis=0)
        )
        
        
        t_rho = t_dotProds + t_scores - ( t_prevAdMask * tf.constant(1e5))
        
        t_maxInx = tf.argmax(t_rho)
        
        self.t_prevAdInx = t_prevAdInx
        self.t_probs = t_probs
        self.t_wAds = t_wAds
        self.t_w = t_w
        
        self.t_maxInx = t_maxInx
        
        
    def getW(self, userInx):
        return (self.c[userInx] + self.a) / np.linalg.norm(self.c[userInx] + self.a, ord=1)
    
    def resetW(self):
#        self.c[:,:] = 0
        self.c = self.cDefault
    
    def subSetIteration(self,probs,prevAdInx):
        w = self.getW()
        
        prevAdSum = 1+self.wAds[prevAdInx].sum(axis=0)
        prevProbSum = probs[prevAdInx].sum()
       
        newAs = np.log(self.wAds + prevAdSum)  
    
        dotProds = (w * newAs).sum(axis=1)
        
        scores = prevProbSum + probs

        rho = dotProds + scores
        rho[prevAdInx] = -np.inf
        
        maxInx = np.argmax(rho)
        
        return maxInx
        
        
    def getSubSet(self, userInx, n=6, return_probs = False):
        probs = probit.getProbs(
                self.regrModel,
                np.array([userInx]*self.wAds.shape[0]),
                np.arange(self.wAds.shape[0])
            )
        
        currAdSet = np.empty(0,dtype=np.int)

        
        with tf.device('/gpu:0'):
            with tf.Session() as sess:

                while len(currAdSet) < n:
#                     t = time.time()
                    newAd = sess.run(self.t_maxInx,feed_dict={
                        self.t_prevAdInx: currAdSet,
                        self.t_probs: probs,
                        self.t_wAds: self.wAds,
                        self.t_w: self.getW(userInx)
                    })

                    currAdSet = np.append(currAdSet,newAd)

        
        if return_probs:
            return currAdSet, probs[currAdSet]
        return currAdSet
    
    def registerClick(self, userInx, adInx):
        self.c[userInx] += self.wAds[adInx]
        
    def presetWeights(self,trainDf):
        groupedDf = trainDf.groupby('user_inx')['ad_inx'].apply(np.array)
        for userInx, adInxArr in groupedDf.iteritems():
            self.c[userInx] += self.wAds[adInxArr].sum(axis=0).ravel()
            
        self.cDefault = self.c.copy()
        