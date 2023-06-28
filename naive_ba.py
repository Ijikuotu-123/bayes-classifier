import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm   # use this or no 5 below but we will use 5 below for speed sake
from scipy.stats import multivariate_normal as mvn


# each input feature is independent
class NaiveBayes(objet):
    def fit(self, X,Y, Smoothing = 10e-3):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels: 
            current_x = X[Y ==c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis = 0),
                'var': current_x.var(axis =0) + Smoothing
            }
            self.priors[c] = float(len (Y[Y==c])) / len(Y)  # probabilities of each class


    def score (self,X,Y):
        P = self.predict(X)
        return np.mean( P ==Y)

    def predict(X):
        N , D = X.shape
        K = len(self.gaussians)       # number of classes
        p = np.zeroes((N,K))     # helps to collect the probability of each class
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X,mean= mean,cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis =1)

if __name__ == '__main__':
    X, Y = get_data(10000)  # this model can accomodate more data than Knn
    Ntrain = len(Y)/2
    Xtrain,Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model= NaiveBayes()
    to = datetime.now()
    model.fit(Xtrain,Ytrain)
    print('Training time:', datetime.now() - t0)

    to = datetime.now()
    print('Train accuracy:', model.score(Xtrain,Ytrain))
    print('Time to compute train accuracy:',datetime.now() - t0, "Train Size:", len(Ytrain) )
  
    to = datetime.now()
    print('Test accuracy:', model.score(Xtest,Ytest))
    print('Time to compute test accuracy:',datetime.now() - t0, "Test Size:", len(Ytest) )
  



