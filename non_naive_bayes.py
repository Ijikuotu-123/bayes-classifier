import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm   # for single dimension guasian
from scipy.stats import multivariate_normal as mvn



# non_naive bayes or bayes classifier: features are dependent. we use full covarience (this uses the tanspose of X)
class NaiveBayes(objet):  # we want to calculate the mean and covariance
    def fit(self, X,Y, Smoothing = 10e-3):
        self.gaussians = dict()    # this contains the mean and covarience
        self.priors = dict()
        labels = set(Y)
        for c in labels: 
            current_x = X[Y ==c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis = 0),
                'cov': np.cov(current_x.T)+np.eye(D)* Smoothing  # cal covarience and not variance
            }
            self.priors[c] = float(len (Y[Y==c])) / len(Y)  # probabilities of each class


    def score (self,X,Y):    # evaluation
        P = self.predict(X)
        return np.mean( P ==Y)

    def predict(X):
        N , D = X.shape
        K = len(self.gaussians)       # number of classes
        p = np.zeroes((N,K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X,mean= mean,cov=cov) + np.log(self.priors[c])
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
  



