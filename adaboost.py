import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
  
  def __init__(self, M):
    self.M = M
    
  def fit(self, X, Y):
    self.models = []
    self.alphas = []

    N, D = X.shape
    W = np.ones(N) / N
    
    for m in range(self.M):
      tree = DecisionTreeClassifier(max_depth=1)
      tree.fit(X, Y, sample_weight=W)
      P = tree.predict(X)
      
      # Picking up incorrect predictions
      err = W.dot(P != Y)
      alpha = 0.5*(np.log(1 - err) - np.log(err))
      
      # vectorized form
      W = W * np.exp(-alpha*Y*P)
      # normalize so it sums to 1
      W = W / W.sum() 
      
      self.models.append(tree)
      self.alphas.append(alpha)
      
    def predict(self, X):
      N, _ = X.shape
      FX = np.zeros(N)
      for alpha, tree in zip(self.alphas, self.models):
        FX += alpha*tree.predict(X)
      return np.sign(FX), FX

    def score(self, X, Y):
      # returning accuracy and exponential loss
      P, FX = self.predict(X)
      L = np.exp(-Y*FX).mean()
      return np.mean(P == Y), L
    
if __name__ == '__main__':
    
  X, Y = get_data()
  Y[Y == 0] = -1 # make the targets -1,+1
  Ntrain = int(0.8*len(X))
  Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
  Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
