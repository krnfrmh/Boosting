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
