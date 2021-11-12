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
