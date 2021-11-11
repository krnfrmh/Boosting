import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
  
  def __init__(self, M):
    self.M = M
    
