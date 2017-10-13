import math
import types
import numpy as np
from random import choice

def identity(x):
    return x

def sigmoid(x):
    return 1. / (1 + np.exp(-1*y))

def tanh(x):
    return np.tanh(y)

def relu(x):
    return x if x > 0.0 else 0.0

class ActivationFunctions:
    def __init__(self):
        self.functions = []
        self.functions.append(identity)
        self.functions.append(sigmoid)
        self.functions.append(tanh)
        self.functions.append(relu)
    def set(self,func=None):
        if func == 'identity':
            return self.functions[0]
        elif func == 'sigmoid':
            return self.functions[1]
        elif func == 'tanh':
            return self.functions[2]
        elif func == 'relu':
            return self.functions[3]
        return choice(self.functions)
