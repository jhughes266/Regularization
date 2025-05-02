import numpy as np

class Sigmoid():
    def __init__(self):
        #init storage list for output values
        self._yStore = 0

    def Forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self._yStore = y
        return y

    def Backward(self, dy):
        dx = dy * (self._yStore * (1 - self._yStore))
        return dx


