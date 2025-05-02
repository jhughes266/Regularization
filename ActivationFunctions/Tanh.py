import numpy as np

class Tanh():
    def __init__(self):
        #init storage list for output values
        self._yStore = 0

    def Forward(self, x):
        #Forward pass caculations
        y = np.tanh(x)
        self._yStore = y
        return y

    def Backward(self, dy):
        #Back pass calculations
        dx = dy * (1 - (self._yStore ** 2))
        return dx