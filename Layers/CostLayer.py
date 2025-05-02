import copy

import numpy as np

class MSECostLayer():
    def __init__(self, batchSize):
        self._batchSize = batchSize
        self._yStore = 0
        self._yhatStore = 0
        self._costSum = 0
        self._correctSum = 0

    @property
    def CostSum(self):
        return self._costSum

    @property
    def CorrectSum(self):
        return self._correctSum

    def _IsCorrect(self, y, yhat):
        if np.argmax(y.flatten()) == np.argmax(yhat.flatten()):
            return 1
        return 0

    def Forward(self, y, yhat):
        #calculate
        self._costSum +=  np.sum(0.5 * np.square(y - yhat))
        #check if is correct
        self._correctSum += self._IsCorrect(y, yhat)
        #store
        self._yStore = y
        self._yhatStore = yhat

    def Backward(self):
        dyhat = self._yhatStore - self._yStore
        return dyhat

    #This display may only be used in the testing of the reccurent cell
    def DisplayInfo(self, display=True, modelWeights=None, lam=0, L1=False, L2=False):
        #model weights are stored in a list
        l1NormSum = 0
        l2NormSum = 0
        if modelWeights != None and L1:
            for w in modelWeights:
                l1NormSum += np.sum(np.abs(w))

        if modelWeights != None and L2:
            for w in modelWeights:
                l1NormSum += np.sum(np.square(w))

        self._costSum = self._costSum / self._batchSize
        if display:
            #print("Sum Cost before L1: " + str(self._costSum) )
            costSumBeforeL1 = copy.deepcopy(self._costSum)
            self._costSum += lam * l1NormSum
            self._costSum += lam * l2NormSum
            #print("Sum Cost after L1: " + str(self._costSum) )
            #print("Correct: " + str(self._correctSum) + " out of " + str(self._batchSize))
            return self._costSum, costSumBeforeL1

    def Reset(self):
        self._costSum = 0
        self._correctSum = 0





