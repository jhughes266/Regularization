import numpy as np
import pandas as pd
import random

class Mnist():
    def __init__(self):
        self._mnistTrainMatrix = None
        self._trainFilePath = r"..\DataSets\mnist_train_reduced.csv"
        self._trainOrder = []
        self._trainingPos = 0

    def _GenerateTrainOrder(self):
        for i in range(self._mnistTrainMatrix.shape[0]):
            self._trainOrder.append(i)

    def _ShuffleTrainOrder(self):
        random.shuffle(self._trainOrder)

    def LoadTrainMatrix(self):
        df = pd.read_csv(self._trainFilePath)
        self._mnistTrainMatrix = df.values
        self._GenerateTrainOrder()

    def GenerateExample(self):
        exampleNum = self._trainOrder[self._trainingPos]
        example = self._mnistTrainMatrix[exampleNum]
        x = ((example[1:].reshape(784, 1)) - (255/2)) * (1/(255/2)) #* (1/ 255)
        y = np.zeros((10, 1))
        y[int(example[0])] = 1
        self._trainingPos += 1
        return x, y

    def TrainDiscriminatorData(self):
        exampleNum = self._trainOrder[self._trainingPos]
        example = self._mnistTrainMatrix[exampleNum]
        x = ((example[1:].reshape(784, 1)) - (255 / 2)) * (1 / (255 / 2))  # * (1/ 255)
        y = np.ones((1, 1))
        self._trainingPos += 1
        return x, y

    def IncrementTrainingPos(self):
        #purely for gan so that we go throught the right amount of trianing examples over the course of an epoch
        self._trainingPos += 1


    def Shuffle(self):
        self._ShuffleTrainOrder()
        self._trainingPos = 0





