import matplotlib.pyplot as plt

from DataProcessing.Mnist import Mnist
from Layers.CostLayer import MSECostLayer
from Layers.DenseLayer import DenseLayer
from Graphing.DynamicGrapher import DynamicGrapher
import numpy as np
import sys
np.set_printoptions(linewidth=np.inf)

epochs = 10
examplesPerEpoch = 50000
batchSize = 10
batchesPerEpoch = int(examplesPerEpoch / batchSize)
lr = 0.1
lam = 0.0001
#data set up
mnist = Mnist()
mnist.LoadTrainMatrix()
#grapher set up
grapher = DynamicGrapher("Batch Number", "Loss")


L1 = DenseLayer(784, 100, "Tanh", batchSize, lr)
L2 = DenseLayer(100, 100, "Tanh", batchSize, lr)
L3 = DenseLayer(100, 10, "Sigmoid", batchSize, lr)
L4 = MSECostLayer(batchSize)
imageCounter = 0

for epochNum in range(epochs):
    mnist.Shuffle()
    for batchNum in range(batchesPerEpoch):
        for exampleNum in range(batchSize):
            x, y = mnist.GenerateExample()


            #forward
            temp = L1.Forward(x)
            temp = L2.Forward(temp)
            yhat = L3.Forward(temp)
            L4.Forward(y, yhat)

            #backward
            dyhat = L4.Backward()
            dtemp = L3.Backward(dyhat)
            dtemp = L2.Backward(dtemp)
            L1.Backward(dtemp)

        modelWeights = [L1._w, L2._w, L3._w]
        _, graphData = L4.DisplayInfo(display=True, modelWeights=modelWeights, lam= lam, L2=True)
        #grapher.UpdatePlot(graphData)
        L3.GradientStep(L2=True, lam=lam)
        L2.GradientStep(L2=True, lam=lam)
        L1.GradientStep(L2=True, lam=lam)

        if batchNum % 500 == 0:
            print("###########")
            print("Sum of Second Layer  weight Matrix", np.sum(np.abs(L2._w)))
            print("Number of image generated:", imageCounter)
            print("Batch Cost:", L4._costSum)
            print("###########")
            plt.imshow(np.abs(L2._w), cmap="gray", vmin=0, vmax=0.1)
            plt.colorbar()
            filename = f"plot_{imageCounter}.png"
            save_path = f"../WeightImages/L2/{filename}"
            plt.savefig(save_path)
            plt.close()
            imageCounter += 1


        L4.Reset()
        L3.Reset()
        L2.Reset()
        L1.Reset()
#grapher.FinishPlot()
plt.close()
